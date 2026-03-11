# HN Sentiment CLI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI that fetches a HN thread, generates an Amazon-style sentiment summary via Claude, and supports RAG-powered follow-up chat.

**Architecture:** Single Python package `hn_sentiment` with one file per concern (fetcher, sampler, embedder, retriever, analyzer, renderer, cli). Comments are fetched from Algolia HN API, embedded into ChromaDB in-memory, summarized via Claude, then the user can chat with full RAG context.

**Tech Stack:** Python 3.11+, click, rich, httpx, chromadb (EphemeralClient), anthropic, pytest, pytest-mock

---

## File Structure

```
hacker-news-sentiment/
├── pyproject.toml               # Package config + dependencies + CLI entry point
├── hn_sentiment/
│   ├── __init__.py              # Empty
│   ├── cli.py                   # Click entry point, URL validation, orchestration
│   ├── fetcher.py               # Algolia API call, recursive comment flattening
│   ├── sampler.py               # Top-voted selection for summary
│   ├── embedder.py              # ChromaDB EphemeralClient, embed all comments
│   ├── retriever.py             # Query ChromaDB, return top-K comments
│   ├── analyzer.py              # Claude API: summarize() and chat()
│   └── renderer.py              # Rich panels, stats, summary display, chat loop
└── tests/
    ├── __init__.py
    ├── test_fetcher.py
    ├── test_sampler.py
    ├── test_embedder.py
    ├── test_retriever.py
    └── test_analyzer.py
```

---

## Chunk 1: Project Setup + Fetcher + Sampler

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `hn_sentiment/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "hn-sentiment"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "rich>=13.7",
    "httpx>=0.27",
    "chromadb>=0.5",
    "anthropic>=0.30",
]

[project.scripts]
hn-sentiment = "hn_sentiment.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["hn_sentiment*"]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]
```

- [ ] **Step 2: Create package and test init files**

```python
# hn_sentiment/__init__.py
# (empty)
```

```python
# tests/__init__.py
# (empty)
```

- [ ] **Step 3: Install in editable mode**

Run: `pip install -e ".[dev]"`
Expected: Package installs successfully, `hn-sentiment` command available

- [ ] **Step 4: Verify install**

Run: `hn-sentiment --help`
Expected: "Error: Missing argument 'URL'" or similar (no crash)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml hn_sentiment/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure"
```

---

### Task 2: fetcher.py — Algolia API + comment tree flattening

**Files:**
- Create: `hn_sentiment/fetcher.py`
- Create: `tests/test_fetcher.py`

The Algolia endpoint `GET https://hn.algolia.com/api/v1/items/{id}` returns:
```json
{
  "id": 12345,
  "type": "story",
  "by": "author",
  "title": "Post Title",
  "score": 100,
  "children": [
    {
      "id": 12346,
      "type": "comment",
      "by": "user1",
      "text": "Comment text",
      "score": 5,
      "children": [
        {
          "id": 12347,
          "type": "comment",
          "by": "user2",
          "text": "Reply text",
          "score": 1,
          "children": []
        }
      ]
    }
  ]
}
```

Comments with `text: null` or `type != "comment"` are filtered out.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fetcher.py
import pytest
from unittest.mock import patch, MagicMock
from hn_sentiment.fetcher import fetch_thread, flatten_comments, extract_item_id


def test_extract_item_id_standard_url():
    url = "https://news.ycombinator.com/item?id=12345"
    assert extract_item_id(url) == "12345"


def test_extract_item_id_invalid_url():
    with pytest.raises(ValueError, match="Invalid HN URL"):
        extract_item_id("https://google.com")


def test_extract_item_id_missing_id():
    with pytest.raises(ValueError, match="Invalid HN URL"):
        extract_item_id("https://news.ycombinator.com/news")


def test_flatten_comments_empty():
    assert flatten_comments([]) == []


def test_flatten_comments_single():
    children = [{"id": 1, "type": "comment", "by": "user", "text": "hello", "score": 3, "children": []}]
    result = flatten_comments(children)
    assert len(result) == 1
    assert result[0] == {"id": "1", "text": "hello", "author": "user", "points": 3, "depth": 0}


def test_flatten_comments_nested():
    children = [
        {
            "id": 1, "type": "comment", "by": "user1", "text": "top", "score": 5,
            "children": [
                {"id": 2, "type": "comment", "by": "user2", "text": "reply", "score": 1, "children": []}
            ]
        }
    ]
    result = flatten_comments(children)
    assert len(result) == 2
    assert result[0]["depth"] == 0
    assert result[1]["depth"] == 1


def test_flatten_comments_filters_null_text():
    children = [
        {"id": 1, "type": "comment", "by": "user", "text": None, "score": 1, "children": []},
        {"id": 2, "type": "comment", "by": "user", "text": "valid", "score": 1, "children": []},
    ]
    result = flatten_comments(children)
    assert len(result) == 1
    assert result[0]["text"] == "valid"


def test_flatten_comments_filters_non_comments():
    children = [
        {"id": 1, "type": "pollopt", "by": "user", "text": "option", "score": 1, "children": []},
        {"id": 2, "type": "comment", "by": "user", "text": "real comment", "score": 1, "children": []},
    ]
    result = flatten_comments(children)
    assert len(result) == 1


def test_fetch_thread_returns_title_and_comments(mocker):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": 12345,
        "type": "story",
        "by": "author",
        "title": "Test Post",
        "score": 100,
        "children": [
            {"id": 1, "type": "comment", "by": "user1", "text": "A comment", "score": 3, "children": []}
        ]
    }
    mocker.patch("httpx.get", return_value=mock_response)

    title, comments = fetch_thread("12345", max_comments=500)
    assert title == "Test Post"
    assert len(comments) == 1
    assert comments[0]["text"] == "A comment"


def test_fetch_thread_raises_on_http_error(mocker):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mocker.patch("httpx.get", return_value=mock_response)

    with pytest.raises(Exception):
        fetch_thread("99999", max_comments=500)


def test_fetch_thread_raises_on_null_title(mocker):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": 12345, "type": "story", "title": None, "children": []}
    mocker.patch("httpx.get", return_value=mock_response)

    with pytest.raises(ValueError, match="not found"):
        fetch_thread("12345", max_comments=500)


def test_fetch_thread_caps_comments(mocker):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": 1, "type": "story", "title": "T", "score": 10,
        "children": [
            {"id": i, "type": "comment", "by": "u", "text": f"c{i}", "score": 1, "children": []}
            for i in range(10)
        ]
    }
    mocker.patch("httpx.get", return_value=mock_response)

    _, comments = fetch_thread("1", max_comments=3)
    assert len(comments) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fetcher.py -v`
Expected: `ImportError` or `ModuleNotFoundError` — fetcher.py doesn't exist yet

- [ ] **Step 3: Implement `hn_sentiment/fetcher.py`**

```python
# hn_sentiment/fetcher.py
import re
from urllib.parse import urlparse, parse_qs
import httpx

ALGOLIA_BASE = "https://hn.algolia.com/api/v1/items"


def extract_item_id(url: str) -> str:
    """Extract HN item ID from a news.ycombinator.com URL."""
    parsed = urlparse(url)
    if parsed.netloc not in ("news.ycombinator.com", "ycombinator.com"):
        raise ValueError(f"Invalid HN URL: {url}")
    params = parse_qs(parsed.query)
    if "id" not in params:
        raise ValueError(f"Invalid HN URL (missing id param): {url}")
    return params["id"][0]


def flatten_comments(children: list, depth: int = 0) -> list[dict]:
    """Recursively flatten a nested comment tree into a list of comment dicts."""
    result = []
    for child in children:
        if child.get("type") != "comment":
            continue
        text = child.get("text")
        if not text:
            continue
        result.append({
            "id": str(child["id"]),
            "text": text,
            "author": child.get("by", "unknown"),
            "points": child.get("score") or 0,
            "depth": depth,
        })
        result.extend(flatten_comments(child.get("children") or [], depth + 1))
    return result


def fetch_thread(item_id: str, max_comments: int) -> tuple[str, list[dict]]:
    """
    Fetch a HN thread from Algolia and return (title, comments).
    Comments are flattened from the full recursive tree and capped at max_comments.
    """
    response = httpx.get(f"{ALGOLIA_BASE}/{item_id}", timeout=30)
    response.raise_for_status()
    data = response.json()

    title = data.get("title")
    if not title:
        raise ValueError(f"Post {item_id} not found or has no title")

    comments = flatten_comments(data.get("children") or [])
    return title, comments[:max_comments]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetcher.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add hn_sentiment/fetcher.py tests/test_fetcher.py
git commit -m "feat: add fetcher with Algolia API and comment flattening"
```

---

### Task 3: sampler.py — top-voted comment selection

**Files:**
- Create: `hn_sentiment/sampler.py`
- Create: `tests/test_sampler.py`

Sampler picks the top N comments by points for the summary. All comments are still embedded for RAG.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sampler.py
import pytest
from hn_sentiment.sampler import sample_for_summary

def make_comment(id, points, text="text"):
    return {"id": str(id), "text": text, "author": "user", "points": points, "depth": 0}


def test_returns_top_voted():
    comments = [make_comment(i, i) for i in range(10)]  # points 0..9
    result = sample_for_summary(comments, top_n=3)
    assert len(result) == 3
    points = [c["points"] for c in result]
    assert points == sorted(points, reverse=True)


def test_returns_all_when_fewer_than_top_n():
    comments = [make_comment(i, i) for i in range(5)]
    result = sample_for_summary(comments, top_n=100)
    assert len(result) == 5


def test_empty_returns_empty():
    assert sample_for_summary([], top_n=50) == []


def test_default_top_n_is_100():
    comments = [make_comment(i, i) for i in range(200)]
    result = sample_for_summary(comments)
    assert len(result) == 100


def test_preserves_comment_fields():
    comments = [make_comment(1, 10, text="hello")]
    result = sample_for_summary(comments, top_n=1)
    assert result[0]["text"] == "hello"
    assert result[0]["points"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sampler.py -v`
Expected: `ImportError` — sampler.py doesn't exist yet

- [ ] **Step 3: Implement `hn_sentiment/sampler.py`**

```python
# hn_sentiment/sampler.py


def sample_for_summary(comments: list[dict], top_n: int = 100) -> list[dict]:
    """
    Return the top_n comments sorted by points descending.
    Used to select high-signal comments for the initial summary.
    """
    return sorted(comments, key=lambda c: c["points"], reverse=True)[:top_n]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sampler.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add hn_sentiment/sampler.py tests/test_sampler.py
git commit -m "feat: add sampler for top-voted comment selection"
```

---

## Chunk 2: Embedder + Retriever

### Task 4: embedder.py — ChromaDB in-memory embedding

**Files:**
- Create: `hn_sentiment/embedder.py`
- Create: `tests/test_embedder.py`

Uses `chromadb.EphemeralClient()` so nothing is written to disk. The default embedding function (`all-MiniLM-L6-v2`) is used automatically by ChromaDB — no explicit embedding config needed.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embedder.py
import pytest
from hn_sentiment.embedder import build_collection


def make_comments(n):
    return [
        {"id": str(i), "text": f"Comment number {i} about topic", "author": f"user{i}", "points": i, "depth": 0}
        for i in range(n)
    ]


def test_collection_contains_all_comments():
    comments = make_comments(5)
    collection = build_collection(comments)
    assert collection.count() == 5


def test_collection_handles_empty():
    collection = build_collection([])
    assert collection.count() == 0


def test_collection_stores_metadata():
    comments = [{"id": "42", "text": "interesting point", "author": "alice", "points": 7, "depth": 1}]
    collection = build_collection(comments)
    result = collection.get(ids=["42"], include=["metadatas"])
    meta = result["metadatas"][0]
    assert meta["author"] == "alice"
    assert meta["points"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embedder.py -v`
Expected: `ImportError` — embedder.py doesn't exist yet

- [ ] **Step 3: Implement `hn_sentiment/embedder.py`**

```python
# hn_sentiment/embedder.py
import chromadb


def build_collection(comments: list[dict]):
    """
    Embed all comments into a ChromaDB in-memory collection.
    Returns the collection for use by the retriever.
    The default embedding function (all-MiniLM-L6-v2) is used automatically.
    """
    client = chromadb.EphemeralClient()
    collection = client.create_collection("hn_comments")

    if not comments:
        return collection

    collection.add(
        ids=[c["id"] for c in comments],
        documents=[c["text"] for c in comments],
        metadatas=[
            {"author": c["author"], "points": c["points"], "depth": c["depth"]}
            for c in comments
        ],
    )
    return collection
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embedder.py -v`
Expected: All tests PASS (note: first run downloads ~80MB model — this is expected)

- [ ] **Step 5: Commit**

```bash
git add hn_sentiment/embedder.py tests/test_embedder.py
git commit -m "feat: add embedder using ChromaDB ephemeral in-memory collection"
```

---

### Task 5: retriever.py — semantic comment retrieval

**Files:**
- Create: `hn_sentiment/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retriever.py
import pytest
from hn_sentiment.embedder import build_collection
from hn_sentiment.retriever import retrieve_relevant


def make_comments():
    return [
        {"id": "1", "text": "Python is a great programming language", "author": "a", "points": 5, "depth": 0},
        {"id": "2", "text": "JavaScript frameworks are overwhelming", "author": "b", "points": 3, "depth": 0},
        {"id": "3", "text": "Rust has excellent memory safety guarantees", "author": "c", "points": 8, "depth": 0},
        {"id": "4", "text": "Docker containers simplify deployment", "author": "d", "points": 2, "depth": 0},
        {"id": "5", "text": "Machine learning requires lots of data", "author": "e", "points": 6, "depth": 0},
    ]


def test_retrieve_returns_results():
    collection = build_collection(make_comments())
    results = retrieve_relevant(collection, "programming languages", top_k=2)
    assert len(results) == 2


def test_retrieve_returns_comment_dicts():
    collection = build_collection(make_comments())
    results = retrieve_relevant(collection, "memory safety", top_k=1)
    assert "text" in results[0]
    assert "author" in results[0]
    assert "points" in results[0]


def test_retrieve_caps_at_collection_size():
    collection = build_collection(make_comments())
    results = retrieve_relevant(collection, "anything", top_k=100)
    assert len(results) <= 5


def test_retrieve_empty_collection():
    collection = build_collection([])
    results = retrieve_relevant(collection, "query", top_k=5)
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`
Expected: `ImportError` — retriever.py doesn't exist yet

- [ ] **Step 3: Implement `hn_sentiment/retriever.py`**

```python
# hn_sentiment/retriever.py


def retrieve_relevant(collection, query: str, top_k: int = 10) -> list[dict]:
    """
    Embed query and return top_k most semantically relevant comments from the collection.
    Returns list of comment dicts with keys: id, text, author, points, depth.
    """
    if collection.count() == 0:
        return []

    k = min(top_k, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )

    comments = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        comments.append({
            "id": results["ids"][0][i],
            "text": doc,
            "author": meta["author"],
            "points": meta["points"],
            "depth": meta["depth"],
        })
    return comments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add hn_sentiment/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever for semantic comment lookup"
```

---

## Chunk 3: Analyzer + Renderer + CLI

### Task 6: analyzer.py — Claude API integration

**Files:**
- Create: `hn_sentiment/analyzer.py`
- Create: `tests/test_analyzer.py`

`summarize()` sends sampled comments to Claude and returns an Amazon-style paragraph.
`chat()` sends retrieved comments + conversation history and returns Claude's reply.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analyzer.py
import pytest
from unittest.mock import MagicMock, patch
from hn_sentiment.analyzer import Analyzer


SAMPLE_COMMENTS = [
    {"id": "1", "text": "I love this product", "author": "alice", "points": 5, "depth": 0},
    {"id": "2", "text": "It has some issues", "author": "bob", "points": 3, "depth": 0},
]


def make_mock_client(response_text: str):
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=response_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    return mock_client


def test_summarize_calls_claude(mocker):
    mock_client = make_mock_client("Customers generally like this.")
    mocker.patch("anthropic.Anthropic", return_value=mock_client)

    analyzer = Analyzer(api_key="test-key")
    result = analyzer.summarize("Test Post", SAMPLE_COMMENTS)

    assert mock_client.messages.create.called
    assert result == "Customers generally like this."


def test_summarize_includes_post_title_in_prompt(mocker):
    mock_client = make_mock_client("Summary here.")
    mocker.patch("anthropic.Anthropic", return_value=mock_client)

    analyzer = Analyzer(api_key="test-key")
    analyzer.summarize("My HN Post Title", SAMPLE_COMMENTS)

    call_kwargs = mock_client.messages.create.call_args
    prompt_text = str(call_kwargs)
    assert "My HN Post Title" in prompt_text


def test_chat_appends_to_history(mocker):
    mock_client = make_mock_client("Here is my answer.")
    mocker.patch("anthropic.Anthropic", return_value=mock_client)

    analyzer = Analyzer(api_key="test-key")
    history = []
    reply = analyzer.chat("Tell me more", SAMPLE_COMMENTS, history)

    assert reply == "Here is my answer."
    assert len(history) == 2  # user message + assistant reply
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_chat_passes_full_history_to_claude(mocker):
    mock_client = make_mock_client("Follow-up answer.")
    mocker.patch("anthropic.Anthropic", return_value=mock_client)

    analyzer = Analyzer(api_key="test-key")
    history = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    analyzer.chat("Second question", SAMPLE_COMMENTS, history)

    call_kwargs = mock_client.messages.create.call_args[1]
    messages_sent = call_kwargs["messages"]
    assert len(messages_sent) == 3  # 2 history + 1 new user message


def test_chat_includes_retrieved_comments_in_context(mocker):
    mock_client = make_mock_client("Answer with context.")
    mocker.patch("anthropic.Anthropic", return_value=mock_client)

    analyzer = Analyzer(api_key="test-key")
    history = []
    analyzer.chat("What do people think?", SAMPLE_COMMENTS, history)

    call_kwargs = mock_client.messages.create.call_args[1]
    messages_sent = call_kwargs["messages"]
    last_user_message = messages_sent[-1]["content"]
    assert "I love this product" in last_user_message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analyzer.py -v`
Expected: `ImportError` — analyzer.py doesn't exist yet

- [ ] **Step 3: Implement `hn_sentiment/analyzer.py`**

```python
# hn_sentiment/analyzer.py
import anthropic

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024


def _format_comments(comments: list[dict]) -> str:
    lines = []
    for c in comments:
        lines.append(f"[{c['author']} | points: {c['points']}]: {c['text']}")
    return "\n\n".join(lines)


class Analyzer:
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    def summarize(self, post_title: str, comments: list[dict]) -> str:
        """
        Generate an Amazon-style natural language summary of the HN thread.
        Returns a single paragraph describing the main views and sentiment.
        """
        formatted = _format_comments(comments)
        prompt = (
            f'Below are comments from a Hacker News post titled "{post_title}".\n\n'
            f"{formatted}\n\n"
            "Write a natural language summary of the main views expressed in these comments, "
            "similar to how Amazon summarizes product reviews. "
            "Describe what commenters generally agree on, what they are divided about, "
            "and highlight any notable perspectives. "
            "Write in third person (e.g. 'Commenters say...', 'Opinions are split on...'). "
            "Keep it to 3-5 sentences."
        )
        message = self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def chat(self, user_question: str, retrieved_comments: list[dict], history: list[dict]) -> str:
        """
        Answer a follow-up question using retrieved comments as context.
        Appends the user message and assistant reply to history in place.
        Returns the assistant's reply text.
        """
        context = _format_comments(retrieved_comments)
        user_content = (
            f"Relevant comments from the thread:\n\n{context}\n\n"
            f"Question: {user_question}"
        )
        history.append({"role": "user", "content": user_content})

        message = self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You are a helpful assistant analyzing a Hacker News thread. "
                "Answer questions based on the comments provided. "
                "If asked to show a specific comment, quote it directly. "
                "Be concise and reference specific commenters when relevant."
            ),
            messages=history,
        )
        reply = message.content[0].text
        history.append({"role": "assistant", "content": reply})
        return reply
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analyzer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add hn_sentiment/analyzer.py tests/test_analyzer.py
git commit -m "feat: add analyzer with Claude summarize and chat"
```

---

### Task 7: renderer.py — Rich terminal UI and chat loop

**Files:**
- Create: `hn_sentiment/renderer.py`

No unit tests for renderer (output is visual). It is exercised by the integration test in Task 8.

- [ ] **Step 1: Implement `hn_sentiment/renderer.py`**

```python
# hn_sentiment/renderer.py
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich import print as rprint

console = Console()


def print_header(post_title: str, total_comments: int, analyzed_comments: int) -> None:
    """Print the post title and comment stats."""
    console.print()
    console.print(Panel(
        Text(post_title, style="bold white", justify="center"),
        border_style="orange3",
        title="[orange3]Hacker News Thread[/orange3]",
    ))
    console.print(
        f"  [dim]Comments:[/dim] [bold]{total_comments}[/bold] total · "
        f"[bold]{analyzed_comments}[/bold] analyzed for summary"
    )
    console.print()


def print_summary(summary: str) -> None:
    """Print the Amazon-style sentiment summary in a styled panel."""
    console.print(Panel(
        summary,
        title="[bold green]Thread Summary[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


def print_chat_response(response: str) -> None:
    """Print a Claude chat response."""
    console.print(Panel(
        response,
        title="[bold blue]Claude[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()


def print_error(message: str) -> None:
    """Print an error message and exit."""
    console.print(f"\n[bold red]Error:[/bold red] {message}\n")


def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[dim]{message}[/dim]")


def run_chat_loop(analyzer, collection, retriever_fn) -> None:
    """
    Run the interactive chat loop.
    analyzer: Analyzer instance
    collection: ChromaDB collection
    retriever_fn: callable(collection, query, top_k) -> list[dict]
    """
    console.print(Rule("[dim]Chat Mode — ask questions about the thread[/dim]"))
    console.print("[dim]Type [bold]exit[/bold] or [bold]quit[/bold] to end, or press Ctrl+C[/dim]\n")

    history = []
    while True:
        try:
            question = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]\n")
            break

        if question.strip().lower() in ("exit", "quit", ""):
            console.print("\n[dim]Goodbye![/dim]\n")
            break

        with console.status("[dim]Thinking...[/dim]"):
            relevant = retriever_fn(collection, question, top_k=10)
            reply = analyzer.chat(question, relevant, history)

        print_chat_response(reply)
```

- [ ] **Step 2: Commit**

```bash
git add hn_sentiment/renderer.py
git commit -m "feat: add Rich terminal renderer and chat loop"
```

---

### Task 8: cli.py — entry point and full integration

**Files:**
- Create: `hn_sentiment/cli.py`

- [ ] **Step 1: Implement `hn_sentiment/cli.py`**

```python
# hn_sentiment/cli.py
import os
import sys
import click
from rich.console import Console

from hn_sentiment.fetcher import extract_item_id, fetch_thread
from hn_sentiment.sampler import sample_for_summary
from hn_sentiment.embedder import build_collection
from hn_sentiment.retriever import retrieve_relevant
from hn_sentiment.analyzer import Analyzer
from hn_sentiment.renderer import (
    print_header,
    print_summary,
    print_error,
    print_info,
    run_chat_loop,
)

console = Console()


@click.command()
@click.argument("url")
@click.option("--max-comments", default=500, show_default=True, help="Max comments to fetch and embed.")
def main(url: str, max_comments: int) -> None:
    """Analyze sentiment and chat about a Hacker News thread."""

    # Validate API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    # Validate and parse URL
    try:
        item_id = extract_item_id(url)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    # Fetch thread
    with console.status("[dim]Fetching thread from Algolia...[/dim]"):
        try:
            title, comments = fetch_thread(item_id, max_comments=max_comments)
        except ValueError as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to fetch thread: {e}")
            sys.exit(1)

    if not comments:
        print_error("This post has no comments yet.")
        sys.exit(0)

    total_comments = len(comments)
    sampled = sample_for_summary(comments)

    # Build ChromaDB collection (embed all comments)
    with console.status(f"[dim]Embedding {total_comments} comments...[/dim]"):
        collection = build_collection(comments)

    print_header(title, total_comments, len(sampled))

    if total_comments >= max_comments:
        print_info(f"Note: thread has more than {max_comments} comments — showing analysis of first {max_comments}.")

    # Generate summary
    analyzer = Analyzer(api_key=api_key)
    with console.status("[dim]Generating summary...[/dim]"):
        try:
            summary = analyzer.summarize(title, sampled)
        except Exception as e:
            print_error(f"Failed to generate summary: {e}")
            sys.exit(1)

    print_summary(summary)

    # Enter chat loop
    run_chat_loop(analyzer, collection, retrieve_relevant)
```

- [ ] **Step 2: Verify the CLI entry point works**

Run: `hn-sentiment --help`
Expected:
```
Usage: hn-sentiment [OPTIONS] URL

  Analyze sentiment and chat about a Hacker News thread.

Options:
  --max-comments INTEGER  Max comments to fetch and embed.  [default: 500]
  --help                  Show this message and exit.
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: All tests PASS, no errors

- [ ] **Step 4: Smoke test with a real HN URL**

```bash
export ANTHROPIC_API_KEY=your-key-here
hn-sentiment "https://news.ycombinator.com/item?id=43230387"
```

Expected:
- Rich header panel with post title
- Comment stats line
- Green summary panel with 3-5 sentence Amazon-style summary
- Chat prompt appears: `You:`
- Type a question, receive a blue response panel
- Type `exit` to quit

- [ ] **Step 5: Commit**

```bash
git add hn_sentiment/cli.py
git commit -m "feat: add CLI entry point and wire full pipeline"
```
