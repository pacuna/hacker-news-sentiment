# HN Sentiment CLI — Design Spec

**Date:** 2026-03-10

## Overview

A CLI application that takes a Hacker News post URL, fetches all comments via the Algolia HN API, performs sentiment analysis and summarization using the Anthropic Claude API, and enters an interactive RAG-powered chat loop for follow-up questions.

---

## Goals

- Parse a HN thread URL and fetch the full comment tree
- Generate an Amazon-style natural language summary of main views and sentiment
- Support interactive follow-up questions about the thread using RAG
- Display results in a rich, styled terminal UI

---

## Architecture

### Components

| File | Responsibility |
|------|---------------|
| `cli.py` | Entry point (click). Parses URL, wires components together, runs the flow |
| `fetcher.py` | Calls Algolia HN API, recursively flattens comment tree with vote counts |
| `sampler.py` | Smart sampling: top-voted comments for summary, fills remainder randomly |
| `embedder.py` | Embeds all fetched comments into a ChromaDB in-memory collection |
| `retriever.py` | Embeds user query, retrieves top-K semantically relevant comments from ChromaDB |
| `analyzer.py` | `summarize()` for initial summary; `chat()` for follow-up questions with history |
| `renderer.py` | Rich UI: styled panels, stats, summary display, interactive chat loop |

---

## Data Flow

1. User runs `hn-sentiment <url> [--max-comments N]`
2. `cli.py` extracts the item ID from the HN URL
3. `fetcher.py` calls `https://hn.algolia.com/api/v1/items/<id>` — single request returns full thread tree. Flattens to a list of comments with text, author, points, depth.
4. Two things happen:
   - `sampler.py` selects top-voted comments for the initial summary
   - `embedder.py` embeds **all** fetched comments into ChromaDB in-memory
5. `analyzer.py.summarize()` sends sampled comments to Claude → returns Amazon-style summary paragraph
6. `renderer.py` displays: post title, total/sampled comment stats, summary panel
7. Chat loop begins:
   - User types a question
   - `retriever.py` embeds the query, retrieves top-K relevant comments from ChromaDB
   - `analyzer.py.chat()` sends retrieved comments + full conversation history to Claude
   - `renderer.py` displays Claude's response
   - Loop continues until user types `exit`/`quit` or hits Ctrl+C

---

## CLI Interface

```
hn-sentiment <url> [--max-comments N]
```

- `<url>` — Full HN post URL, e.g. `https://news.ycombinator.com/item?id=12345`
- `--max-comments` — Maximum comments to fetch and embed (default: 500)
- `ANTHROPIC_API_KEY` — Read from environment variable

---

## Sampling Strategy

- **For summary:** Take top N comments sorted by points descending (top 50–100 depending on limit). Goal: high-signal comments for quality summary.
- **For RAG:** All fetched comments are embedded — even low-voted or deeply nested ones are retrievable via semantic search.

---

## RAG Design

- **Embedding model:** ChromaDB default (`all-MiniLM-L6-v2` via sentence-transformers, runs locally, ~80MB one-time download)
- **Storage:** ChromaDB in-memory collection — lives for the duration of the session, no disk writes
- **Retrieval:** On each user query, embed the query and return top-K (K=10) most similar comments by cosine similarity
- **Context injection:** Retrieved comments are prepended to the Claude API call along with the conversation history

---

## Conversation History

- Maintained as a Python list of `{"role": ..., "content": ...}` dicts in memory
- Passed in full on every chat API call
- Session ends when the user exits — no persistence

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid or non-HN URL | Clear error message before any API call |
| Post not found (Algolia null) | Friendly error, exit |
| Post has no comments | Inform user, exit gracefully |
| Algolia API failure | Show error with HTTP status code |
| `ANTHROPIC_API_KEY` missing | Prompt user to set the env var, exit |
| Large thread exceeds limit | Show "Analyzed X of Y total comments" note |

---

## Dependencies

- `click` — CLI framework
- `rich` — Terminal UI (panels, colors, styled text)
- `httpx` — HTTP client for Algolia API
- `chromadb` — In-memory vector store
- `anthropic` — Claude API client

---

## Out of Scope

- Persistent sessions / resuming past conversations
- Sentiment per individual comment
- Exporting results to file
- Authentication / rate limit handling beyond basic errors
