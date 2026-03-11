# HN Sentiment

A CLI tool that fetches any Hacker News thread, generates an Amazon-style sentiment summary of the comments using Claude, and lets you have a follow-up conversation about the discussion powered by RAG.

## Features

- **Single-request fetch** — pulls the full comment tree from the [Algolia HN API](https://hn.algolia.com/api)
- **Smart sampling** — top-voted comments are selected for the initial summary; all comments are embedded for chat
- **Amazon-style summary** — natural language overview of what commenters agree on, disagree about, and the notable perspectives
- **RAG-powered chat** — ask follow-up questions; semantically relevant comments are retrieved and injected into each reply
- **General knowledge fallback** — if comments don't cover your question, Claude answers from its own knowledge and says so
- **Rich terminal UI** — styled panels, colors, and an interactive prompt

## Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

## Installation

```bash
git clone https://github.com/youruser/hacker-news-sentiment.git
cd hacker-news-sentiment
pip install -e .
```

## Usage

```bash
export ANTHROPIC_API_KEY=your-key-here
hn-sentiment <url> [--max-comments N]
```

| Argument | Description |
|----------|-------------|
| `url` | Full HN post URL, e.g. `https://news.ycombinator.com/item?id=12345` |
| `--max-comments` | Max comments to fetch and embed (default: `500`) |

After the summary is displayed, you enter an interactive chat loop. Type `exit` or `quit` (or press `Ctrl+C`) to end.

## Example

Running against [Yann LeCun raises $1B to build AI that understands the physical world](https://news.ycombinator.com/item?id=47320600):

```
$ hn-sentiment "https://news.ycombinator.com/item?id=47320600"
```

```
╭───────────────────────────── Hacker News Thread ─────────────────────────────╮
│    Yann LeCun raises $1B to build AI that understands the physical world     │
╰──────────────────────────────────────────────────────────────────────────────╯
  Comments: 477 total · 100 analyzed for summary

╭─────────────────────────────── Thread Summary ───────────────────────────────╮
│                                                                              │
│  Commenters are broadly skeptical about whether current LLM-based            │
│  approaches can achieve AGI, with many agreeing that architectural           │
│  limitations — particularly the lack of continuous learning, grounding in    │
│  physical reality, and genuine causal understanding — represent fundamental  │
│  bottlenecks rather than engineering problems to be solved incrementally. A  │
│  recurring point of agreement is that humans themselves rely heavily on      │
│  pattern-matching and associative reasoning rather than pure logical         │
│  deduction, with several commenters citing Kahneman's System 1/System 2      │
│  framework and the Wason selection task as evidence that the gap between     │
│  human and LLM cognition may be smaller than commonly assumed. Opinions are  │
│  sharply divided on whether AI will benefit humanity broadly or primarily    │
│  serve to concentrate power among the wealthy, and on whether LeCun's        │
│  world-model approach represents a genuinely promising alternative path or   │
│  just another overhyped research direction. Some commenters express          │
│  cautious optimism about exploring diverse architectural approaches —        │
│  including physical world models and embodied learning — while others argue  │
│  that without solving continuous learning and something analogous to         │
│  biological motivation or self-preservation drives, no current approach can  │
│  bridge the gap to true general intelligence. A notable minority view holds  │
│  that the entire framing of comparing machine and human intelligence is      │
│  misguided, arguing instead that AI represents a fundamentally different     │
│  shape of intelligence that should be evaluated on its practical utility     │
│  rather than its resemblance to human cognition.                             │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

────────────────── Chat Mode — ask questions about the thread ──────────────────
Type exit or quit to end, or press Ctrl+C

You: What are the main criticisms people have?
```

```
╭─────────────────────────────────── Claude ───────────────────────────────────╮
│                                                                              │
│  Based on the comments in the thread, there are several criticisms raised:   │
│                                                                              │
│  1. Criticism of LeCun's output at Meta/Facebook: One commenter suggests     │
│  he didn't produce enough impactful work despite having significant          │
│  resources, though another pushes back noting he was in a research group,   │
│  not a product group.                                                        │
│                                                                              │
│  2. Criticism of LeCun's stance on autoregressive models: Some see his      │
│  skepticism of autoregressive models as contrarian, though at least one     │
│  commenter defends this, praising him for "not just chasing hype."           │
│                                                                              │
│  3. Criticism of LLMs' reasoning capabilities: One commenter argues LLMs    │
│  lack genuine reasoning, pointing to the missing feedback loop and noting   │
│  that LLMs' world models are formed almost exclusively from language.        │
│                                                                              │
│  4. A counter-criticism defending LLMs: One commenter pushes back, citing   │
│  Kahneman's framework and the Wason selection task to argue humans mostly    │
│  do pattern matching too, making the gap potentially smaller than assumed.   │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## How it works

1. **Fetch** — Calls `https://hn.algolia.com/api/v1/items/{id}` to get the full comment tree in one request
2. **Sample** — Selects the top 100 comments by vote count for the initial summary
3. **Embed** — Embeds all fetched comments into a ChromaDB in-memory vector store using `all-MiniLM-L6-v2`
4. **Summarize** — Sends sampled comments to Claude with a prompt to produce an Amazon-style summary
5. **Chat** — On each question, retrieves the 10 most semantically relevant comments via cosine similarity and injects them as context for Claude's reply

## Architecture

```
hn_sentiment/
├── cli.py        # Entry point, error handling, orchestration
├── fetcher.py    # Algolia API, recursive comment flattening
├── sampler.py    # Top-voted comment selection
├── embedder.py   # ChromaDB EphemeralClient (in-memory)
├── retriever.py  # Semantic comment retrieval
├── analyzer.py   # Claude API: summarize() and chat()
└── renderer.py   # Rich terminal UI and chat loop
```
