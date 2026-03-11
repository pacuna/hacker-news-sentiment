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
