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
