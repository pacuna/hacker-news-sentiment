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
