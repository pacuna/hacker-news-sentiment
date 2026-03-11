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
