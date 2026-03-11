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
