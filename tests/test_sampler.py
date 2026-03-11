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
