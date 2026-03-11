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
