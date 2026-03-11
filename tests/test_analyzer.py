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
