import anthropic

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024


def _format_comments(comments: list[dict]) -> str:
    lines = []
    for c in comments:
        lines.append(f"[{c['author']} | points: {c['points']}]: {c['text']}")
    return "\n\n".join(lines)


class Analyzer:
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    def summarize(self, post_title: str, comments: list[dict]) -> str:
        """
        Generate an Amazon-style natural language summary of the HN thread.
        Returns a single paragraph describing the main views and sentiment.
        """
        formatted = _format_comments(comments)
        prompt = (
            f'Below are comments from a Hacker News post titled "{post_title}".\n\n'
            f"{formatted}\n\n"
            "Write a natural language summary of the main views expressed in these comments, "
            "similar to how Amazon summarizes product reviews. "
            "Describe what commenters generally agree on, what they are divided about, "
            "and highlight any notable perspectives. "
            "Write in third person (e.g. 'Commenters say...', 'Opinions are split on...'). "
            "Keep it to 3-5 sentences."
        )
        message = self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def chat(self, user_question: str, retrieved_comments: list[dict], history: list[dict]) -> str:
        """
        Answer a follow-up question using retrieved comments as context.
        Appends the user message and assistant reply to history in place.
        Returns the assistant's reply text.
        """
        context = _format_comments(retrieved_comments)
        user_content = (
            f"Relevant comments from the thread:\n\n{context}\n\n"
            f"Question: {user_question}"
        )
        history.append({"role": "user", "content": user_content})

        message = self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You are a helpful assistant analyzing a Hacker News thread. "
                "Answer questions based on the comments provided. "
                "If asked to show a specific comment, quote it directly. "
                "Be concise and reference specific commenters when relevant."
            ),
            messages=list(history),
        )
        reply = message.content[0].text
        history.append({"role": "assistant", "content": reply})
        return reply
