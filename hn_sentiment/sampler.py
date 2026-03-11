def sample_for_summary(comments: list[dict], top_n: int = 100) -> list[dict]:
    """
    Return the top_n comments sorted by points descending.
    Used to select high-signal comments for the initial summary.
    """
    return sorted(comments, key=lambda c: c["points"], reverse=True)[:top_n]
