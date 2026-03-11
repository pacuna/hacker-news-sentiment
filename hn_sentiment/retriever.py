def retrieve_relevant(collection, query: str, top_k: int = 10) -> list[dict]:
    """
    Embed query and return top_k most semantically relevant comments from the collection.
    Returns list of comment dicts with keys: id, text, author, points, depth.
    """
    if collection.count() == 0:
        return []

    k = min(top_k, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )

    comments = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        comments.append({
            "id": results["ids"][0][i],
            "text": doc,
            "author": meta["author"],
            "points": meta["points"],
            "depth": meta["depth"],
        })
    return comments
