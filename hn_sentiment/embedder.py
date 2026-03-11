import chromadb


def build_collection(comments: list[dict]):
    """
    Embed all comments into a ChromaDB in-memory collection.
    Returns the collection for use by the retriever.
    The default embedding function (all-MiniLM-L6-v2) is used automatically.
    """
    client = chromadb.EphemeralClient()
    # Delete existing collection if it exists to ensure clean state
    try:
        client.delete_collection("hn_comments")
    except Exception:
        pass
    collection = client.create_collection("hn_comments")

    if not comments:
        return collection

    collection.add(
        ids=[c["id"] for c in comments],
        documents=[c["text"] for c in comments],
        metadatas=[
            {"author": c["author"], "points": c["points"], "depth": c["depth"]}
            for c in comments
        ],
    )
    return collection
