import chromadb
import uuid


def build_collection(comments: list[dict]):
    """
    Embed all comments into a ChromaDB in-memory collection.
    Returns the collection for use by the retriever.
    The default embedding function (all-MiniLM-L6-v2) is used automatically.
    """
    client = chromadb.EphemeralClient()
    # Use a unique collection name to avoid conflicts in tests
    collection_name = f"hn_comments_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(collection_name)

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
