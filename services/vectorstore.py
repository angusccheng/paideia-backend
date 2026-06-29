"""
vectorstore.py — Manages the ChromaDB connection and all interactions
with the vector database. Think of this as the "database layer" of the app.
"""

import chromadb
from chromadb.config import Settings

# ChromaDB client is created once and reused across the app (singleton pattern)
_client = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Return the shared ChromaDB client, creating it on first call."""
    global _client
    if _client is None:
        # PersistentClient stores data on disk so it survives server restarts
        _client = chromadb.PersistentClient(
            path="chroma_db",
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(student_id: str) -> chromadb.Collection:
    """
    Return the ChromaDB collection for a given student.
    Each student gets their own collection named 'student_<id>'
    so their documents are fully isolated from other students.
    """
    client = get_chroma_client()
    collection_name = f"student_{student_id}"
    # get_or_create_collection is idempotent — safe to call repeatedly
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text embeddings
    )


def list_collections_for_student(student_id: str) -> list[str]:
    """Return a list of unique source filenames stored for this student."""
    collection = get_collection(student_id)
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return []
    # Deduplicate filenames from chunk metadata
    seen = set()
    filenames = []
    for meta in results["metadatas"]:
        fname = meta.get("source", "unknown")
        if fname not in seen:
            seen.add(fname)
            filenames.append(fname)
    return filenames


def delete_student_collection(student_id: str) -> int:
    """
    Delete all documents for a student.
    Returns the number of chunks that were deleted.
    """
    client = get_chroma_client()
    collection_name = f"student_{student_id}"
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        client.delete_collection(name=collection_name)
        return count
    except Exception:
        # Collection didn't exist — nothing to delete
        return 0


def upsert_chunks(
    student_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """
    Store text chunks and their embeddings in ChromaDB.
    upsert means: insert if new, update if the ID already exists.
    """
    collection = get_collection(student_id)
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query_chunks(
    student_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[str]:
    """
    Find the top_k most relevant chunks for a query embedding.
    Filters strictly to this student's collection so they only see
    their own material.
    """
    collection = get_collection(student_id)
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # can't request more than we have
        include=["documents"],
    )
    # results["documents"] is a list of lists (one list per query)
    return results["documents"][0] if results["documents"] else []
