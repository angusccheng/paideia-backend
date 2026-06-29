"""
vectorstore.py — Supabase pgvector integration using supabase-py only.
No native database drivers needed — all queries go through the Supabase
REST API (httpx under the hood), which works on any deployment platform.

Vector similarity search uses a Postgres function called via supabase.rpc()
because the supabase-py client cannot run raw SQL with the <=> operator directly.
The function must exist in your Supabase database — see README for the SQL.
"""

import os
from typing import Optional

from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Supabase client — created once, reused across all requests
# ---------------------------------------------------------------------------

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Return the shared Supabase client, creating it on first call."""
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
        _supabase_client = create_client(url, key)
    return _supabase_client


# ---------------------------------------------------------------------------
# Chunk operations
# ---------------------------------------------------------------------------

async def upsert_chunks(
    student_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """
    Insert or update chunk rows in the 'chunks' table via supabase-py.
    Supabase upsert uses ON CONFLICT (id) DO UPDATE internally, so
    re-uploading the same file overwrites rather than duplicates.

    Table schema:
      id          TEXT PRIMARY KEY
      student_id  TEXT NOT NULL
      source      TEXT NOT NULL
      chunk_index INTEGER NOT NULL
      content     TEXT NOT NULL
      embedding   vector(1536) NOT NULL
    """
    supabase = get_supabase_client()

    # Build the rows list — embedding is passed as a plain Python list,
    # which Supabase serialises as a JSON array. pgvector accepts this format.
    rows = [
        {
            "id": ids[i],
            "student_id": student_id,
            "source": metadatas[i].get("source", "unknown"),
            "chunk_index": metadatas[i].get("chunk_index", i),
            "content": chunks[i],
            "embedding": embeddings[i],   # list[float] → pgvector accepts JSON array
        }
        for i in range(len(chunks))
    ]

    supabase.table("chunks").upsert(rows).execute()


async def query_chunks(
    student_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[str]:
    """
    Find the top_k most semantically similar chunks for this student.

    Calls the 'match_chunks' Postgres function via supabase.rpc().
    That function runs the pgvector <=> cosine distance query server-side,
    which avoids needing a direct database connection from Python.

    The SQL function must exist in your Supabase database:
      create or replace function match_chunks(
        p_student_id text,
        p_embedding  vector(1536),
        p_limit      int
      )
      returns table(content text)
      language sql as $$
        select content from chunks
        where student_id = p_student_id
        order by embedding <=> p_embedding
        limit p_limit;
      $$;
    """
    supabase = get_supabase_client()

    response = supabase.rpc(
        "match_chunks",
        {
            "p_student_id": student_id,
            "p_embedding": query_embedding,   # passed as JSON array
            "p_limit": top_k,
        },
    ).execute()

    return [row["content"] for row in (response.data or [])]


async def list_collections_for_student(student_id: str) -> list[str]:
    """Return a list of unique source filenames stored for this student."""
    supabase = get_supabase_client()

    response = (
        supabase.table("chunks")
        .select("source")
        .eq("student_id", student_id)
        .execute()
    )

    # Deduplicate while preserving order
    seen = set()
    filenames = []
    for row in (response.data or []):
        src = row["source"]
        if src not in seen:
            seen.add(src)
            filenames.append(src)
    return sorted(filenames)


async def delete_student_collection(student_id: str) -> int:
    """
    Delete all chunks belonging to this student.
    Returns the number of rows deleted.
    """
    supabase = get_supabase_client()

    # Fetch count before deleting so we can return it
    count_response = (
        supabase.table("chunks")
        .select("id", count="exact")
        .eq("student_id", student_id)
        .execute()
    )
    count = count_response.count or 0

    supabase.table("chunks").delete().eq("student_id", student_id).execute()

    return count


# ---------------------------------------------------------------------------
# Document record operations
# ---------------------------------------------------------------------------

async def save_document_record(
    student_id: str,
    filename: str,
    file_path: str,
) -> str:
    """
    Insert a row into the 'documents' table to track uploaded files.
    Returns the new document's id.

    Table schema:
      id          UUID DEFAULT gen_random_uuid() PRIMARY KEY
      student_id  TEXT NOT NULL
      filename    TEXT NOT NULL
      file_path   TEXT NOT NULL
      created_at  TIMESTAMPTZ DEFAULT now()
    """
    supabase = get_supabase_client()

    response = (
        supabase.table("documents")
        .insert({
            "student_id": student_id,
            "filename": filename,
            "file_path": file_path,
        })
        .execute()
    )
    return response.data[0]["id"]
