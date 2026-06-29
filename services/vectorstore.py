"""
vectorstore.py — Supabase pgvector integration via psycopg2.
All vector storage and retrieval goes through a Postgres database
with the pgvector extension enabled on Supabase.

Two clients are used:
  - supabase-py  → Storage bucket operations and simple table inserts
  - psycopg2     → Direct SQL for pgvector similarity search (<=> operator)
"""

import os
from typing import Optional

import psycopg2
import psycopg2.extras
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Supabase client (for Storage and simple table operations)
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
# psycopg2 connection helper
# ---------------------------------------------------------------------------

def get_connection() -> psycopg2.extensions.connection:
    """
    Open and return a new psycopg2 connection using DATABASE_URL.
    Each function opens its own connection and closes it when done.
    This avoids connection pool complexity while keeping things simple.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL must be set.")
    return psycopg2.connect(database_url)


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
    Insert or update chunk rows in the 'chunks' table.
    ON CONFLICT (id) DO UPDATE means re-uploading the same file
    overwrites existing rows rather than creating duplicates.

    Table schema:
      id          TEXT PRIMARY KEY
      student_id  TEXT NOT NULL
      source      TEXT NOT NULL
      chunk_index INTEGER NOT NULL
      content     TEXT NOT NULL
      embedding   vector(1536) NOT NULL
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for i in range(len(chunks)):
                # Convert the Python list to the string format pgvector expects: [0.1,0.2,...]
                embedding_str = "[" + ",".join(str(x) for x in embeddings[i]) + "]"

                cur.execute(
                    """
                    INSERT INTO chunks (id, student_id, source, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        content     = EXCLUDED.content,
                        embedding   = EXCLUDED.embedding,
                        source      = EXCLUDED.source,
                        chunk_index = EXCLUDED.chunk_index
                    """,
                    (
                        ids[i],
                        student_id,
                        metadatas[i].get("source", "unknown"),
                        metadatas[i].get("chunk_index", i),
                        chunks[i],
                        embedding_str,
                    ),
                )
        conn.commit()
    finally:
        conn.close()


async def query_chunks(
    student_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[str]:
    """
    Find the top_k most semantically similar chunks for this student
    using pgvector cosine distance (<=>).

    Filters strictly by student_id so students never see each other's notes.
    """
    # Convert the embedding list to the pgvector string format
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content
                FROM chunks
                WHERE student_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (student_id, embedding_str, top_k),
            )
            rows = cur.fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


async def list_collections_for_student(student_id: str) -> list[str]:
    """Return a list of unique source filenames stored for this student."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT source FROM chunks WHERE student_id = %s ORDER BY source",
                (student_id,),
            )
            rows = cur.fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


async def delete_student_collection(student_id: str) -> int:
    """
    Delete all chunks belonging to this student.
    Returns the number of rows deleted.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE student_id = %s",
                (student_id,),
            )
            count = cur.rowcount  # psycopg2 gives the deleted row count directly
        conn.commit()
        return count
    finally:
        conn.close()


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
    # Use supabase-py for the simple insert — no vector types involved
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
