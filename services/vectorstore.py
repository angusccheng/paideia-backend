"""
vectorstore.py — Supabase pgvector integration.
Replaces ChromaDB. All vector storage and retrieval goes through
a Postgres database with the pgvector extension enabled on Supabase.

Two clients are used:
  - supabase-py  → Storage bucket operations (uploading raw files)
  - asyncpg      → Direct SQL for vector similarity search, because
                   supabase-py does not support the <=> operator natively
"""

import os
from typing import Optional

import asyncpg
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
# asyncpg connection pool (for pgvector SQL queries)
# ---------------------------------------------------------------------------

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """
    Return the shared asyncpg connection pool, creating it on first call.
    asyncpg is used instead of supabase-py because we need to run raw SQL
    with the <=> cosine distance operator from pgvector.
    """
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise EnvironmentError("DATABASE_URL must be set.")
        # Register a codec so asyncpg can send Python lists as vector[] columns
        _pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=5,
            init=_register_vector_codec,
        )
    return _pool


async def _register_vector_codec(connection: asyncpg.Connection) -> None:
    """
    Tell asyncpg how to encode/decode the pgvector 'vector' type.
    Without this, passing a list of floats as a vector column fails.
    """
    await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await connection.set_type_codec(
        "vector",
        encoder=lambda v: "[" + ",".join(str(x) for x in v) + "]",
        decoder=lambda v: [float(x) for x in v.strip("[]").split(",")],
        schema="pg_catalog",
        format="text",
    )


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
    Uses ON CONFLICT (id) DO UPDATE so re-uploading the same file
    overwrites existing rows rather than creating duplicates.

    Table schema expected:
      id          TEXT PRIMARY KEY
      student_id  TEXT NOT NULL
      source      TEXT NOT NULL
      chunk_index INTEGER NOT NULL
      content     TEXT NOT NULL
      embedding   vector(1536) NOT NULL
    """
    pool = await get_pool()

    # Build rows as tuples matching the INSERT column order
    rows = [
        (
            ids[i],
            student_id,
            metadatas[i].get("source", "unknown"),
            metadatas[i].get("chunk_index", i),
            chunks[i],
            embeddings[i],
        )
        for i in range(len(chunks))
    ]

    async with pool.acquire() as conn:
        # executemany with upsert — one round-trip per chunk batch
        await conn.executemany(
            """
            INSERT INTO chunks (id, student_id, source, chunk_index, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6::vector)
            ON CONFLICT (id) DO UPDATE SET
                content    = EXCLUDED.content,
                embedding  = EXCLUDED.embedding,
                source     = EXCLUDED.source,
                chunk_index = EXCLUDED.chunk_index
            """,
            rows,
        )


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
    pool = await get_pool()

    # Convert the embedding list to the string format pgvector expects
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content
            FROM chunks
            WHERE student_id = $1
            ORDER BY embedding <=> $2::vector
            LIMIT $3
            """,
            student_id,
            embedding_str,
            top_k,
        )

    return [row["content"] for row in rows]


async def list_collections_for_student(student_id: str) -> list[str]:
    """Return a list of unique source filenames stored for this student."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT source FROM chunks WHERE student_id = $1 ORDER BY source",
            student_id,
        )
    return [row["source"] for row in rows]


async def delete_student_collection(student_id: str) -> int:
    """
    Delete all chunks belonging to this student.
    Returns the number of rows deleted.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chunks WHERE student_id = $1",
            student_id,
        )
    # asyncpg returns a status string like "DELETE 34" — parse the count
    try:
        return int(result.split()[-1])
    except (IndexError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Document record operations
# ---------------------------------------------------------------------------

async def save_document_record(
    student_id: str,
    filename: str,
    file_path: str,
) -> str:
    """
    Insert a row into the 'documents' table to track what files
    a student has uploaded (separate from the chunk rows).
    Returns the new document's id.

    Table schema expected:
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
