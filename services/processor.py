"""
processor.py — Handles file parsing, text chunking, and embedding generation.
This is the "processing pipeline" that turns raw files into vectors we can search.
"""

import hashlib
import io
import os
from typing import Optional

import pdfplumber
from openai import OpenAI

# ---------------------------------------------------------------------------
# OpenAI client — initialised once, reused across requests
# ---------------------------------------------------------------------------

_openai_client: Optional[OpenAI] = None

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500       # approximate tokens per chunk
CHUNK_OVERLAP = 50     # tokens shared between adjacent chunks


def get_openai_client() -> OpenAI:
    """Return the shared OpenAI client, reading the API key from the environment."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Use pdfplumber to extract all text from a PDF file.
    pdfplumber handles complex layouts better than basic PDF readers.
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode a plain text file — try UTF-8 first, fall back to latin-1."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Route the file to the correct parser based on its extension.
    Raises ValueError for unsupported file types so the API can
    return a clear error message to the caller.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: '{filename}'. "
            "Please upload a PDF (.pdf) or plain text (.txt) file."
        )


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks measured in words (a rough proxy for tokens).

    Why overlap? If a key sentence falls at the boundary between two chunks,
    the overlap ensures it appears fully in at least one of them, so we never
    miss relevant context during retrieval.

    chunk_size=500 words ≈ 500 tokens for typical English text.
    overlap=50 words means consecutive chunks share their last/first 50 words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move forward by (chunk_size - overlap) so the next chunk re-uses
        # the last `overlap` words of the current chunk
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Call the OpenAI Embeddings API to convert a list of text strings
    into vectors. Each vector is a list of floats (1536 dimensions for
    text-embedding-3-small) that encodes the semantic meaning of the text.

    We send all texts in a single API call for efficiency.
    """
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # The API returns embeddings in the same order as the input texts
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Unique ID generation
# ---------------------------------------------------------------------------

def make_chunk_id(student_id: str, source: str, chunk_index: int) -> str:
    """
    Generate a stable, unique ID for each chunk.
    Using a hash of (student_id + source + index) means re-uploading the same
    file will upsert (overwrite) existing chunks rather than creating duplicates.
    """
    raw = f"{student_id}::{source}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def process_file(
    file_bytes: bytes,
    filename: str,
    student_id: str,
) -> tuple[list[str], list[list[float]], list[dict], list[str]]:
    """
    Full processing pipeline for an uploaded file:
      1. Parse raw bytes → plain text
      2. Split text → overlapping chunks
      3. Embed chunks → vectors
      4. Build metadata and IDs for each chunk

    Returns a tuple of (chunks, embeddings, metadatas, ids) ready to be
    handed directly to vectorstore.upsert_chunks().
    """
    # Step 1: Extract text
    text = extract_text(file_bytes, filename)
    if not text.strip():
        raise ValueError("The uploaded file appears to be empty or contains no readable text.")

    # Step 2: Chunk the text
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Could not produce any text chunks from the file.")

    # Step 3: Embed all chunks in one API call
    embeddings = embed_texts(chunks)

    # Step 4: Build per-chunk metadata and stable IDs
    metadatas = [
        {
            "student_id": student_id,
            "source": filename,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    ids = [make_chunk_id(student_id, filename, i) for i in range(len(chunks))]

    return chunks, embeddings, metadatas, ids
