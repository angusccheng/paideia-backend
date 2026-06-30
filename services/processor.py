"""
processor.py — Handles file parsing, semantic chunking, and embedding generation.
This is the "processing pipeline" that turns raw files into vectors we can search.

New in this version: concept-aware semantic chunking.
Instead of splitting text every N words, we ask GPT-4o-mini to identify the
lesson/concept structure of the document first, then split along those boundaries.
This means each chunk represents one complete idea rather than an arbitrary slice.
"""

import hashlib
import io
import json
import os
from typing import Optional

import pdfplumber
from openai import OpenAI

# ---------------------------------------------------------------------------
# OpenAI client — initialised once, reused across requests
# ---------------------------------------------------------------------------

_openai_client: Optional[OpenAI] = None

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500        # words — used by fallback fixed-size chunker
CHUNK_OVERLAP = 50      # words — overlap for fallback chunker
MAX_WORDS_PER_GPT_CALL = 6000   # ~8k tokens; stay under GPT-4o-mini context safely
MAX_CONCEPT_WORDS = 800         # concepts longer than this get sub-chunked


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
    """Use pdfplumber to extract all text from a PDF file."""
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
    """Route the file to the correct parser based on its extension."""
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
# Fallback: fixed-size chunking (kept exactly as before)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping fixed-size chunks measured in words.
    Used as a fallback if semantic chunking fails for any reason.
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
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking — Step 1: extract structure via GPT-4o-mini
# ---------------------------------------------------------------------------

STRUCTURE_PROMPT = """You are an expert at analysing educational documents.
Read the text below and identify its lesson/chapter structure and the key concepts within each lesson.

Return ONLY valid JSON in exactly this format — no markdown, no explanation:
{
  "lessons": [
    {
      "lesson_title": "Lesson 1: Title Here",
      "concepts": [
        {
          "concept_title": "Concept Name",
          "start_marker": "first 6-8 words where this concept begins in the text",
          "end_marker": "last 6-8 words where this concept ends in the text"
        }
      ]
    }
  ]
}

Rules:
- start_marker and end_marker must be verbatim substrings from the text provided
- Each concept should represent one distinct idea or topic
- If the document has no clear lesson structure, treat the whole document as one lesson
- Keep concept titles concise (3-6 words)

TEXT:
"""


def _call_structure_gpt(text_section: str) -> dict:
    """
    Send one section of text to GPT-4o-mini and parse the returned JSON structure.
    Returns an empty dict if the response cannot be parsed.
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": STRUCTURE_PROMPT + text_section}],
        temperature=0,          # deterministic — we need consistent JSON
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def extract_structure(text: str) -> dict:
    """
    Identify the lesson and concept structure of a document using GPT-4o-mini.

    For documents longer than MAX_WORDS_PER_GPT_CALL words, we split into
    sections, process each separately, then merge the results into one structure.

    Returns a dict like:
      { "lessons": [ { "lesson_title": "...", "concepts": [...] } ] }

    Returns an empty dict if extraction fails — callers should fall back to
    fixed-size chunking in that case.
    """
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return {}

    # Short document — single GPT call
    if total_words <= MAX_WORDS_PER_GPT_CALL:
        return _call_structure_gpt(text)

    # Long document — split into sections and merge results
    sections = []
    start = 0
    while start < total_words:
        end = min(start + MAX_WORDS_PER_GPT_CALL, total_words)
        sections.append(" ".join(words[start:end]))
        start = end

    merged_lessons = []
    for i, section in enumerate(sections):
        result = _call_structure_gpt(section)
        lessons = result.get("lessons", [])
        # Prefix lesson titles with section number to avoid duplicates
        for lesson in lessons:
            if len(sections) > 1:
                lesson["lesson_title"] = f"Part {i+1}: {lesson.get('lesson_title', 'Untitled')}"
            merged_lessons.append(lesson)

    return {"lessons": merged_lessons}


# ---------------------------------------------------------------------------
# Semantic chunking — Step 2: split text along concept boundaries
# ---------------------------------------------------------------------------

def _sub_chunk(text: str, lesson: str, concept: str, max_words: int = MAX_CONCEPT_WORDS) -> list[dict]:
    """
    If a concept's text is longer than max_words, split it into smaller
    sub-chunks while keeping the same lesson and concept labels.
    """
    words = text.split()
    if len(words) <= max_words:
        return [{"content": text, "lesson": lesson, "concept": concept}]

    result = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        result.append({
            "content": " ".join(words[start:end]),
            "lesson": lesson,
            "concept": concept,
        })
        start = end
    return result


def chunk_by_concept(text: str, structure: dict) -> list[dict]:
    """
    Split the document text along the concept boundaries identified by extract_structure.

    For each concept, we locate its start and end within the full text using
    the start_marker and end_marker strings, then extract that slice.

    If a concept section exceeds MAX_CONCEPT_WORDS words it is further split
    into sub-chunks that keep the same lesson/concept labels.

    Returns a list of dicts:
      [{ "content": "...", "lesson": "...", "concept": "..." }, ...]
    """
    lessons = structure.get("lessons", [])
    if not lessons:
        return []

    chunks = []

    for lesson in lessons:
        lesson_title = lesson.get("lesson_title", "Unknown Lesson")
        concepts = lesson.get("concepts", [])

        for concept in concepts:
            concept_title = concept.get("concept_title", "Unknown Concept")
            start_marker = concept.get("start_marker", "").strip()
            end_marker = concept.get("end_marker", "").strip()

            if not start_marker:
                continue

            # Locate the start of this concept in the full text
            start_pos = text.find(start_marker)
            if start_pos == -1:
                # Fuzzy fallback: try the first 4 words of the marker
                short_marker = " ".join(start_marker.split()[:4])
                start_pos = text.find(short_marker)
            if start_pos == -1:
                continue  # can't locate this concept — skip it

            # Locate the end of this concept
            end_pos = -1
            if end_marker:
                end_pos = text.find(end_marker, start_pos)
                if end_pos != -1:
                    # Include the end marker text itself
                    end_pos += len(end_marker)

            if end_pos == -1:
                # No end marker found — take text from start to end of document
                # (next concept's start_marker will bound it in the next iteration)
                concept_text = text[start_pos:]
            else:
                concept_text = text[start_pos:end_pos]

            concept_text = concept_text.strip()
            if not concept_text:
                continue

            # Sub-chunk if the concept section is too long
            chunks.extend(_sub_chunk(concept_text, lesson_title, concept_title))

    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Call the OpenAI Embeddings API to convert a list of text strings into vectors.
    We send all texts in a single API call for efficiency.
    """
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Unique ID generation
# ---------------------------------------------------------------------------

def make_chunk_id(student_id: str, source: str, chunk_index: int) -> str:
    """
    Generate a stable, unique ID for each chunk.
    Using a hash means re-uploading the same file overwrites existing chunks.
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
      2. Extract document structure via GPT-4o-mini (lesson/concept map)
      3. Split text along concept boundaries → semantic chunks
         (falls back to fixed-size chunking if structure extraction fails)
      4. Embed each chunk
      5. Build metadata including lesson + concept labels

    Returns (chunks, embeddings, metadatas, ids) for vectorstore.upsert_chunks().
    """
    # Step 1: Extract text from the file
    text = extract_text(file_bytes, filename)
    if not text.strip():
        raise ValueError("The uploaded file appears to be empty or contains no readable text.")

    # Step 2 & 3: Try semantic chunking, fall back to fixed-size if anything fails
    chunk_dicts: list[dict] = []
    try:
        structure = extract_structure(text)
        if structure.get("lessons"):
            chunk_dicts = chunk_by_concept(text, structure)
    except Exception:
        # Structure extraction failed — chunk_dicts stays empty, fallback triggers below
        pass

    if not chunk_dicts:
        # Fallback: fixed-size chunking with no lesson/concept labels
        raw_chunks = chunk_text(text)
        chunk_dicts = [
            {"content": c, "lesson": "Unknown", "concept": "Unknown"}
            for c in raw_chunks
        ]

    if not chunk_dicts:
        raise ValueError("Could not produce any text chunks from the file.")

    # Step 4: Embed the content of each chunk
    contents = [c["content"] for c in chunk_dicts]
    embeddings = embed_texts(contents)

    # Step 5: Build metadata rows — now includes lesson and concept
    metadatas = [
        {
            "student_id": student_id,
            "source": filename,
            "chunk_index": i,
            "lesson": chunk_dicts[i].get("lesson", "Unknown"),
            "concept": chunk_dicts[i].get("concept", "Unknown"),
        }
        for i in range(len(chunk_dicts))
    ]
    ids = [make_chunk_id(student_id, filename, i) for i in range(len(chunk_dicts))]

    return contents, embeddings, metadatas, ids


# ---------------------------------------------------------------------------
# Supabase Storage upload
# ---------------------------------------------------------------------------

def upload_file_to_supabase(
    file_bytes: bytes,
    student_id: str,
    filename: str,
) -> str:
    """
    Upload the raw file to Supabase Storage under the 'student-documents' bucket.
    Storage path: {student_id}/{filename}
    Returns the storage path string.
    """
    from services.vectorstore import get_supabase_client

    supabase = get_supabase_client()
    storage_path = f"{student_id}/{filename}"

    supabase.storage.from_("student-documents").upload(
        path=storage_path,
        file=file_bytes,
        file_options={"upsert": "true"},
    )

    return storage_path
