"""
query.py — POST /query, GET /documents/{student_id}, DELETE /documents/{student_id}.
These are the endpoints Vapi and the frontend will call during a tutoring session.
All vectorstore functions are now async (Supabase pgvector via asyncpg).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.processor import embed_texts
from services.vectorstore import (
    delete_student_collection,
    list_collections_for_student,
    query_chunks,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    student_id: str   # identifies which student's notes to search
    query: str        # the question Vapi's AI is asking mid-conversation


# ---------------------------------------------------------------------------
# POST /query  — called by Vapi mid-conversation
# ---------------------------------------------------------------------------

@router.post("/query")
async def query_documents(body: QueryRequest):
    """
    Embed the query string and retrieve the top 5 most relevant chunks
    from that student's Supabase pgvector collection.

    Vapi calls this endpoint mid-conversation. The returned "context" string
    is injected into the AI's system prompt so it can answer using only the
    student's own notes.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        # Embed the query using the same model used at upload time
        query_embedding = embed_texts([body.query])[0]

        # Retrieve top 5 chunks filtered to this student only
        chunks = await query_chunks(
            student_id=body.student_id,
            query_embedding=query_embedding,
            top_k=5,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    if not chunks:
        return JSONResponse(
            status_code=200,
            content={
                "context": "",
                "message": "No relevant content found. The student may not have uploaded any notes yet.",
                "chunks_returned": 0,
            },
        )

    context_string = "\n\n".join(chunks)

    return JSONResponse(
        status_code=200,
        content={
            "context": context_string,
            "chunks_returned": len(chunks),
        },
    )


# ---------------------------------------------------------------------------
# GET /documents/{student_id}  — list all uploaded files for a student
# ---------------------------------------------------------------------------

@router.get("/documents/{student_id}")
async def list_documents(student_id: str):
    """Return the list of unique filenames uploaded and indexed for this student."""
    try:
        filenames = await list_collections_for_student(student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve documents: {str(e)}")

    return JSONResponse(
        status_code=200,
        content={
            "student_id": student_id,
            "documents": filenames,
            "total": len(filenames),
        },
    )


# ---------------------------------------------------------------------------
# DELETE /documents/{student_id}  — wipe all notes for a student
# ---------------------------------------------------------------------------

@router.delete("/documents/{student_id}")
async def delete_documents(student_id: str):
    """Delete all uploaded notes and embeddings for a student."""
    try:
        chunks_deleted = await delete_student_collection(student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

    return JSONResponse(
        status_code=200,
        content={
            "message": f"All documents for student '{student_id}' have been deleted.",
            "chunks_deleted": chunks_deleted,
        },
    )
