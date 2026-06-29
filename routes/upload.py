"""
upload.py — POST /upload endpoint.
Receives a file from the student, runs it through the processing pipeline,
and stores the resulting chunks + embeddings in ChromaDB.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.processor import process_file
from services.vectorstore import upsert_chunks

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),          # the uploaded file
    student_id: str = Form(...),           # which student owns this file
):
    """
    Accept a PDF or TXT file, chunk + embed it, and store it in ChromaDB
    under the student's own collection.

    Form fields:
      - file        — the document (PDF or TXT)
      - student_id  — a string identifier for the student (e.g. "student_42")
    """

    # Read the raw bytes from the upload stream
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Run the full parsing → chunking → embedding pipeline
        chunks, embeddings, metadatas, ids = process_file(
            file_bytes=file_bytes,
            filename=file.filename,
            student_id=student_id,
        )

        # Persist everything to ChromaDB
        upsert_chunks(
            student_id=student_id,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    except ValueError as e:
        # process_file raises ValueError for unsupported types or empty files
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Catch-all for unexpected errors (e.g. OpenAI API down)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return JSONResponse(
        status_code=200,
        content={
            "message": "File uploaded and indexed successfully.",
            "student_id": student_id,
            "filename": file.filename,
            "chunks_stored": len(chunks),
        },
    )
