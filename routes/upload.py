"""
upload.py — POST /upload endpoint.
Receives a file from the student, runs it through the processing pipeline,
stores chunks in Supabase pgvector, saves the raw file to Supabase Storage,
and records document metadata.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.processor import process_file, upload_file_to_supabase
from services.vectorstore import upsert_chunks, save_document_record

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),     # the uploaded file
    student_id: str = Form(...),      # which student owns this file
):
    """
    Accept a PDF or TXT file, chunk + embed it, and store everything in Supabase:
      1. Raw file → Supabase Storage bucket 'student-documents'
      2. Chunks + embeddings → Supabase pgvector 'chunks' table
      3. Document metadata → Supabase 'documents' table

    Form fields:
      - file        — the document (PDF or TXT)
      - student_id  — a string identifier for the student (e.g. "student_42")
    """
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Step 1: Parse → chunk → embed
        chunks, embeddings, metadatas, ids = process_file(
            file_bytes=file_bytes,
            filename=file.filename,
            student_id=student_id,
        )

        # Step 2: Store chunks + embeddings in Supabase pgvector
        await upsert_chunks(
            student_id=student_id,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        # Step 3: Upload the raw file to Supabase Storage
        storage_path = upload_file_to_supabase(file_bytes, student_id, file.filename)

        # Step 4: Save a document record so we can list files per student
        await save_document_record(
            student_id=student_id,
            filename=file.filename,
            file_path=storage_path,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
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
