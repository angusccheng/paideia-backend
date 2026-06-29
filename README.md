# Paideia — AI Study Tutor Backend

RAG (Retrieval-Augmented Generation) backend for the Paideia voice tutor. Students upload their own notes; the Vapi voice assistant answers questions using only that material.

## Architecture

```
Student uploads PDF/TXT
        │
        ▼
  /upload endpoint
        │
  pdfplumber parses text
        │
  Text split into 500-word chunks (50-word overlap)
        │
  OpenAI text-embedding-3-small embeds each chunk
        │
  ChromaDB stores chunks + vectors (per student collection)
        │
        ▼
  Vapi calls /query mid-conversation
        │
  Query is embedded → top 5 chunks retrieved
        │
  "context" string returned → injected into Vapi system prompt
```

## Project Structure

```
paideia-backend/
├── main.py                  # FastAPI app, middleware, routers
├── routes/
│   ├── upload.py            # POST /upload
│   └── query.py             # POST /query, GET/DELETE /documents/{id}
├── services/
│   ├── processor.py         # File parsing, chunking, embedding
│   └── vectorstore.py       # ChromaDB read/write helpers
├── chroma_db/               # Auto-created — persistent vector store on disk
├── requirements.txt
├── .env                     # Your secrets (never commit this)
└── .env.example             # Safe template to commit
```

## Setup

### 1. Clone and enter the project

```bash
cd paideia-backend
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...
```

### 5. Run the server

```bash
uvicorn main:app --reload --port 8000
```

The server starts at **http://localhost:8000**.  
Interactive API docs: **http://localhost:8000/docs**

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}`. Used by Vapi and load balancers to confirm the server is alive.

---

### `POST /upload`
Upload a student's notes for indexing.

**Form fields:**
| Field | Type | Description |
|---|---|---|
| `file` | File | PDF or TXT document |
| `student_id` | string | Unique identifier for the student |

**Response:**
```json
{
  "message": "File uploaded and indexed successfully.",
  "student_id": "student_42",
  "filename": "biology_notes.pdf",
  "chunks_stored": 34
}
```

---

### `POST /query`
Retrieve the most relevant chunks from a student's notes. **Called by Vapi mid-conversation.**

**Body (JSON):**
```json
{
  "student_id": "student_42",
  "query": "What is the role of mitochondria?"
}
```

**Response:**
```json
{
  "context": "Mitochondria are the powerhouse of the cell...\n\nThey produce ATP via...",
  "chunks_returned": 5
}
```

Plug `context` into your Vapi system prompt like:
```
You are a tutor. Use only the following notes to answer the student's questions:

{{context}}
```

---

### `GET /documents/{student_id}`
List all files uploaded by a student.

**Response:**
```json
{
  "student_id": "student_42",
  "documents": ["biology_notes.pdf", "chapter3.txt"],
  "total": 2
}
```

---

### `DELETE /documents/{student_id}`
Delete all uploaded notes and embeddings for a student.

**Response:**
```json
{
  "message": "All documents for student 'student_42' have been deleted.",
  "chunks_deleted": 34
}
```

---

## Vapi Integration

1. Deploy this backend to a public URL (Railway, Render, fly.io, etc.).
2. In your Vapi assistant settings, add a **Tool** with:
   - **Method:** POST
   - **URL:** `https://your-domain.com/query`
   - **Body:** `{ "student_id": "{{student_id}}", "query": "{{query}}" }`
3. Inject the returned `context` field into your assistant's system prompt.
