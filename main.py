"""
main.py — Application entry point.
Creates the FastAPI app, registers middleware, and wires up all route modules.
"""

from dotenv import load_dotenv

# Load environment variables from .env before anything else runs,
# so OPENAI_API_KEY is available when the OpenAI client is first created.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.query import router as query_router
from routes.voice import router as voice_router

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Paideia",
    description="AI study tutor — RAG backend for Vapi voice assistant",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS middleware
# Allow any origin during development. In production, replace "*" with
# your exact frontend domain (e.g. "https://paideia.app") to lock it down.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# Upload endpoint: POST /upload
app.include_router(upload_router)

# Query, list, and delete endpoints: POST /query, GET/DELETE /documents/{id}
app.include_router(query_router)

# Voice pipeline: WS /voice/session, POST /voice/transcribe, /voice/speak, /voice/respond
app.include_router(voice_router)

# ---------------------------------------------------------------------------
# Health check — GET /health
# Vapi and load balancers use this to confirm the server is alive.
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok"}
