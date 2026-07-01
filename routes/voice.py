"""
voice.py — Voice pipeline endpoints.

WebSocket endpoint for full duplex voice sessions, plus HTTP endpoints
for individual steps (transcribe, speak, respond) and session reporting.
"""

import base64
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from services.llm import get_llm_response, generate_session_report
from services.processor import embed_texts
from services.stt import transcribe
from services.tts import speak
from services.vectorstore import query_chunks, get_concept_summary

router = APIRouter(prefix="/voice")


# ---------------------------------------------------------------------------
# WebSocket — full voice conversation loop
# ---------------------------------------------------------------------------

@router.websocket("/session")
async def voice_session(websocket: WebSocket, student_id: str):
    """
    Full voice conversation over WebSocket.

    Connect with: ws://server/voice/session?student_id=<id>

    Message protocol (JSON):
      Client → Server:
        { "type": "audio", "data": "<base64-encoded audio bytes>", "mime_type": "webm" }
        { "type": "text",  "text": "typed message" }

      Server → Client:
        { "type": "transcript",     "text": "..." }
        { "type": "response",       "text": "..." }
        { "type": "audio",          "data": "<base64 mp3>" }
        { "type": "session_report", "data": { ...report... } }  ← sent once on disconnect
        { "type": "error",          "message": "..." }
    """
    await websocket.accept()

    # Conversation history — persists for the lifetime of this connection
    history: list[dict] = []

    # Concept tracking — every chunk retrieved during this session.
    # Each entry: { "lesson": "...", "concept": "...", "chunk_content": "..." }
    # Used at the end to build the mastery report.
    concepts_touched: list[dict] = []

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON. Send { type, data/text, mime_type }.",
                }))
                continue

            msg_type = message.get("type")

            # ------------------------------------------------------------------
            # START_SESSION — send a warm greeting, no RAG needed
            # ------------------------------------------------------------------
            if msg_type == "START_SESSION":
                try:
                    greeting = await get_llm_response(
                        student_message="",
                        context="",
                        history=[],
                        override_instruction=(
                            "The session is just starting. Greet the student warmly, "
                            "tell them you have access to their uploaded notes, and ask "
                            "what they want to focus on or what they already know about "
                            "the topic. Keep it to 2 sentences max."
                        ),
                    )
                    await websocket.send_text(json.dumps({"type": "response", "text": greeting}))
                    audio_bytes = await speak(greeting)
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": f"Greeting failed: {e}"}))
                continue

            # ------------------------------------------------------------------
            # Audio input — transcribe, then RAG + LLM + TTS
            # ------------------------------------------------------------------
            if msg_type == "audio":
                audio_b64 = message.get("data", "")
                mime_type = message.get("mime_type", "webm")

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Could not decode base64 audio data.",
                    }))
                    continue

                try:
                    # Always pass audio/webm — browsers record in webm format
                    # and Whisper needs the correct extension to decode it
                    transcript = await transcribe(audio_bytes, "audio/webm")
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error", "message": f"Transcription failed: {e}",
                    }))
                    continue

                if not transcript:
                    await websocket.send_text(json.dumps({
                        "type": "error", "message": "No speech detected.",
                    }))
                    continue

                await websocket.send_text(json.dumps({
                    "type": "transcript", "text": transcript,
                }))
                student_text = transcript

            # ------------------------------------------------------------------
            # Text input — skip transcription
            # ------------------------------------------------------------------
            elif msg_type == "text":
                student_text = message.get("text", "").strip()
                if not student_text:
                    continue
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type '{msg_type}'. Use 'audio' or 'text'.",
                }))
                continue

            # RAG retrieval — query_chunks now returns dicts with content/lesson/concept
            try:
                query_embedding = embed_texts([student_text])[0]
                chunk_dicts = await query_chunks(student_id, query_embedding, top_k=5)

                # Track every concept that surfaced during this session
                for chunk in chunk_dicts:
                    concepts_touched.append({
                        "lesson": chunk.get("lesson", "Unknown"),
                        "concept": chunk.get("concept", "Unknown"),
                        "chunk_content": chunk.get("content", ""),
                    })

                # Build the context string for the LLM from the content fields
                context = "\n\n".join(c["content"] for c in chunk_dicts)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"RAG retrieval failed: {e}",
                }))
                continue

            # LLM response
            try:
                reply_text = await get_llm_response(student_text, context, history)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"LLM failed: {e}",
                }))
                continue

            history.append({"role": "user", "content": student_text})
            history.append({"role": "assistant", "content": reply_text})

            await websocket.send_text(json.dumps({
                "type": "response", "text": reply_text,
            }))

            # TTS — non-fatal if it fails, client still has the text
            try:
                audio_bytes = await speak(reply_text)
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await websocket.send_text(json.dumps({
                    "type": "audio", "data": audio_b64,
                }))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"TTS failed: {e}",
                }))

    except WebSocketDisconnect:
        # Session ended — generate and send the mastery report before closing.
        # This only runs when the client disconnects gracefully (close frame received).
        # We wrap everything in try/except so a report failure never crashes the handler.
        try:
            full_concept_map = await get_concept_summary(student_id)
            report = await generate_session_report(history, concepts_touched, full_concept_map)
            await websocket.send_text(json.dumps({
                "type": "session_report",
                "data": report,
            }))
        except Exception:
            pass  # report failure is non-fatal — session data is already in history

    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error", "message": f"Session error: {e}",
            }))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POST /voice/transcribe
# ---------------------------------------------------------------------------

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    mime_type: str = Form("webm"),
):
    """Accept an audio file and return its Whisper transcription."""
    audio_bytes = await file.read()
    try:
        text = await transcribe(audio_bytes, mime_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return JSONResponse({"transcript": text})


# ---------------------------------------------------------------------------
# POST /voice/speak
# ---------------------------------------------------------------------------

@router.post("/speak")
async def text_to_speech(text: str = Form(...)):
    """Accept a text string and return Deepgram TTS audio as an mp3 download."""
    try:
        audio_bytes = await speak(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=response.mp3"},
    )


# ---------------------------------------------------------------------------
# POST /voice/respond
# ---------------------------------------------------------------------------

class RespondRequest(BaseModel):
    student_id: str
    text: str
    history: list[dict] = []


@router.post("/respond")
async def respond(body: RespondRequest):
    """RAG + LLM text response — no audio. Useful for text-based chat UI."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    try:
        query_embedding = embed_texts([body.text])[0]
        chunk_dicts = await query_chunks(body.student_id, query_embedding, top_k=5)
        context = "\n\n".join(c["content"] for c in chunk_dicts)
        reply = await get_llm_response(body.text, context, body.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {e}")

    return JSONResponse({"response": reply, "context_chunks": len(chunk_dicts)})


# ---------------------------------------------------------------------------
# POST /voice/session-report — standalone report endpoint (HTTP backup)
# ---------------------------------------------------------------------------

class SessionReportRequest(BaseModel):
    student_id: str
    conversation_history: list[dict]
    concepts_touched: list[dict] = []   # optional — if not provided, report covers all concepts


@router.post("/session-report")
async def session_report(body: SessionReportRequest):
    """
    Generate a mastery report from a completed session.

    Use this as a backup to the WebSocket close event — the frontend can
    call this at any time with the conversation history it has been tracking.

    Body:
      student_id           — the student's ID
      conversation_history — list of {role, content} dicts
      concepts_touched     — optional list of {lesson, concept, chunk_content} dicts
    """
    try:
        full_concept_map = await get_concept_summary(body.student_id)
        report = await generate_session_report(
            body.conversation_history,
            body.concepts_touched,
            full_concept_map,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    return JSONResponse({"report": report})
