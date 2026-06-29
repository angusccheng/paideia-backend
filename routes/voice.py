"""
voice.py — Voice pipeline endpoints.

WebSocket endpoint for full duplex voice sessions, plus HTTP endpoints
for individual steps (transcribe, speak, respond) so the frontend can
call them independently if needed.
"""

import base64
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from services.llm import get_llm_response
from services.processor import embed_texts
from services.stt import transcribe
from services.tts import speak
from services.vectorstore import query_chunks

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
        { "type": "text",  "text": "typed message" }   ← optional text shortcut

      Server → Client:
        { "type": "transcript", "text": "..." }         ← what Whisper heard
        { "type": "response",   "text": "..." }         ← tutor text reply
        { "type": "audio",      "data": "<base64 mp3>" }← TTS audio to play
        { "type": "error",      "message": "..." }      ← something went wrong
    """
    await websocket.accept()

    # Conversation history persists for the lifetime of this WebSocket connection.
    # Each entry is {"role": "user"|"assistant", "content": "..."}.
    history: list[dict] = []

    try:
        while True:
            # Wait for the next message from the client
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
            # Handle audio input — transcribe first, then RAG + LLM + TTS
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

                # Step 1: Transcribe audio → text via Whisper
                try:
                    transcript = await transcribe(audio_bytes, mime_type)
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

                # Send the transcript back so the UI can show what was heard
                await websocket.send_text(json.dumps({
                    "type": "transcript", "text": transcript,
                }))

                # Steps 2-5 are shared with the text path below
                student_text = transcript

            # ------------------------------------------------------------------
            # Handle text input — skip transcription, go straight to RAG
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

            # Step 2: Embed the student's message and query ChromaDB
            try:
                query_embedding = embed_texts([student_text])[0]
                chunks = query_chunks(student_id, query_embedding, top_k=5)
                context = "\n\n".join(chunks)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"RAG retrieval failed: {e}",
                }))
                continue

            # Step 3: Get GPT-4o-mini response grounded in retrieved context
            try:
                reply_text = await get_llm_response(student_text, context, history)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"LLM failed: {e}",
                }))
                continue

            # Update conversation history with this turn
            history.append({"role": "user", "content": student_text})
            history.append({"role": "assistant", "content": reply_text})

            # Send the text reply so the UI can display it
            await websocket.send_text(json.dumps({
                "type": "response", "text": reply_text,
            }))

            # Step 4: Convert reply to speech via Deepgram TTS
            try:
                audio_bytes = await speak(reply_text)
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await websocket.send_text(json.dumps({
                    "type": "audio", "data": audio_b64,
                }))
            except Exception as e:
                # TTS failure is non-fatal — the client still has the text
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"TTS failed: {e}",
                }))

    except WebSocketDisconnect:
        # Client closed the connection — nothing to do, just stop the loop
        pass
    except Exception as e:
        # Unexpected error — try to notify the client before closing
        try:
            await websocket.send_text(json.dumps({
                "type": "error", "message": f"Session error: {e}",
            }))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POST /voice/transcribe — convert uploaded audio to text
# ---------------------------------------------------------------------------

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    mime_type: str = Form("webm"),
):
    """
    Accept an audio file and return its Whisper transcription.
    Useful for testing STT independently of the full pipeline.
    """
    audio_bytes = await file.read()
    try:
        text = await transcribe(audio_bytes, mime_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return JSONResponse({"transcript": text})


# ---------------------------------------------------------------------------
# POST /voice/speak — convert text to speech, return audio bytes
# ---------------------------------------------------------------------------

@router.post("/speak")
async def text_to_speech(text: str = Form(...)):
    """
    Accept a text string and return Deepgram TTS audio as an mp3 download.
    Useful for testing TTS independently.
    """
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
# POST /voice/respond — RAG + LLM text response (no audio)
# ---------------------------------------------------------------------------

class RespondRequest(BaseModel):
    student_id: str
    text: str
    history: list[dict] = []   # optional — pass prior turns for context


@router.post("/respond")
async def respond(body: RespondRequest):
    """
    Accept student_id and a text message.
    Retrieves relevant notes from ChromaDB and returns the tutor's text reply.
    No audio involved — useful for text-based chat UI or debugging the RAG pipeline.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    try:
        query_embedding = embed_texts([body.text])[0]
        chunks = query_chunks(body.student_id, query_embedding, top_k=5)
        context = "\n\n".join(chunks)
        reply = await get_llm_response(body.text, context, body.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {e}")

    return JSONResponse({"response": reply, "context_chunks": len(chunks)})
