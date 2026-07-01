"""
stt.py — Speech-to-Text using OpenAI Whisper.
Converts raw audio bytes into a transcribed text string.
"""

from services.processor import get_openai_client

# Audio formats Whisper accepts — we pass the mime type so the API
# knows how to decode the bytes without needing a real filename.
SUPPORTED_FORMATS = {"webm", "wav", "mp4", "m4a", "ogg", "flac", "mp3"}


async def transcribe(audio_bytes: bytes, mime_type: str = "webm") -> str:
    """
    Send raw audio bytes to OpenAI Whisper and return the transcribed text.

    mime_type: the audio format string e.g. "webm", "wav", "m4a".
    The Whisper API needs a filename with the correct extension so it
    can detect the codec — we fake one using the mime_type.
    """
    client = get_openai_client()

    # Normalise the mime_type: strip any "audio/" prefix the browser might send
    # e.g. "audio/webm" → "webm"
    fmt = mime_type.lower().replace("audio/", "").split(";")[0].strip()
    if fmt not in SUPPORTED_FORMATS:
        fmt = "webm"  # safe default for browser-recorded audio

    # Whisper requires a (filename, bytes, mime_type) tuple so it can
    # detect the codec from the extension — a plain BytesIO without a
    # recognised filename causes a 400 "invalid file format" error.
    file_tuple = (f"audio.{fmt}", audio_bytes, f"audio/{fmt}")

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple,
        response_format="text",
    )

    # response is a plain string when response_format="text"
    return response.strip() if isinstance(response, str) else response.text.strip()
