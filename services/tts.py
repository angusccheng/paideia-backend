"""
tts.py — Text-to-Speech using Deepgram Aura.
Converts a text string into raw audio bytes ready to stream to the client.
"""

import os
import httpx

DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
DEEPGRAM_MODEL = "aura-asteria-en"  # warm, natural female voice


async def speak(text: str) -> bytes:
    """
    Send text to the Deepgram TTS API and return raw audio bytes (mp3).

    Deepgram Aura is significantly cheaper and lower-latency than
    OpenAI TTS, and aura-asteria-en sounds natural for a tutor persona.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPGRAM_API_KEY is not set in the environment.")

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }

    params = {"model": DEEPGRAM_MODEL}
    body = {"text": text}

    # Use httpx for async HTTP — avoids blocking the event loop
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            DEEPGRAM_TTS_URL,
            headers=headers,
            params=params,
            json=body,
        )
        response.raise_for_status()
        return response.content  # raw mp3 bytes
