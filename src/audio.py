import io
import logging

import httpx

from src.settings import settings

logger = logging.getLogger(__name__)


def transcribe_audio_with_mistral(
    audio_bytes: bytes,
) -> str:
    api_key = (
        settings.mistral_api_key.get_secret_value().strip()
        if settings.mistral_api_key
        else None
    )
    if not api_key:
        raise ValueError("Mistral API key is not set")

    files = {
        "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
    }
    data = {"model": "voxtral-mini-latest"}
    headers = {"x-api-key": api_key}

    # Make request to Mistral's transcription API
    with httpx.Client() as client:
        response = client.post(
            "https://api.mistral.ai/v1/audio/transcriptions",
            files=files,
            data=data,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        # Parse the response
        result = response.json()
        transcription = result.get("text", "").strip()

        logger.info(f"Transcription successful: {transcription[:100]}...")
        return transcription
