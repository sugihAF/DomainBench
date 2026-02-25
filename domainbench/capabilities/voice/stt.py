"""
Speech-to-Text service adapters.

Provides a unified interface for transcribing audio to text.
Supported providers: OpenAI Whisper, Deepgram.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import httpx

MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds


def _retry_on_transient(func):
    """Decorator to retry on transient HTTP errors (429, 5xx)."""
    def wrapper(*args, **kwargs):
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    last_err = e
                    continue
                raise
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    last_err = e
                    continue
                raise
        raise last_err  # Should not reach here
    return wrapper


@dataclass
class STTResult:
    """Result from a speech-to-text transcription."""
    text: str
    latency_ms: float
    provider: str
    confidence: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseSTT(ABC):
    """Abstract base class for STT services."""

    provider_name: str = "base"

    @abstractmethod
    def transcribe(self, audio_data: bytes, format: str = "wav") -> STTResult:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes.
            format: Audio format (wav, mp3, webm, etc.).

        Returns:
            STTResult with transcribed text and latency.
        """
        pass


class WhisperSTT(BaseSTT):
    """OpenAI Whisper API for speech-to-text."""

    provider_name = "whisper"

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "whisper-1",
        language: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.language = language
        self.params = params or {}

    @_retry_on_transient
    def transcribe(self, audio_data: bytes, format: str = "wav") -> STTResult:
        start = time.perf_counter()

        # Map format to MIME type
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "webm": "audio/webm",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }
        mime_type = mime_map.get(format, f"audio/{format}")

        form_data = {"model": self.model}
        if self.language:
            form_data["language"] = self.language
        form_data.update({k: str(v) for k, v in self.params.items()})

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (f"audio.{format}", audio_data, mime_type)},
                data=form_data,
            )
            resp.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000
        result = resp.json()

        return STTResult(
            text=result.get("text", ""),
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            raw=result,
        )


class DeepgramSTT(BaseSTT):
    """Deepgram API for speech-to-text."""

    provider_name = "deepgram"

    def __init__(
        self,
        api_key_env: str = "DEEPGRAM_API_KEY",
        model: str = "nova-2",
        language: str = "en",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.language = language
        self.params = params or {}

    @_retry_on_transient
    def transcribe(self, audio_data: bytes, format: str = "wav") -> STTResult:
        start = time.perf_counter()

        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "webm": "audio/webm",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }
        content_type = mime_map.get(format, f"audio/{format}")

        query_params = {
            "model": self.model,
            "language": self.language,
            "smart_format": "true",
        }
        query_params.update({k: str(v) for k, v in self.params.items()})

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": content_type,
                },
                params=query_params,
                content=audio_data,
            )
            resp.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000
        result = resp.json()

        # Extract transcript from Deepgram response structure
        text = ""
        confidence = None
        channels = result.get("results", {}).get("channels", [])
        if channels:
            alternatives = channels[0].get("alternatives", [])
            if alternatives:
                text = alternatives[0].get("transcript", "")
                confidence = alternatives[0].get("confidence")

        return STTResult(
            text=text,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            confidence=confidence,
            raw=result,
        )


class GoogleSTT(BaseSTT):
    """Google Cloud Speech-to-Text API."""

    provider_name = "google"

    def __init__(
        self,
        api_key_env: str = "GOOGLE_API_KEY",
        model: str = "chirp",
        language: str = "en-US",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.language = language
        self.params = params or {}

    @_retry_on_transient
    def transcribe(self, audio_data: bytes, format: str = "wav") -> STTResult:
        import base64

        start = time.perf_counter()

        encoding_map = {
            "wav": "LINEAR16",
            "mp3": "MP3",
            "flac": "FLAC",
            "ogg": "OGG_OPUS",
            "webm": "WEBM_OPUS",
        }
        encoding = encoding_map.get(format, "LINEAR16")

        body = {
            "config": {
                "encoding": encoding,
                "languageCode": self.language,
                "model": self.model,
                **self.params,
            },
            "audio": {
                "content": base64.b64encode(audio_data).decode("utf-8"),
            },
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}",
                json=body,
            )
            resp.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000
        result = resp.json()

        text = ""
        confidence = None
        results = result.get("results", [])
        if results:
            alternatives = results[0].get("alternatives", [])
            if alternatives:
                text = alternatives[0].get("transcript", "")
                confidence = alternatives[0].get("confidence")

        return STTResult(
            text=text,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            confidence=confidence,
            raw=result,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STT_PROVIDERS = {
    "whisper": WhisperSTT,
    "deepgram": DeepgramSTT,
    "google": GoogleSTT,
}


def create_stt(provider: str, **kwargs) -> BaseSTT:
    """
    Create an STT instance from a provider name.

    Args:
        provider: One of 'whisper', 'deepgram', 'google'.
        **kwargs: Provider-specific arguments.

    Returns:
        Configured BaseSTT instance.
    """
    cls = _STT_PROVIDERS.get(provider.lower())
    if cls is None:
        supported = ", ".join(_STT_PROVIDERS.keys())
        raise ValueError(f"Unknown STT provider '{provider}'. Supported: {supported}")
    return cls(**kwargs)
