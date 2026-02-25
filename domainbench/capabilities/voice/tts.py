"""
Text-to-Speech service adapters.

Provides a unified interface for synthesizing speech from text.
Supported providers: OpenAI TTS, ElevenLabs, Google TTS.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import httpx

MAX_RETRIES = 3
RETRY_BACKOFF = 1.0


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
        raise last_err
    return wrapper


@dataclass
class TTSResult:
    """Result from a text-to-speech synthesis."""
    audio_data: bytes
    latency_ms: float
    provider: str
    format: str = "wav"
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseTTS(ABC):
    """Abstract base class for TTS services."""

    provider_name: str = "base"

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech.

        Returns:
            TTSResult with audio bytes and latency.
        """
        pass


class OpenAITTS(BaseTTS):
    """OpenAI TTS API (tts-1, tts-1-hd)."""

    provider_name = "openai"

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "wav",
        speed: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = speed
        self.params = params or {}

    @_retry_on_transient
    def synthesize(self, text: str) -> TTSResult:
        start = time.perf_counter()

        body = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
            "speed": self.speed,
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000

        return TTSResult(
            audio_data=resp.content,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            format=self.response_format,
        )


class ElevenLabsTTS(BaseTTS):
    """ElevenLabs TTS API."""

    provider_name = "elevenlabs"

    def __init__(
        self,
        api_key_env: str = "ELEVENLABS_API_KEY",
        model: str = "eleven_flash_v2_5",
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        output_format: str = "mp3_44100_128",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.voice_id = voice_id
        self.output_format = output_format
        self.params = params or {}

    @_retry_on_transient
    def synthesize(self, text: str) -> TTSResult:
        start = time.perf_counter()

        body = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.params.get("stability", 0.7),
                "similarity_boost": self.params.get("similarity_boost", 0.8),
            },
        }

        # Optional voice settings
        for key in ("style", "use_speaker_boost"):
            if key in self.params:
                body["voice_settings"][key] = self.params[key]

        query_params = {"output_format": self.output_format}
        if "optimize_streaming_latency" in self.params:
            query_params["optimize_streaming_latency"] = str(
                self.params["optimize_streaming_latency"]
            )

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=body,
                params=query_params,
            )
            resp.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000

        # Determine format from output_format
        fmt = "mp3" if "mp3" in self.output_format else "wav"

        return TTSResult(
            audio_data=resp.content,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            format=fmt,
        )


class GoogleTTS(BaseTTS):
    """Google Cloud Text-to-Speech API."""

    provider_name = "google"

    def __init__(
        self,
        api_key_env: str = "GOOGLE_API_KEY",
        model: str = "en-US-Journey-D",
        language_code: str = "en-US",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.language_code = language_code
        self.params = params or {}

    @_retry_on_transient
    def synthesize(self, text: str) -> TTSResult:
        import base64

        start = time.perf_counter()

        body = {
            "input": {"text": text},
            "voice": {
                "languageCode": self.language_code,
                "name": self.model,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": self.params.get("sample_rate", 24000),
            },
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_key}",
                json=body,
            )
            resp.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000
        result = resp.json()

        audio_data = base64.b64decode(result.get("audioContent", ""))

        return TTSResult(
            audio_data=audio_data,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            format="wav",
            raw=result,
        )


class CartesiaTTS(BaseTTS):
    """Cartesia TTS API (Sonic)."""

    provider_name = "cartesia"

    def __init__(
        self,
        api_key_env: str = "CARTESIA_API_KEY",
        model: str = "sonic-2",
        voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.voice_id = voice_id
        self.params = params or {}

    @_retry_on_transient
    def synthesize(self, text: str) -> TTSResult:
        start = time.perf_counter()

        body = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": self.params.get("sample_rate", 24000),
            },
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "X-API-Key": self.api_key,
                    "Cartesia-Version": "2024-06-10",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000

        return TTSResult(
            audio_data=resp.content,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            format="wav",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TTS_PROVIDERS = {
    "openai": OpenAITTS,
    "elevenlabs": ElevenLabsTTS,
    "google": GoogleTTS,
    "cartesia": CartesiaTTS,
}


def create_tts(provider: str, **kwargs) -> BaseTTS:
    """
    Create a TTS instance from a provider name.

    Args:
        provider: One of 'openai', 'elevenlabs', 'google', 'cartesia'.
        **kwargs: Provider-specific arguments.

    Returns:
        Configured BaseTTS instance.
    """
    cls = _TTS_PROVIDERS.get(provider.lower())
    if cls is None:
        supported = ", ".join(_TTS_PROVIDERS.keys())
        raise ValueError(f"Unknown TTS provider '{provider}'. Supported: {supported}")
    return cls(**kwargs)
