"""
Speech-to-Speech model adapters.

Provides a unified interface for end-to-end voice models that accept audio
input and produce audio output directly (no separate STT/TTS stages).

Supported models:
  - OpenAI gpt-4o-audio-preview (via Chat Completions with audio modality)
  - OpenAI Realtime gpt-realtime-1.5 (via WebSocket Realtime API)
  - Gemini native audio (via google-genai SDK)
"""

import base64
import io
import json
import logging
import os
import struct
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class S2SToolCall:
    """A tool/function call returned by an S2S model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class S2SResult:
    """Result from a speech-to-speech model turn."""
    text: str
    audio_data: Optional[bytes] = None
    audio_format: str = "wav"
    tool_calls: List[S2SToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    provider: str = ""
    audio_id: Optional[str] = None  # For OpenAI audio response referencing
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseS2S(ABC):
    """Abstract base class for speech-to-speech models."""

    provider_name: str = "base"

    @abstractmethod
    def process_turn(
        self,
        audio_data: bytes,
        audio_format: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system_prompt: str,
    ) -> S2SResult:
        """
        Process a single conversation turn with audio input.

        Args:
            audio_data: User's audio input bytes.
            audio_format: Audio format (wav, mp3, etc.).
            messages: Previous conversation messages for context.
            tools: Tool/function definitions (OpenAI format).
            system_prompt: System prompt for the model.

        Returns:
            S2SResult with response text, audio, and any tool calls.
        """
        pass

    @abstractmethod
    def send_tool_response(
        self,
        tool_call_id: str,
        result: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system_prompt: str,
    ) -> S2SResult:
        """
        Send a tool call response and get the model's follow-up.

        Args:
            tool_call_id: ID of the tool call being responded to.
            result: Tool execution result.
            messages: Full conversation messages including the tool call.
            tools: Tool/function definitions.
            system_prompt: System prompt.

        Returns:
            S2SResult with the model's follow-up response.
        """
        pass


class OpenAIS2S(BaseS2S):
    """
    OpenAI speech-to-speech via Chat Completions with audio modality.

    Uses gpt-4o-audio-preview (or similar) with modalities=["text", "audio"].
    Sends audio input, receives both text and audio output.
    Supports function calling in audio mode.
    """

    provider_name = "openai"

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-4o-audio-preview",
        voice: str = "alloy",
        audio_format: str = "wav",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.params = params or {}

        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)

    def process_turn(
        self,
        audio_data: bytes,
        audio_format: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system_prompt: str,
    ) -> S2SResult:
        # Encode audio to base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        # Build the user message with audio content
        user_msg = {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": audio_format,
                    },
                }
            ],
        }

        # Build full message list
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)
        full_messages.append(user_msg)

        return self._call_api(full_messages, tools)

    def send_tool_response(
        self,
        tool_call_id: str,
        result: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system_prompt: str,
    ) -> S2SResult:
        # Build full message list — the engine already appended the tool
        # response to `messages`, so we just prepend the system prompt.
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        return self._call_api(full_messages, tools)

    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> S2SResult:
        start = time.perf_counter()

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "modalities": ["text", "audio"],
            "audio": {"voice": self.voice, "format": self.audio_format},
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = [{"type": "function", "function": f} for f in tools]

        response = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        message = response.choices[0].message

        # Extract text
        text = message.content or ""

        # Extract audio
        audio_bytes = None
        audio_id = None
        if hasattr(message, "audio") and message.audio:
            if hasattr(message.audio, "data") and message.audio.data:
                audio_bytes = base64.b64decode(message.audio.data)
            if hasattr(message.audio, "transcript") and message.audio.transcript:
                text = message.audio.transcript
            if hasattr(message.audio, "id"):
                audio_id = message.audio.id

        # Extract tool calls
        s2s_tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if isinstance(
                        tc.function.arguments, str
                    ) else tc.function.arguments
                except (json.JSONDecodeError, TypeError):
                    args = {}
                s2s_tool_calls.append(S2SToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return S2SResult(
            text=text,
            audio_data=audio_bytes,
            audio_format=self.audio_format,
            tool_calls=s2s_tool_calls,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            audio_id=audio_id,
        )


class GeminiS2S(BaseS2S):
    """
    Gemini speech-to-speech via google-genai SDK with native audio.

    Uses gemini-2.0-flash or gemini-2.5-flash with audio input/output.
    """

    provider_name = "gemini"

    def __init__(
        self,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = "gemini-2.0-flash",
        voice: str = "Kore",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.voice = voice
        self.params = params or {}

        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def process_turn(
        self,
        audio_data: bytes,
        audio_format: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system_prompt: str,
    ) -> S2SResult:
        from google.genai import types

        start = time.perf_counter()

        # Build contents
        contents = []

        # Add previous conversation as text context
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            content_text = msg.get("content", "")
            if isinstance(content_text, str) and content_text:
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content_text)],
                ))

        # Add current audio input
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "webm": "audio/webm",
            "ogg": "audio/ogg",
        }
        mime_type = mime_map.get(audio_format, f"audio/{audio_format}")

        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_bytes(data=audio_data, mime_type=mime_type)],
        ))

        # Build tool declarations
        gemini_tools = None
        if tools:
            func_declarations = []
            for t in tools:
                params_schema = t.get("parameters", {"type": "object", "properties": {}})
                func_declarations.append(types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=params_schema,
                ))
            gemini_tools = [types.Tool(function_declarations=func_declarations)]

        # Configure generation
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.params.get("temperature", 0.2),
        )

        if gemini_tools:
            config.tools = gemini_tools

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        # Extract text and tool calls
        text = ""
        s2s_tool_calls = []

        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    s2s_tool_calls.append(S2SToolCall(
                        id=f"gemini_call_{fc.name}_{int(time.time() * 1000)}",
                        name=fc.name,
                        arguments=args,
                    ))

        return S2SResult(
            text=text,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            tool_calls=s2s_tool_calls,
        )

    def send_tool_response(
        self,
        tool_call_id: str,
        result: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system_prompt: str,
    ) -> S2SResult:
        from google.genai import types

        start = time.perf_counter()

        # Build contents from message history
        contents = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            content_text = msg.get("content", "")
            if isinstance(content_text, str) and content_text:
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content_text)],
                ))

        # Add function response
        # Extract function name from tool_call_id (format: gemini_call_{name})
        func_name = tool_call_id.replace("gemini_call_", "", 1)
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_function_response(
                name=func_name,
                response=result,
            )],
        ))

        # Build tool declarations
        gemini_tools = None
        if tools:
            func_declarations = []
            for t in tools:
                params_schema = t.get("parameters", {"type": "object", "properties": {}})
                func_declarations.append(types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=params_schema,
                ))
            gemini_tools = [types.Tool(function_declarations=func_declarations)]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.params.get("temperature", 0.2),
        )
        if gemini_tools:
            config.tools = gemini_tools

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        text = ""
        s2s_tool_calls = []
        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    s2s_tool_calls.append(S2SToolCall(
                        id=f"gemini_call_{fc.name}_{int(time.time() * 1000)}",
                        name=fc.name,
                        arguments=args,
                    ))

        return S2SResult(
            text=text,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
            tool_calls=s2s_tool_calls,
        )


class OpenAIRealtimeS2S(BaseS2S):
    """
    OpenAI speech-to-speech via the Realtime WebSocket API.

    Uses gpt-realtime-1.5 (or similar) over a persistent WebSocket connection.
    Audio is exchanged as raw PCM16 24 kHz mono, converted to/from WAV at the
    boundary.  Turn detection is disabled; we commit input and create responses
    explicitly.

    Requires: ``pip install websocket-client``
    """

    provider_name = "openai_realtime"

    # Realtime API audio sample rate
    _SAMPLE_RATE = 24000
    _SAMPLE_WIDTH = 2  # 16-bit PCM
    _CHANNELS = 1

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-realtime-1.5",
        voice: str = "alloy",
        audio_format: str = "wav",
        params: Optional[Dict[str, Any]] = None,
    ):
        # Init connection state first so __del__ is safe even if __init__ fails
        self._ws = None
        self._session_configured = False

        try:
            import websocket  # noqa: F401
        except ImportError:
            raise ImportError(
                "websocket-client is required for OpenAI Realtime API. "
                "Install it with: pip install websocket-client"
            )

        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.params = params or {}

    # -- audio helpers -------------------------------------------------------

    @staticmethod
    def _wav_to_pcm24k(wav_bytes: bytes) -> bytes:
        """Read a WAV file and return raw PCM16 24 kHz mono bytes."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        # Convert to mono if stereo
        if n_channels == 2 and sampwidth == 2:
            samples = struct.unpack(f"<{len(raw) // 2}h", raw)
            mono = []
            for i in range(0, len(samples), 2):
                mono.append((samples[i] + samples[i + 1]) // 2)
            raw = struct.pack(f"<{len(mono)}h", *mono)
            n_channels = 1

        # Resample to 24 kHz if needed (simple linear interpolation)
        if framerate != 24000:
            samples = struct.unpack(f"<{len(raw) // 2}h", raw)
            ratio = framerate / 24000
            new_len = int(len(samples) / ratio)
            resampled = []
            for i in range(new_len):
                src = i * ratio
                idx = int(src)
                frac = src - idx
                if idx + 1 < len(samples):
                    val = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
                else:
                    val = samples[min(idx, len(samples) - 1)]
                resampled.append(max(-32768, min(32767, val)))
            raw = struct.pack(f"<{len(resampled)}h", *resampled)

        return raw

    @classmethod
    def _pcm_to_wav(cls, pcm_bytes: bytes) -> bytes:
        """Wrap raw PCM16 24 kHz mono bytes in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(cls._CHANNELS)
            wf.setsampwidth(cls._SAMPLE_WIDTH)
            wf.setframerate(cls._SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    # -- WebSocket management ------------------------------------------------

    def _ensure_connection(self, system_prompt: str, tools: Optional[List[Dict[str, Any]]]):
        """Create the WebSocket connection and send session.update if needed."""
        if self._ws is not None:
            return

        import websocket

        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1",
        ]
        self._ws = websocket.create_connection(url, header=headers, timeout=60)

        # Configure the session (flat fields per Realtime API spec)
        session_cfg: Dict[str, Any] = {
            "modalities": ["text", "audio"],
            "instructions": system_prompt,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
        }

        if tools:
            session_cfg["tools"] = [
                {
                    "type": "function",
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]

        self._ws.send(json.dumps({"type": "session.update", "session": session_cfg}))

        # Read until we get session.updated confirmation (discard other events)
        while True:
            evt = json.loads(self._ws.recv())
            if evt.get("type") == "session.updated":
                break
            if evt.get("type") == "error":
                raise RuntimeError(f"Realtime session error: {evt}")

        self._session_configured = True

    def _read_response(self) -> S2SResult:
        """Read server events until response.done; return aggregated result."""
        start = time.perf_counter()

        audio_chunks: list[str] = []
        transcript_parts: list[str] = []
        tool_calls: list[S2SToolCall] = []

        while True:
            raw = self._ws.recv()
            evt = json.loads(raw)
            evt_type = evt.get("type", "")

            if evt_type == "response.audio.delta":
                delta = evt.get("delta", "")
                if delta:
                    audio_chunks.append(delta)

            elif evt_type == "response.audio_transcript.delta":
                delta = evt.get("delta", "")
                if delta:
                    transcript_parts.append(delta)

            elif evt_type == "response.done":
                # Extract function calls from the response output items
                response_obj = evt.get("response", {})
                for item in response_obj.get("output", []):
                    if item.get("type") == "function_call":
                        try:
                            args = json.loads(item.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        tool_calls.append(S2SToolCall(
                            id=item.get("call_id", ""),
                            name=item.get("name", ""),
                            arguments=args,
                        ))
                break

            elif evt_type == "error":
                # Log but don't break — recoverable errors are common;
                # we keep waiting for response.done.
                logger.warning("Realtime API error event: %s", evt)

        latency_ms = (time.perf_counter() - start) * 1000

        # Decode audio
        audio_data = None
        if audio_chunks:
            pcm_bytes = base64.b64decode("".join(audio_chunks))
            audio_data = self._pcm_to_wav(pcm_bytes)

        text = "".join(transcript_parts)

        return S2SResult(
            text=text,
            audio_data=audio_data,
            audio_format="wav",
            tool_calls=tool_calls,
            latency_ms=round(latency_ms, 1),
            provider=self.provider_name,
        )

    # -- BaseS2S interface ---------------------------------------------------

    def process_turn(
        self,
        audio_data: bytes,
        audio_format: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system_prompt: str,
    ) -> S2SResult:
        self._ensure_connection(system_prompt, tools)

        # Convert WAV input to raw PCM16 24 kHz mono
        pcm = self._wav_to_pcm24k(audio_data)
        audio_b64 = base64.b64encode(pcm).decode("utf-8")

        # Send audio input
        self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))
        self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Wait for commit confirmation before creating response
        while True:
            evt = json.loads(self._ws.recv())
            if evt.get("type") == "input_audio_buffer.committed":
                break
            if evt.get("type") == "error":
                err = evt.get("error", {})
                raise RuntimeError(f"Audio buffer commit failed: {err.get('message', evt)}")

        self._ws.send(json.dumps({"type": "response.create"}))

        return self._read_response()

    def send_tool_response(
        self,
        tool_call_id: str,
        result: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system_prompt: str,
    ) -> S2SResult:
        self._ensure_connection(system_prompt, tools)

        # Send the function call output
        self._ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": json.dumps(result),
            },
        }))
        self._ws.send(json.dumps({"type": "response.create"}))

        return self._read_response()

    def close(self):
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._session_configured = False

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_S2S_PROVIDERS = {
    "openai": OpenAIS2S,
    "openai_realtime": OpenAIRealtimeS2S,
    "gemini": GeminiS2S,
}


def create_s2s(provider: str, **kwargs) -> BaseS2S:
    """
    Create an S2S instance from a provider name.

    Args:
        provider: One of 'openai', 'openai_realtime', 'gemini'.
        **kwargs: Provider-specific arguments.

    Returns:
        Configured BaseS2S instance.
    """
    cls = _S2S_PROVIDERS.get(provider.lower())
    if cls is None:
        supported = ", ".join(_S2S_PROVIDERS.keys())
        raise ValueError(f"Unknown S2S provider '{provider}'. Supported: {supported}")
    return cls(**kwargs)
