"""
Speech-to-Speech model adapters.

Provides a unified interface for end-to-end voice models that accept audio
input and produce audio output directly (no separate STT/TTS stages).

Supported models:
  - OpenAI gpt-4o-audio-preview (via Chat Completions with audio modality)
  - Gemini native audio (via google-genai SDK)
"""

import base64
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_S2S_PROVIDERS = {
    "openai": OpenAIS2S,
    "gemini": GeminiS2S,
}


def create_s2s(provider: str, **kwargs) -> BaseS2S:
    """
    Create an S2S instance from a provider name.

    Args:
        provider: One of 'openai', 'gemini'.
        **kwargs: Provider-specific arguments.

    Returns:
        Configured BaseS2S instance.
    """
    cls = _S2S_PROVIDERS.get(provider.lower())
    if cls is None:
        supported = ", ".join(_S2S_PROVIDERS.keys())
        raise ValueError(f"Unknown S2S provider '{provider}'. Supported: {supported}")
    return cls(**kwargs)
