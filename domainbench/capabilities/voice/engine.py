"""
Voice benchmark engine — sequential multi-turn execution.

Orchestrates running a voice scenario against one of three pipeline types:
  - text:             Text In -> LLM -> Text Out
  - cascaded:         Text In -> [TTS] -> Audio -> [STT] -> Text -> [LLM] -> Text -> [TTS] -> Audio
  - speech_to_speech: Text In -> [TTS] -> Audio -> [S2S Model] -> Audio + Text

Records per-turn transcripts, tool calls, per-stage latency, and delegates
to the judge/scorer for evaluation.
"""

import json
import os
import struct
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional

from domainbench.capabilities.voice.config import (
    VoiceScenario,
    VoiceTurnResult,
    VoiceJudgment,
    VoiceRunResult,
    PipelineConfig,
)
from domainbench.capabilities.voice.judge import VoiceJudge
from domainbench.capabilities.voice.scorer import score_run, compute_latency_stats
from domainbench.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# Tool-schema helpers
# ---------------------------------------------------------------------------

def _sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively fix common schema issues (e.g., array missing 'items')."""
    if not isinstance(schema, dict):
        return schema
    result = dict(schema)
    if result.get("type") == "array" and "items" not in result:
        result["items"] = {"type": "string"}
    if "properties" in result and isinstance(result["properties"], dict):
        result["properties"] = {
            k: _sanitize_schema(v) for k, v in result["properties"].items()
        }
    if "items" in result and isinstance(result["items"], dict):
        result["items"] = _sanitize_schema(result["items"])
    return result


def _build_tools_for_provider(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert scenario tool definitions to OpenAI-compatible function schemas."""
    openai_tools = []
    for tool in tools:
        fn = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
        }
        if "parameters" in tool:
            fn["parameters"] = _sanitize_schema(tool["parameters"])
        else:
            fn["parameters"] = {"type": "object", "properties": {}}
        openai_tools.append(fn)
    return openai_tools


def _extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool calls from a provider response.

    Handles OpenAI, Anthropic, and Gemini response formats.
    """
    calls: List[Dict[str, Any]] = []

    # OpenAI format
    raw = response.get("raw")
    if raw and hasattr(raw, "choices"):
        for choice in raw.choices:
            msg = choice.message if hasattr(choice, "message") else None
            if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.function
                    try:
                        args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    calls.append({"name": fn.name, "arguments": args})

    # Anthropic format (from tool_calls in normalized response)
    if not calls:
        for tc in response.get("tool_calls", []):
            name = tc.get("name", tc.get("function", {}).get("name", ""))
            args = tc.get("arguments", tc.get("input", tc.get("function", {}).get("arguments", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if name:
                calls.append({"name": name, "arguments": args})

    return calls


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def _load_audio_file(path: str) -> bytes:
    """Load audio bytes from a file path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return p.read_bytes()


def _get_audio_format(path: str) -> str:
    """Infer audio format from file extension."""
    ext = Path(path).suffix.lower().lstrip(".")
    return ext if ext else "wav"


# ---------------------------------------------------------------------------
# VoiceEngine
# ---------------------------------------------------------------------------

class VoiceEngine:
    """
    Orchestrates sequential multi-turn voice benchmark execution.

    Supports three pipeline types:
      - text: Direct text to LLM (existing behavior).
      - cascaded: TTS(input) -> STT -> LLM -> TTS(response).
      - speech_to_speech: TTS(input) -> S2S Model -> response.
    """

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        model: Optional[str] = None,
        judge: Optional[VoiceJudge] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        stt=None,        # BaseSTT instance (for cascaded)
        tts=None,        # BaseTTS instance (for cascaded response TTS)
        input_tts=None,  # BaseTTS instance (for synthesizing user audio)
        s2s=None,        # BaseS2S instance (for speech-to-speech)
        save_audio: bool = False,
        audio_dir: Optional[Path] = None,
    ):
        self.provider = provider
        self.model = model
        self.judge = judge
        self.pipeline_config = pipeline_config or PipelineConfig(type="text")
        self.stt = stt
        self.tts = tts
        self.input_tts = input_tts
        self.s2s = s2s
        self.save_audio = save_audio
        self.audio_dir = audio_dir

    # ------------------------------------------------------------------
    # Audio persistence
    # ------------------------------------------------------------------

    def _save_audio(
        self,
        audio_data: Optional[bytes],
        fmt: str,
        scenario_id: str,
        run_index: int,
        turn_index: int,
        stage: str,
    ) -> Optional[str]:
        """Write audio bytes to disk and return the path relative to audio_dir.

        Returns None when saving is disabled or there is no data to write.
        """
        if not self.save_audio or not self.audio_dir or not audio_data:
            return None

        subdir = self.audio_dir / scenario_id / f"run_{run_index}"
        subdir.mkdir(parents=True, exist_ok=True)

        filename = f"turn_{turn_index}_{stage}.{fmt}"
        filepath = subdir / filename
        filepath.write_bytes(audio_data)
        return str(filepath.relative_to(self.audio_dir))

    # ------------------------------------------------------------------
    # Silence padding detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_silence_pad(
        audio_data: Optional[bytes],
        audio_format: str = "wav",
        window_ms: float = 10.0,
        threshold: float = 0.02,
    ) -> Optional[float]:
        """Detect leading silence duration in audio data (WAV only).

        Reads PCM samples from a WAV file, divides into short windows,
        computes RMS energy per window, and returns the duration (in ms)
        before the first window that exceeds *threshold* (relative to
        the maximum possible amplitude for the sample width).

        Returns ``None`` when audio_data is None, empty, or not WAV.
        """
        if not audio_data or audio_format.lower() not in ("wav", "wave"):
            return None

        try:
            with wave.open(BytesIO(audio_data), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()  # bytes per sample
                frame_rate = wf.getframerate()
                n_frames = wf.getnframes()

                if n_frames == 0 or frame_rate == 0:
                    return 0.0

                raw = wf.readframes(n_frames)
        except Exception:
            return None

        # Determine struct format for unpacking samples
        if sample_width == 1:
            fmt_char = "b"  # signed 8-bit
            max_val = 127.0
        elif sample_width == 2:
            fmt_char = "h"  # signed 16-bit
            max_val = 32767.0
        elif sample_width == 4:
            fmt_char = "i"  # signed 32-bit
            max_val = 2147483647.0
        else:
            return None

        total_samples = len(raw) // sample_width
        if total_samples == 0:
            return 0.0

        try:
            samples = struct.unpack(f"<{total_samples}{fmt_char}", raw[:total_samples * sample_width])
        except struct.error:
            return None

        # Convert to mono by averaging channels
        if n_channels > 1:
            mono = []
            for i in range(0, len(samples), n_channels):
                chunk = samples[i:i + n_channels]
                mono.append(sum(chunk) / n_channels)
            samples = mono

        # Window size in samples
        window_samples = max(1, int(frame_rate * window_ms / 1000.0))

        # Scan windows for first one exceeding threshold
        for start in range(0, len(samples), window_samples):
            window = samples[start:start + window_samples]
            if not window:
                break
            rms = (sum(s * s for s in window) / len(window)) ** 0.5
            normalized_rms = rms / max_val
            if normalized_rms > threshold:
                return round(start / frame_rate * 1000.0, 1)

        # Entire audio is silent
        return round(len(samples) / frame_rate * 1000.0, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario: VoiceScenario,
        run_index: int = 0,
        verbose: bool = False,
    ) -> VoiceRunResult:
        """
        Execute a single run of a scenario.

        Routes to the appropriate pipeline handler based on config type.
        """
        ptype = self.pipeline_config.type

        if ptype == "text":
            turn_results = self._run_text(scenario, verbose)
        elif ptype == "cascaded":
            turn_results = self._run_cascaded(scenario, verbose, run_index=run_index)
        elif ptype == "speech_to_speech":
            turn_results = self._run_s2s(scenario, verbose, run_index=run_index)
        else:
            raise ValueError(f"Unknown pipeline type: {ptype}")

        # Judge the run
        judgments: List[VoiceJudgment] = []
        if self.judge:
            judgments = self.judge.evaluate(scenario, turn_results)

        # Score the run
        include_turn_taking = ptype != "text"
        scores = score_run(judgments, include_turn_taking=include_turn_taking)
        latency_stats = compute_latency_stats(turn_results)

        return VoiceRunResult(
            scenario_id=scenario.id,
            model_name=self._model_display_name(),
            pipeline_type=ptype,
            run_index=run_index,
            turn_results=turn_results,
            judgments=judgments,
            pass_rate=scores.get("pass_rate", 0.0),
            dimension_scores=scores.get("dimension_scores", {}),
            latency_stats=latency_stats,
        )

    def run_multiple(
        self,
        scenario: VoiceScenario,
        num_runs: int = 1,
        verbose: bool = False,
    ) -> List[VoiceRunResult]:
        """Execute multiple runs of the same scenario for consistency measurement."""
        results = []
        for run_idx in range(num_runs):
            if verbose:
                print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
            result = self.run_scenario(scenario, run_index=run_idx, verbose=verbose)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Text pipeline (existing behavior)
    # ------------------------------------------------------------------

    def _run_text(
        self, scenario: VoiceScenario, verbose: bool
    ) -> List[VoiceTurnResult]:
        """Text-only pipeline: send text turns directly to LLM."""
        system_content = scenario.system_prompt
        if scenario.knowledge_base:
            system_content += "\n\n## Knowledge Base\n\n" + scenario.knowledge_base

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]
        tools = _build_tools_for_provider(scenario.tools) if scenario.tools else []
        turn_results: List[VoiceTurnResult] = []

        for i, turn in enumerate(scenario.turns):
            if verbose:
                print(f"  Turn {i}/{len(scenario.turns) - 1}: {turn.input[:60]}...")

            messages.append({"role": "user", "content": turn.input})

            # Call LLM with retry
            response = self._call_llm_with_retry(messages, tools, verbose, i)
            llm_latency = response.pop("_latency_ms", 0)

            assistant_text = response.get("content", "")
            tool_calls = _extract_tool_calls(response)
            usage = response.get("usage", {})

            turn_result = VoiceTurnResult(
                turn_index=i,
                user_input=turn.input,
                assistant_text=assistant_text or "",
                golden_text=turn.golden_text,
                tool_calls=tool_calls,
                expected_function_call=turn.required_function_call,
                function_call_response=turn.function_call_response,
                ttfb_ms=llm_latency,
                latency_ms=llm_latency,
                llm_latency_ms=llm_latency,
                tokens={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )
            turn_results.append(turn_result)

            # Update context
            self._append_assistant_context(messages, assistant_text, tool_calls, turn, i)

        return turn_results

    # ------------------------------------------------------------------
    # Cascaded pipeline: TTS(input) -> STT -> LLM -> TTS(response)
    # ------------------------------------------------------------------

    def _run_cascaded(
        self, scenario: VoiceScenario, verbose: bool, run_index: int = 0,
    ) -> List[VoiceTurnResult]:
        """Cascaded pipeline: STT -> LLM -> TTS with real audio processing."""
        if not self.stt:
            raise ValueError("Cascaded pipeline requires an STT service. Configure 'stt' in pipeline YAML.")
        if not self.provider:
            raise ValueError("Cascaded pipeline requires an LLM provider. Configure 'llm' in pipeline YAML.")

        # Input TTS for synthesizing user audio from text
        synth_tts = self.input_tts or self.tts
        if not synth_tts:
            raise ValueError(
                "Cascaded pipeline requires a TTS service for synthesizing input audio. "
                "Configure 'tts' or 'input_tts' in pipeline YAML, or provide audio_file "
                "paths in your scenario."
            )

        system_content = scenario.system_prompt
        if scenario.knowledge_base:
            system_content += "\n\n## Knowledge Base\n\n" + scenario.knowledge_base

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]
        tools = _build_tools_for_provider(scenario.tools) if scenario.tools else []
        turn_results: List[VoiceTurnResult] = []

        for i, turn in enumerate(scenario.turns):
            if verbose:
                print(f"  Turn {i}/{len(scenario.turns) - 1}: {turn.input[:60]}...")

            # --- Step 1: Get input audio ---
            input_synth_latency = 0.0
            audio_format = "wav"
            audio_data = None

            try:
                if turn.audio_file:
                    if verbose:
                        print(f"    Loading audio: {turn.audio_file}")
                    audio_data = _load_audio_file(turn.audio_file)
                    audio_format = _get_audio_format(turn.audio_file)
                else:
                    if verbose:
                        print(f"    Synthesizing input audio...")
                    tts_result = synth_tts.synthesize(turn.input)
                    audio_data = tts_result.audio_data
                    audio_format = tts_result.format
                    input_synth_latency = tts_result.latency_ms
            except Exception as e:
                if verbose:
                    print(f"    [ERROR] Audio preparation failed: {e}")
                # Fall back to text input directly (skip STT)
                audio_data = None

            # Save input audio
            input_tts_path = self._save_audio(
                audio_data, audio_format, scenario.id, run_index, i, "input_tts",
            )

            # --- Step 2: STT (transcribe) ---
            stt_latency = 0.0
            if audio_data is not None:
                try:
                    if verbose:
                        print(f"    STT transcribing...")
                    stt_result = self.stt.transcribe(audio_data, format=audio_format)
                    transcribed_text = stt_result.text
                    stt_latency = stt_result.latency_ms
                    if verbose:
                        print(f"    STT result: {transcribed_text[:80]}...")
                except Exception as e:
                    if verbose:
                        print(f"    [ERROR] STT failed: {e}, using original text")
                    transcribed_text = turn.input
            else:
                # No audio available — use original text as fallback
                transcribed_text = turn.input
                if verbose:
                    print(f"    Using original text (no audio): {turn.input[:60]}...")

            # --- Step 3: LLM processing (using transcribed text) ---
            messages.append({"role": "user", "content": transcribed_text})

            response = self._call_llm_with_retry(messages, tools, verbose, i)
            llm_latency = response.pop("_latency_ms", 0)

            assistant_text = response.get("content", "")
            tool_calls = _extract_tool_calls(response)
            usage = response.get("usage", {})

            # --- Step 4: TTS (synthesize response) ---
            tts_response_latency = 0.0
            response_tts_path = None
            response_audio_data = None
            response_audio_format = "wav"
            if self.tts and assistant_text:
                if verbose:
                    print(f"    TTS synthesizing response...")
                try:
                    resp_tts = self.tts.synthesize(assistant_text)
                    tts_response_latency = resp_tts.latency_ms
                    response_audio_data = resp_tts.audio_data
                    response_audio_format = resp_tts.format
                    response_tts_path = self._save_audio(
                        resp_tts.audio_data, resp_tts.format,
                        scenario.id, run_index, i, "response_tts",
                    )
                except Exception as e:
                    if verbose:
                        print(f"    [WARN] Response TTS failed: {e}")

            # Total pipeline latency = STT + LLM + TTS (excluding input synthesis)
            total_latency = stt_latency + llm_latency + tts_response_latency

            # Silence padding detection on response audio
            silence_pad = self._detect_silence_pad(
                response_audio_data, response_audio_format,
            )

            # V2V = pipeline processing latency + silence padding
            v2v = total_latency + (silence_pad or 0.0) if total_latency > 0 else None

            # Build audio_files dict
            audio_files = None
            if self.save_audio:
                af = {}
                if input_tts_path:
                    af["input_tts"] = input_tts_path
                    af["stt_input"] = input_tts_path  # same audio sent to STT
                if response_tts_path:
                    af["response_tts"] = response_tts_path
                if af:
                    audio_files = af

            turn_result = VoiceTurnResult(
                turn_index=i,
                user_input=turn.input,
                assistant_text=assistant_text or "",
                golden_text=turn.golden_text,
                tool_calls=tool_calls,
                expected_function_call=turn.required_function_call,
                function_call_response=turn.function_call_response,
                ttfb_ms=stt_latency + llm_latency,  # Time until first response byte
                latency_ms=total_latency,
                transcribed_input=transcribed_text,
                input_synthesis_latency_ms=input_synth_latency,
                stt_latency_ms=stt_latency,
                llm_latency_ms=llm_latency,
                tts_latency_ms=tts_response_latency,
                tokens={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                audio_files=audio_files,
                silence_pad_ms=silence_pad,
                v2v_ms=round(v2v, 1) if v2v is not None else None,
            )
            turn_results.append(turn_result)

            # Update context
            self._append_assistant_context(messages, assistant_text, tool_calls, turn, i)

        return turn_results

    # ------------------------------------------------------------------
    # Speech-to-Speech pipeline: TTS(input) -> S2S Model -> response
    # ------------------------------------------------------------------

    def _run_s2s(
        self, scenario: VoiceScenario, verbose: bool, run_index: int = 0,
    ) -> List[VoiceTurnResult]:
        """Speech-to-speech pipeline: end-to-end audio model."""
        if not self.s2s:
            raise ValueError(
                "Speech-to-speech pipeline requires an S2S model. "
                "Configure 'model' in pipeline YAML."
            )

        # Input TTS for synthesizing user audio
        synth_tts = self.input_tts or self.tts
        if not synth_tts:
            raise ValueError(
                "Speech-to-speech pipeline requires a TTS service for synthesizing "
                "input audio. Configure 'input_tts' or 'tts' in pipeline YAML, "
                "or provide audio_file paths in your scenario."
            )

        system_content = scenario.system_prompt
        if scenario.knowledge_base:
            system_content += "\n\n## Knowledge Base\n\n" + scenario.knowledge_base

        tools = _build_tools_for_provider(scenario.tools) if scenario.tools else []

        # Conversation history for the S2S model (text-based context)
        s2s_messages: List[Dict[str, Any]] = []
        turn_results: List[VoiceTurnResult] = []

        for i, turn in enumerate(scenario.turns):
            if verbose:
                print(f"  Turn {i}/{len(scenario.turns) - 1}: {turn.input[:60]}...")

            # --- Step 1: Get input audio ---
            input_synth_latency = 0.0
            audio_format = "wav"

            try:
                if turn.audio_file:
                    audio_data = _load_audio_file(turn.audio_file)
                    audio_format = _get_audio_format(turn.audio_file)
                else:
                    if verbose:
                        print(f"    Synthesizing input audio...")
                    tts_result = synth_tts.synthesize(turn.input)
                    audio_data = tts_result.audio_data
                    audio_format = tts_result.format
                    input_synth_latency = tts_result.latency_ms
            except Exception as e:
                if verbose:
                    print(f"    [ERROR] Audio preparation failed: {e}")
                # Record error turn and continue
                turn_results.append(VoiceTurnResult(
                    turn_index=i,
                    user_input=turn.input,
                    assistant_text=f"[Audio error: {e}]",
                    golden_text=turn.golden_text,
                    expected_function_call=turn.required_function_call,
                    function_call_response=turn.function_call_response,
                    input_synthesis_latency_ms=input_synth_latency,
                ))
                s2s_messages.append({"role": "user", "content": turn.input})
                s2s_messages.append({"role": "assistant", "content": f"[Audio error: {e}]"})
                continue

            # Save input audio
            input_tts_path = self._save_audio(
                audio_data, audio_format, scenario.id, run_index, i, "input_tts",
            )

            # --- Step 2: Send audio to S2S model ---
            if verbose:
                print(f"    Sending to S2S model...")

            s2s_result = None
            for _attempt in range(3):
                try:
                    s2s_result = self.s2s.process_turn(
                        audio_data=audio_data,
                        audio_format=audio_format,
                        messages=s2s_messages,
                        tools=tools if tools else None,
                        system_prompt=system_content,
                    )
                    break
                except Exception as e:
                    if _attempt == 2:
                        if verbose:
                            print(f"    [ERROR] S2S turn {i} failed after 3 attempts: {e}")
                        from domainbench.capabilities.voice.s2s import S2SResult
                        s2s_result = S2SResult(
                            text=f"[Error: {e}]",
                            provider=self.s2s.provider_name,
                        )
                    else:
                        time.sleep(1)

            model_latency = s2s_result.latency_ms
            assistant_text = s2s_result.text or ""

            # Convert S2S tool calls to standard format
            tool_calls = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in s2s_result.tool_calls
            ]

            if verbose and tool_calls:
                print(f"    Tool calls: {[tc['name'] for tc in tool_calls]}")

            # Add the user turn to context first (correct message ordering)
            s2s_messages.append({
                "role": "user",
                "content": turn.input,
            })

            # --- Step 3: Handle tool calls ---
            if s2s_result.tool_calls:
                for tc in s2s_result.tool_calls:
                    # Determine tool response
                    fn_response = None
                    if turn.required_function_call and turn.function_call_response:
                        expected_name = turn.required_function_call.get("name", "")
                        if tc.name == expected_name:
                            fn_response = turn.function_call_response
                    if fn_response is None:
                        fn_response = {"status": "acknowledged"}

                    # Send tool response back to model
                    if verbose:
                        print(f"    Sending tool response for {tc.name}...")

                    # Add the assistant's tool call to context (OpenAI-compatible format)
                    # Note: content must be None (not empty string) for tool-call messages
                    s2s_messages.append({
                        "role": "assistant",
                        "content": assistant_text if assistant_text else None,
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }],
                    })
                    # Add the tool result to context
                    s2s_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(fn_response),
                    })

                    try:
                        followup = self.s2s.send_tool_response(
                            tool_call_id=tc.id,
                            result=fn_response,
                            messages=s2s_messages,
                            tools=tools if tools else None,
                            system_prompt=system_content,
                        )
                        # Use followup text if available
                        if followup.text:
                            assistant_text = followup.text
                        model_latency += followup.latency_ms
                    except Exception as e:
                        if verbose:
                            print(f"    [WARN] Tool response failed: {e}")

            # Add the final assistant response to context
            s2s_messages.append({
                "role": "assistant",
                "content": assistant_text,
            })

            # Save S2S output audio
            s2s_output_path = self._save_audio(
                s2s_result.audio_data, s2s_result.audio_format,
                scenario.id, run_index, i, "s2s_output",
            )

            # Silence padding detection on S2S output audio
            silence_pad = self._detect_silence_pad(
                s2s_result.audio_data, s2s_result.audio_format,
            )

            # V2V = model processing latency + silence padding
            v2v = model_latency + (silence_pad or 0.0) if model_latency > 0 else None

            # Build audio_files dict
            audio_files = None
            if self.save_audio:
                af = {}
                if input_tts_path:
                    af["input_tts"] = input_tts_path
                if s2s_output_path:
                    af["s2s_output"] = s2s_output_path
                if af:
                    audio_files = af

            turn_result = VoiceTurnResult(
                turn_index=i,
                user_input=turn.input,
                assistant_text=assistant_text,
                golden_text=turn.golden_text,
                tool_calls=tool_calls,
                expected_function_call=turn.required_function_call,
                function_call_response=turn.function_call_response,
                ttfb_ms=model_latency,
                latency_ms=model_latency + input_synth_latency,
                input_synthesis_latency_ms=input_synth_latency,
                audio_files=audio_files,
                silence_pad_ms=silence_pad,
                v2v_ms=round(v2v, 1) if v2v is not None else None,
            )
            turn_results.append(turn_result)

        return turn_results

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _model_display_name(self) -> str:
        """Build a display name for the model being benchmarked."""
        ptype = self.pipeline_config.type
        if ptype == "speech_to_speech" and self.s2s:
            return f"{self.s2s.provider_name}/{getattr(self.s2s, 'model', 'unknown')}"
        if self.provider and self.model:
            return f"{self.provider.name}/{self.model}"
        return "unknown"

    def _call_llm_with_retry(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        verbose: bool,
        turn_index: int,
    ) -> Dict[str, Any]:
        """Call the LLM with retry logic. Returns response dict with _latency_ms added."""
        response = None
        for _attempt in range(3):
            try:
                start = time.perf_counter()
                if tools:
                    try:
                        response = self.provider.function_call(
                            model=self.model,
                            messages=messages,
                            functions=tools,
                            temperature=0.2,
                        )
                    except NotImplementedError:
                        response = self.provider.chat_completion(
                            model=self.model,
                            messages=messages,
                            temperature=0.2,
                        )
                else:
                    response = self.provider.chat_completion(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                    )
                latency_ms = (time.perf_counter() - start) * 1000
                response["_latency_ms"] = round(latency_ms, 1)
                return response
            except Exception as e:
                if _attempt == 2:
                    if verbose:
                        print(f"    [ERROR] Turn {turn_index} LLM failed after 3 attempts: {e}")
                    return {
                        "content": f"[Error: {e}]",
                        "usage": {},
                        "_latency_ms": 0,
                    }
                time.sleep(1)
        return {"content": "", "usage": {}, "_latency_ms": 0}

    def _append_assistant_context(
        self,
        messages: List[Dict[str, Any]],
        assistant_text: str,
        tool_calls: List[Dict[str, Any]],
        turn,
        turn_index: int,
    ):
        """Append assistant response and tool responses to the conversation context."""
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_text or "",
        }

        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": f"call_{turn_index}_{j}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(
                            tc["arguments"] if isinstance(tc["arguments"], dict) else {}
                        ),
                    },
                }
                for j, tc in enumerate(tool_calls)
            ]
            messages.append(assistant_msg)

            # Inject tool responses
            for j, tc in enumerate(tool_calls):
                fn_response = None
                if turn.required_function_call and turn.function_call_response:
                    expected_name = turn.required_function_call.get("name", "")
                    if tc["name"] == expected_name:
                        fn_response = turn.function_call_response
                if fn_response is None:
                    fn_response = {"status": "acknowledged"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{turn_index}_{j}",
                    "content": json.dumps(fn_response),
                })
        else:
            messages.append(assistant_msg)
