"""
Voice benchmark configuration models.

Defines Pydantic models for pipeline configurations, voice scenarios,
turn results, judgments, and run results.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline configuration models
# ---------------------------------------------------------------------------

class STTConfig(BaseModel):
    """Speech-to-Text service configuration."""
    provider: str
    model: str
    api_key_env: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """LLM service configuration."""
    provider: str
    model: str
    api_key_env: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None


class TTSConfig(BaseModel):
    """Text-to-Speech service configuration."""
    provider: str
    model: str
    api_key_env: Optional[str] = None
    voice_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class S2SModelConfig(BaseModel):
    """Speech-to-speech model configuration."""
    provider: str
    model: str
    api_key_env: Optional[str] = None
    voice: str = "alloy"
    audio_format: str = "wav"
    params: Dict[str, Any] = Field(default_factory=dict)


class InputTTSConfig(BaseModel):
    """
    TTS configuration for synthesizing user input audio.

    Used by cascaded and S2S pipelines when scenario turns don't have
    pre-recorded audio files. Defaults to OpenAI TTS.
    """
    provider: str = "openai"
    model: str = "tts-1"
    api_key_env: Optional[str] = None
    voice: str = "alloy"
    params: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """
    Voice pipeline configuration.

    Supports three pipeline types:
    - "text": LLM only (no audio). Use the ``llm`` field.
    - "cascaded": STT -> LLM -> TTS. Use ``stt``, ``llm``, ``tts`` fields.
    - "speech_to_speech": End-to-end model. Use the ``model`` field.
    """
    type: str = "text"
    stt: Optional[STTConfig] = None
    llm: Optional[LLMConfig] = None
    tts: Optional[TTSConfig] = None
    model: Optional[S2SModelConfig] = None  # For speech-to-speech
    input_tts: Optional[InputTTSConfig] = None  # For synthesizing user audio


# ---------------------------------------------------------------------------
# Voice scenario (dataset) models
# ---------------------------------------------------------------------------

class VoiceTurn(BaseModel):
    """A single turn in a voice scenario."""
    input: str
    golden_text: str
    required_function_call: Optional[Dict[str, Any]] = None
    function_call_response: Optional[Dict[str, Any]] = None
    audio_file: Optional[str] = None


class VoiceScenario(BaseModel):
    """
    A complete voice agent evaluation scenario.

    One scenario = one multi-turn conversation with system prompt,
    knowledge base, tool definitions, and turn sequence.
    """
    id: str
    domain: Optional[str] = None
    system_prompt: str
    knowledge_base: str
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    turns: List[VoiceTurn]


# ---------------------------------------------------------------------------
# Execution result models
# ---------------------------------------------------------------------------

class VoiceTurnResult(BaseModel):
    """Recorded result for a single executed turn."""
    turn_index: int
    user_input: str
    assistant_text: str
    golden_text: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    expected_function_call: Optional[Dict[str, Any]] = None
    function_call_response: Optional[Dict[str, Any]] = None
    ttfb_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    tokens: Optional[Dict[str, int]] = None
    # Per-stage latency (cascaded pipeline)
    transcribed_input: Optional[str] = None  # What STT produced
    input_synthesis_latency_ms: Optional[float] = None  # TTS for input audio
    stt_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    tts_latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Judging result models
# ---------------------------------------------------------------------------

class VoiceJudgment(BaseModel):
    """Binary judgment for a single turn across four dimensions."""
    turn_index: int
    turn_taking: bool = True
    tool_use_correct: bool = True
    instruction_following: bool = True
    kb_grounding: bool = True
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Aggregate run result model
# ---------------------------------------------------------------------------

class VoiceRunResult(BaseModel):
    """Complete result for one benchmark run (one scenario, one model)."""
    scenario_id: str
    model_name: str
    pipeline_type: str = "text"
    run_index: int = 0
    turn_results: List[VoiceTurnResult] = Field(default_factory=list)
    judgments: List[VoiceJudgment] = Field(default_factory=list)
    pass_rate: float = 0.0
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    latency_stats: Dict[str, float] = Field(default_factory=dict)
