"""
Voice agent benchmark capability.

Multi-turn evaluation with tool use, instruction following, knowledge
grounding, and turn-taking scoring via LLM-as-Judge with two-phase
realignment.

Supports three pipeline types:
  - text: LLM only (no audio).
  - cascaded: STT -> LLM -> TTS.
  - speech_to_speech: End-to-end audio models.
"""

from domainbench.capabilities.voice.capability import VoiceCapability
from domainbench.capabilities.voice.engine import VoiceEngine
from domainbench.capabilities.voice.judge import VoiceJudge
from domainbench.capabilities.voice.scorer import score_run, compute_latency_stats, aggregate_runs
from domainbench.capabilities.voice.generator import (
    generate_with_ai,
    generate_builtin_hotel,
    list_builtin_domains,
)
from domainbench.capabilities.voice.config import (
    VoiceScenario,
    VoiceTurn,
    VoiceTurnResult,
    VoiceJudgment,
    VoiceRunResult,
    PipelineConfig,
    STTConfig,
    LLMConfig,
    TTSConfig,
    S2SModelConfig,
    InputTTSConfig,
)
from domainbench.capabilities.voice.stt import (
    BaseSTT,
    WhisperSTT,
    DeepgramSTT,
    GoogleSTT,
    create_stt,
)
from domainbench.capabilities.voice.tts import (
    BaseTTS,
    OpenAITTS,
    ElevenLabsTTS,
    GoogleTTS,
    CartesiaTTS,
    create_tts,
)
from domainbench.capabilities.voice.s2s import (
    BaseS2S,
    OpenAIS2S,
    OpenAIRealtimeS2S,
    GeminiS2S,
    create_s2s,
)

__all__ = [
    "VoiceCapability",
    "VoiceEngine",
    "VoiceJudge",
    "score_run",
    "compute_latency_stats",
    "aggregate_runs",
    "generate_with_ai",
    "generate_builtin_hotel",
    "list_builtin_domains",
    "VoiceScenario",
    "VoiceTurn",
    "VoiceTurnResult",
    "VoiceJudgment",
    "VoiceRunResult",
    "PipelineConfig",
    "STTConfig",
    "LLMConfig",
    "TTSConfig",
    "S2SModelConfig",
    "InputTTSConfig",
    "BaseSTT",
    "WhisperSTT",
    "DeepgramSTT",
    "GoogleSTT",
    "create_stt",
    "BaseTTS",
    "OpenAITTS",
    "ElevenLabsTTS",
    "GoogleTTS",
    "CartesiaTTS",
    "create_tts",
    "BaseS2S",
    "OpenAIS2S",
    "OpenAIRealtimeS2S",
    "GeminiS2S",
    "create_s2s",
]
