"""
Comprehensive tests for aiewf-eval style metrics:
  - Silence padding detection
  - Voice-to-Voice (V2V) latency
  - Tool vs Non-Tool latency separation
"""

import io
import struct
import wave
import math
import pytest
from typing import List, Dict, Any

from domainbench.capabilities.voice.config import (
    VoiceTurnResult,
    VoiceRunResult,
    VoiceScenario,
    VoiceTurn,
    PipelineConfig,
)
from domainbench.capabilities.voice.scorer import (
    compute_latency_stats,
    aggregate_runs,
    _compute_group_stats,
)
from domainbench.capabilities.voice.engine import VoiceEngine


# ============================================================================
# WAV generation helpers
# ============================================================================

def make_wav(
    samples: List[int],
    sample_rate: int = 16000,
    sample_width: int = 2,
    n_channels: int = 1,
) -> bytes:
    """Generate a WAV file from a list of integer samples."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        if sample_width == 1:
            fmt = f"<{len(samples)}b"
        elif sample_width == 2:
            fmt = f"<{len(samples)}h"
        elif sample_width == 4:
            fmt = f"<{len(samples)}i"
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        raw = struct.pack(fmt, *samples)
        wf.writeframes(raw)
    return buf.getvalue()


def make_silent_then_loud_wav(
    silence_ms: float,
    loud_ms: float = 100.0,
    sample_rate: int = 16000,
    sample_width: int = 2,
    n_channels: int = 1,
    amplitude: float = 0.5,
) -> bytes:
    """Create a WAV with `silence_ms` of silence followed by `loud_ms` of loud signal."""
    silence_samples = int(sample_rate * silence_ms / 1000.0) * n_channels
    loud_samples = int(sample_rate * loud_ms / 1000.0) * n_channels

    if sample_width == 1:
        max_val = 127
    elif sample_width == 2:
        max_val = 32767
    elif sample_width == 4:
        max_val = 2147483647
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    loud_val = int(max_val * amplitude)
    samples = [0] * silence_samples + [loud_val] * loud_samples
    return make_wav(samples, sample_rate, sample_width, n_channels)


# ============================================================================
# Tests: Silence padding detection
# ============================================================================

class TestSilencePadDetection:
    """Test VoiceEngine._detect_silence_pad() static method."""

    def test_none_audio(self):
        """Returns None when audio_data is None."""
        result = VoiceEngine._detect_silence_pad(None)
        assert result is None

    def test_empty_audio(self):
        """Returns None when audio_data is empty bytes."""
        result = VoiceEngine._detect_silence_pad(b"")
        assert result is None

    def test_non_wav_format(self):
        """Returns None for non-WAV formats (mp3, ogg, etc.)."""
        audio = make_silent_then_loud_wav(100)
        assert VoiceEngine._detect_silence_pad(audio, "mp3") is None
        assert VoiceEngine._detect_silence_pad(audio, "ogg") is None
        assert VoiceEngine._detect_silence_pad(audio, "flac") is None

    def test_wav_format_case_insensitive(self):
        """Accepts WAV format in any case."""
        audio = make_silent_then_loud_wav(50)
        result_lower = VoiceEngine._detect_silence_pad(audio, "wav")
        result_upper = VoiceEngine._detect_silence_pad(audio, "WAV")
        result_wave = VoiceEngine._detect_silence_pad(audio, "wave")
        assert result_lower is not None
        assert result_upper is not None
        assert result_wave is not None

    def test_no_silence_instant_loud(self):
        """Audio that starts loud immediately should have ~0ms silence pad."""
        # Create WAV with zero silence, only loud
        audio = make_silent_then_loud_wav(0, loud_ms=200)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert result == 0.0

    def test_50ms_silence(self):
        """Audio with 50ms of silence before loud signal."""
        audio = make_silent_then_loud_wav(50)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        # With 10ms window, should detect ~50ms silence (within 10ms tolerance)
        assert 40.0 <= result <= 60.0

    def test_100ms_silence(self):
        """Audio with 100ms of silence before loud signal."""
        audio = make_silent_then_loud_wav(100)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 90.0 <= result <= 110.0

    def test_500ms_silence(self):
        """Audio with 500ms of silence before loud signal."""
        audio = make_silent_then_loud_wav(500)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 490.0 <= result <= 510.0

    def test_all_silence(self):
        """Audio that is completely silent returns total duration."""
        sample_rate = 16000
        duration_ms = 200
        samples = [0] * (sample_rate * duration_ms // 1000)
        audio = make_wav(samples, sample_rate)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 190.0 <= result <= 210.0

    def test_all_loud(self):
        """Audio that is entirely loud returns 0ms."""
        sample_rate = 16000
        samples = [10000] * (sample_rate * 100 // 1000)
        audio = make_wav(samples, sample_rate)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert result == 0.0

    def test_8bit_wav(self):
        """Works with 8-bit WAV files."""
        # 8-bit: max_val = 127, need amplitude above threshold (0.02)
        # 0.02 * 127 = 2.54, so need samples > ~3
        sample_rate = 16000
        silence = [0] * (sample_rate * 50 // 1000)
        loud = [100] * (sample_rate * 100 // 1000)
        audio = make_wav(silence + loud, sample_rate, sample_width=1)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 40.0 <= result <= 60.0

    def test_32bit_wav(self):
        """Works with 32-bit WAV files."""
        audio = make_silent_then_loud_wav(100, sample_width=4)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 90.0 <= result <= 110.0

    def test_stereo_wav(self):
        """Works with stereo (2-channel) WAV files."""
        audio = make_silent_then_loud_wav(100, n_channels=2)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 90.0 <= result <= 110.0

    def test_44100_sample_rate(self):
        """Works with 44100 Hz sample rate."""
        audio = make_silent_then_loud_wav(100, sample_rate=44100)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 90.0 <= result <= 110.0

    def test_8000_sample_rate(self):
        """Works with 8000 Hz sample rate."""
        audio = make_silent_then_loud_wav(100, sample_rate=8000)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 90.0 <= result <= 110.0

    def test_custom_threshold(self):
        """Custom threshold changes detection sensitivity."""
        # Create audio with very quiet (but non-zero) initial signal
        sample_rate = 16000
        quiet = [100] * (sample_rate * 50 // 1000)  # very quiet
        loud = [20000] * (sample_rate * 100 // 1000)
        audio = make_wav(quiet + loud, sample_rate)

        # Low threshold → detects quiet as non-silence
        result_low = VoiceEngine._detect_silence_pad(audio, threshold=0.001)
        assert result_low is not None
        assert result_low < 10.0  # Should find sound almost immediately

        # High threshold → treats quiet as silence
        result_high = VoiceEngine._detect_silence_pad(audio, threshold=0.5)
        assert result_high is not None
        assert result_high >= 40.0  # Should skip the quiet part

    def test_custom_window_ms(self):
        """Custom window size works correctly."""
        audio = make_silent_then_loud_wav(100)
        result = VoiceEngine._detect_silence_pad(audio, window_ms=20.0)
        assert result is not None
        assert 80.0 <= result <= 120.0

    def test_corrupt_wav_returns_none(self):
        """Corrupt/invalid WAV data returns None."""
        result = VoiceEngine._detect_silence_pad(b"not a wav file")
        assert result is None

    def test_very_short_audio(self):
        """Very short audio (< 1 window) works."""
        # 1ms of audio at 16kHz = 16 samples
        samples = [10000] * 16
        audio = make_wav(samples, 16000)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert result == 0.0


# ============================================================================
# Tests: V2V and silence_pad fields on VoiceTurnResult
# ============================================================================

class TestVoiceTurnResultFields:
    """Test that new fields exist and work on VoiceTurnResult."""

    def test_default_none(self):
        """New fields default to None."""
        tr = VoiceTurnResult(
            turn_index=0,
            user_input="hi",
            assistant_text="hello",
            golden_text="hello",
        )
        assert tr.silence_pad_ms is None
        assert tr.v2v_ms is None

    def test_set_values(self):
        """New fields can be set."""
        tr = VoiceTurnResult(
            turn_index=0,
            user_input="hi",
            assistant_text="hello",
            golden_text="hello",
            silence_pad_ms=45.2,
            v2v_ms=1234.5,
        )
        assert tr.silence_pad_ms == 45.2
        assert tr.v2v_ms == 1234.5

    def test_serialization(self):
        """New fields appear in model_dump() / JSON serialization."""
        tr = VoiceTurnResult(
            turn_index=0,
            user_input="hi",
            assistant_text="hello",
            golden_text="hello",
            silence_pad_ms=50.0,
            v2v_ms=500.0,
        )
        data = tr.model_dump()
        assert data["silence_pad_ms"] == 50.0
        assert data["v2v_ms"] == 500.0

    def test_none_serialization(self):
        """None values serialize correctly."""
        tr = VoiceTurnResult(
            turn_index=0,
            user_input="hi",
            assistant_text="hello",
            golden_text="hello",
        )
        data = tr.model_dump()
        assert data["silence_pad_ms"] is None
        assert data["v2v_ms"] is None


# ============================================================================
# Tests: compute_latency_stats with new metrics
# ============================================================================

class TestComputeLatencyStats:
    """Test the updated compute_latency_stats function."""

    def _make_turn(
        self,
        ttfb=None,
        latency=None,
        v2v=None,
        silence_pad=None,
        tool_calls=None,
    ) -> VoiceTurnResult:
        return VoiceTurnResult(
            turn_index=0,
            user_input="hi",
            assistant_text="hello",
            golden_text="hello",
            ttfb_ms=ttfb,
            latency_ms=latency,
            v2v_ms=v2v,
            silence_pad_ms=silence_pad,
            tool_calls=tool_calls or [],
        )

    def test_empty_turns(self):
        """Empty turn list returns empty stats."""
        stats = compute_latency_stats([])
        assert stats == {}

    def test_basic_ttfb_latency(self):
        """Original TTFB and latency stats still computed."""
        turns = [
            self._make_turn(ttfb=100, latency=200),
            self._make_turn(ttfb=150, latency=300),
        ]
        stats = compute_latency_stats(turns)
        assert "ttfb_median_ms" in stats
        assert "ttfb_mean_ms" in stats
        assert "latency_median_ms" in stats
        assert "latency_mean_ms" in stats

    def test_v2v_stats(self):
        """V2V stats are computed when v2v_ms is present."""
        turns = [
            self._make_turn(v2v=1000),
            self._make_turn(v2v=1200),
            self._make_turn(v2v=1100),
        ]
        stats = compute_latency_stats(turns)
        assert "v2v_median_ms" in stats
        assert "v2v_mean_ms" in stats
        assert "v2v_max_ms" in stats
        assert stats["v2v_median_ms"] == 1100.0
        assert abs(stats["v2v_mean_ms"] - 1100.0) < 1
        assert stats["v2v_max_ms"] == 1200.0

    def test_silence_pad_stats(self):
        """Silence pad stats are computed."""
        turns = [
            self._make_turn(silence_pad=40.0),
            self._make_turn(silence_pad=60.0),
            self._make_turn(silence_pad=50.0),
        ]
        stats = compute_latency_stats(turns)
        assert "silence_pad_median_ms" in stats
        assert "silence_pad_mean_ms" in stats
        assert stats["silence_pad_median_ms"] == 50.0
        assert stats["silence_pad_mean_ms"] == 50.0

    def test_tool_vs_non_tool_separation(self):
        """Tool and non-tool V2V are computed separately."""
        tool_call = [{"name": "search", "arguments": {}}]
        turns = [
            self._make_turn(v2v=1000, tool_calls=tool_call),  # tool turn
            self._make_turn(v2v=800),                          # non-tool turn
            self._make_turn(v2v=1200, tool_calls=tool_call),  # tool turn
            self._make_turn(v2v=900),                          # non-tool turn
        ]
        stats = compute_latency_stats(turns)

        # Tool V2V
        assert "tool_v2v_mean_ms" in stats
        assert stats["tool_v2v_mean_ms"] == 1100.0

        # Non-tool V2V
        assert "non_tool_v2v_mean_ms" in stats
        assert stats["non_tool_v2v_mean_ms"] == 850.0
        assert "non_tool_v2v_median_ms" in stats
        assert stats["non_tool_v2v_median_ms"] == 850.0
        assert "non_tool_v2v_max_ms" in stats
        assert stats["non_tool_v2v_max_ms"] == 900.0

    def test_tool_latency_separation(self):
        """Tool and non-tool latency are split."""
        tool_call = [{"name": "book", "arguments": {}}]
        turns = [
            self._make_turn(latency=500, tool_calls=tool_call),
            self._make_turn(latency=300),
            self._make_turn(latency=400),
        ]
        stats = compute_latency_stats(turns)
        assert "tool_latency_mean_ms" in stats
        assert stats["tool_latency_mean_ms"] == 500.0
        assert "non_tool_latency_mean_ms" in stats
        assert stats["non_tool_latency_mean_ms"] == 350.0

    def test_only_tool_turns(self):
        """Works when all turns have tool calls."""
        tool_call = [{"name": "fn", "arguments": {}}]
        turns = [
            self._make_turn(v2v=1000, tool_calls=tool_call),
            self._make_turn(v2v=1200, tool_calls=tool_call),
        ]
        stats = compute_latency_stats(turns)
        assert "tool_v2v_mean_ms" in stats
        assert "non_tool_v2v_mean_ms" not in stats

    def test_only_non_tool_turns(self):
        """Works when no turns have tool calls."""
        turns = [
            self._make_turn(v2v=800),
            self._make_turn(v2v=900),
        ]
        stats = compute_latency_stats(turns)
        assert "non_tool_v2v_mean_ms" in stats
        assert "tool_v2v_mean_ms" not in stats

    def test_none_v2v_excluded(self):
        """Turns with v2v_ms=None are excluded from V2V stats."""
        turns = [
            self._make_turn(v2v=1000),
            self._make_turn(v2v=None),
            self._make_turn(v2v=1200),
        ]
        stats = compute_latency_stats(turns)
        assert stats["v2v_mean_ms"] == 1100.0

    def test_none_silence_pad_excluded(self):
        """Turns with silence_pad_ms=None are excluded from silence stats."""
        turns = [
            self._make_turn(silence_pad=40.0),
            self._make_turn(silence_pad=None),
            self._make_turn(silence_pad=60.0),
        ]
        stats = compute_latency_stats(turns)
        assert stats["silence_pad_mean_ms"] == 50.0

    def test_mixed_metrics(self):
        """All metric types work together."""
        tool_call = [{"name": "fn", "arguments": {}}]
        turns = [
            self._make_turn(ttfb=100, latency=200, v2v=250, silence_pad=50),
            self._make_turn(ttfb=120, latency=250, v2v=300, silence_pad=50, tool_calls=tool_call),
            self._make_turn(ttfb=110, latency=220, v2v=270, silence_pad=50),
        ]
        stats = compute_latency_stats(turns)
        # All stat types present
        assert "ttfb_median_ms" in stats
        assert "latency_median_ms" in stats
        assert "v2v_median_ms" in stats
        assert "silence_pad_mean_ms" in stats
        assert "tool_v2v_mean_ms" in stats
        assert "non_tool_v2v_mean_ms" in stats


# ============================================================================
# Tests: _compute_group_stats helper
# ============================================================================

class TestComputeGroupStats:
    """Test the _compute_group_stats helper function."""

    def test_empty_list(self):
        """Empty list returns empty dict."""
        assert _compute_group_stats([], "test") == {}

    def test_single_value(self):
        """Single value: median = mean = p95 = max."""
        result = _compute_group_stats([100.0], "test")
        assert result["test_median_ms"] == 100.0
        assert result["test_mean_ms"] == 100.0
        assert result["test_p95_ms"] == 100.0
        assert result["test_max_ms"] == 100.0

    def test_multiple_values(self):
        """Multiple values compute correct statistics."""
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = _compute_group_stats(values, "lat")
        assert result["lat_median_ms"] == 300.0
        assert result["lat_mean_ms"] == 300.0
        assert result["lat_max_ms"] == 500.0


# ============================================================================
# Tests: aggregate_runs with new metrics
# ============================================================================

class TestAggregateRuns:
    """Test aggregate_runs() with new V2V and silence pad metrics."""

    def _make_run_result(
        self, scenario_id="test", model="m", v2v_median=None,
        silence_pad_mean=None, tool_v2v_mean=None, non_tool_v2v_median=None,
    ) -> VoiceRunResult:
        latency_stats = {"ttfb_median_ms": 100.0, "latency_median_ms": 200.0}
        if v2v_median is not None:
            latency_stats["v2v_median_ms"] = v2v_median
        if silence_pad_mean is not None:
            latency_stats["silence_pad_mean_ms"] = silence_pad_mean
        if tool_v2v_mean is not None:
            latency_stats["tool_v2v_mean_ms"] = tool_v2v_mean
        if non_tool_v2v_median is not None:
            latency_stats["non_tool_v2v_median_ms"] = non_tool_v2v_median

        return VoiceRunResult(
            scenario_id=scenario_id,
            model_name=model,
            pipeline_type="cascaded",
            run_index=0,
            pass_rate=80.0,
            dimension_scores={
                "tool_use_correct": 90.0,
                "instruction_following": 80.0,
                "kb_grounding": 70.0,
                "turn_taking": 85.0,
            },
            latency_stats=latency_stats,
        )

    def test_empty_runs(self):
        """Empty run list returns empty dict."""
        assert aggregate_runs([]) == {}

    def test_single_run_with_v2v(self):
        """Single run with V2V stats propagates to aggregation."""
        run = self._make_run_result(v2v_median=1000.0, silence_pad_mean=50.0)
        result = aggregate_runs([run])
        assert result["latency"]["v2v_median_ms"] == 1000.0
        assert result["latency"]["silence_pad_mean_ms"] == 50.0

    def test_multiple_runs_aggregate_v2v(self):
        """Multiple runs correctly aggregate V2V median."""
        runs = [
            self._make_run_result(v2v_median=900.0, silence_pad_mean=40.0),
            self._make_run_result(v2v_median=1100.0, silence_pad_mean=60.0),
        ]
        result = aggregate_runs(runs)
        # Median of [900, 1100] = 1000
        assert result["latency"]["v2v_median_ms"] == 1000.0
        # Median of [40, 60] = 50
        assert result["latency"]["silence_pad_mean_ms"] == 50.0

    def test_tool_non_tool_aggregation(self):
        """Tool and non-tool V2V stats aggregate correctly."""
        runs = [
            self._make_run_result(tool_v2v_mean=1200.0, non_tool_v2v_median=800.0),
            self._make_run_result(tool_v2v_mean=1400.0, non_tool_v2v_median=900.0),
        ]
        result = aggregate_runs(runs)
        assert result["latency"]["tool_v2v_mean_ms"] == 1300.0
        assert result["latency"]["non_tool_v2v_median_ms"] == 850.0

    def test_missing_v2v_stats(self):
        """Runs without V2V stats don't break aggregation."""
        runs = [
            self._make_run_result(),
            self._make_run_result(),
        ]
        result = aggregate_runs(runs)
        assert "v2v_median_ms" not in result["latency"]
        assert "silence_pad_mean_ms" not in result["latency"]


# ============================================================================
# Tests: V2V computation in engine context
# ============================================================================

class TestV2VComputation:
    """Test V2V = pipeline latency + silence padding logic."""

    def test_v2v_cascaded_calculation(self):
        """V2V for cascaded = stt + llm + tts + silence_pad."""
        stt_lat = 200.0
        llm_lat = 500.0
        tts_lat = 300.0
        total = stt_lat + llm_lat + tts_lat  # 1000
        silence_pad = 50.0
        v2v = total + silence_pad  # 1050

        tr = VoiceTurnResult(
            turn_index=0,
            user_input="test",
            assistant_text="response",
            golden_text="response",
            stt_latency_ms=stt_lat,
            llm_latency_ms=llm_lat,
            tts_latency_ms=tts_lat,
            latency_ms=total,
            silence_pad_ms=silence_pad,
            v2v_ms=v2v,
        )
        assert tr.v2v_ms == 1050.0

    def test_v2v_none_when_no_latency(self):
        """V2V should be None when there's no pipeline latency."""
        tr = VoiceTurnResult(
            turn_index=0,
            user_input="test",
            assistant_text="response",
            golden_text="response",
        )
        assert tr.v2v_ms is None

    def test_v2v_with_zero_silence(self):
        """V2V works when silence padding is 0."""
        tr = VoiceTurnResult(
            turn_index=0,
            user_input="test",
            assistant_text="response",
            golden_text="response",
            latency_ms=1000.0,
            silence_pad_ms=0.0,
            v2v_ms=1000.0,
        )
        assert tr.v2v_ms == 1000.0


# ============================================================================
# Tests: Edge cases in silence detection
# ============================================================================

class TestSilenceDetectionEdgeCases:
    """Edge cases and regression tests for silence padding detection."""

    def test_gradually_increasing_audio(self):
        """Audio that gradually increases in volume."""
        sample_rate = 16000
        duration_samples = sample_rate  # 1 second
        # Ramp from 0 to max over 1 second
        samples = [int(32767 * (i / duration_samples)) for i in range(duration_samples)]
        audio = make_wav(samples, sample_rate)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        # Should detect somewhere early where ramp crosses threshold
        assert result < 200.0  # Should detect within first 200ms

    def test_alternating_silence_and_sound(self):
        """Audio with silence, sound, silence, sound pattern."""
        sample_rate = 16000
        ms_to_samples = lambda ms: int(sample_rate * ms / 1000)
        silence = [0] * ms_to_samples(100)
        loud = [20000] * ms_to_samples(50)
        silence2 = [0] * ms_to_samples(100)
        loud2 = [20000] * ms_to_samples(50)
        audio = make_wav(silence + loud + silence2 + loud2, sample_rate)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        # Should detect the FIRST non-silent segment
        assert 90.0 <= result <= 110.0

    def test_very_quiet_noise(self):
        """Audio with very quiet noise (below threshold) then loud."""
        sample_rate = 16000
        # Quiet noise: values of ~10 out of 32767 (0.03% amplitude)
        quiet = [10] * (sample_rate * 100 // 1000)
        loud = [20000] * (sample_rate * 100 // 1000)
        audio = make_wav(quiet + loud, sample_rate)
        # With default threshold of 0.02, RMS of 10/32767 ≈ 0.0003 → below threshold
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert result >= 90.0  # Should skip the quiet part

    def test_negative_samples(self):
        """Audio with negative sample values (real audio oscillates)."""
        sample_rate = 16000
        silence = [0] * (sample_rate * 50 // 1000)
        # Oscillating signal
        loud = []
        for i in range(sample_rate * 100 // 1000):
            val = int(20000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            loud.append(val)
        audio = make_wav(silence + loud, sample_rate)
        result = VoiceEngine._detect_silence_pad(audio)
        assert result is not None
        assert 40.0 <= result <= 60.0


# ============================================================================
# Tests: Integration with scorer and full pipeline
# ============================================================================

class TestIntegrationLatencyPipeline:
    """Integration tests for the full metrics pipeline."""

    def test_full_scorer_pipeline(self):
        """End-to-end: turn results → compute_latency_stats → aggregate_runs."""
        tool_call = [{"name": "search", "arguments": {"q": "test"}}]

        # Create realistic turn results
        turns = [
            VoiceTurnResult(
                turn_index=0, user_input="hi", assistant_text="hello",
                golden_text="hello", ttfb_ms=500, latency_ms=800,
                v2v_ms=850.0, silence_pad_ms=50.0,
            ),
            VoiceTurnResult(
                turn_index=1, user_input="search for X", assistant_text="searching...",
                golden_text="I'll search for X",
                tool_calls=tool_call,
                ttfb_ms=600, latency_ms=1200,
                v2v_ms=1260.0, silence_pad_ms=60.0,
            ),
            VoiceTurnResult(
                turn_index=2, user_input="thanks", assistant_text="you're welcome",
                golden_text="you're welcome", ttfb_ms=400, latency_ms=700,
                v2v_ms=740.0, silence_pad_ms=40.0,
            ),
        ]

        stats = compute_latency_stats(turns)

        # V2V overall
        assert "v2v_median_ms" in stats
        assert stats["v2v_median_ms"] == 850.0

        # Silence pad
        assert "silence_pad_mean_ms" in stats
        assert stats["silence_pad_mean_ms"] == 50.0

        # Tool V2V (only turn 1: 1260.0)
        assert stats["tool_v2v_mean_ms"] == 1260.0

        # Non-tool V2V (turns 0 and 2: 850, 740)
        assert stats["non_tool_v2v_median_ms"] == 795.0
        assert stats["non_tool_v2v_max_ms"] == 850.0

        # Now aggregate
        run = VoiceRunResult(
            scenario_id="test",
            model_name="test_model",
            pipeline_type="cascaded",
            turn_results=turns,
            latency_stats=stats,
            pass_rate=80.0,
            dimension_scores={
                "tool_use_correct": 100.0,
                "instruction_following": 80.0,
                "kb_grounding": 80.0,
                "turn_taking": 100.0,
            },
        )
        agg = aggregate_runs([run])
        assert agg["latency"]["v2v_median_ms"] == 850.0
        assert agg["latency"]["silence_pad_mean_ms"] == 50.0

    def test_scorer_backward_compatibility(self):
        """Old turn results without new fields still work."""
        turns = [
            VoiceTurnResult(
                turn_index=0, user_input="hi", assistant_text="hello",
                golden_text="hello", ttfb_ms=100, latency_ms=200,
            ),
        ]
        stats = compute_latency_stats(turns)
        assert "ttfb_median_ms" in stats
        assert "latency_median_ms" in stats
        # New metrics should NOT be present (all None)
        assert "v2v_median_ms" not in stats
        assert "silence_pad_mean_ms" not in stats

    def test_aggregate_backward_compatibility(self):
        """Old run results without new latency keys still aggregate."""
        run = VoiceRunResult(
            scenario_id="test",
            model_name="model",
            pipeline_type="text",
            latency_stats={"ttfb_median_ms": 100.0, "latency_median_ms": 200.0},
            pass_rate=75.0,
            dimension_scores={
                "tool_use_correct": 80.0,
                "instruction_following": 70.0,
                "kb_grounding": 75.0,
                "turn_taking": 80.0,
            },
        )
        agg = aggregate_runs([run])
        assert "v2v_median_ms" not in agg["latency"]
        assert "pass_rate_mean" in agg

    def test_p95_computation(self):
        """P95 is computed correctly for V2V and silence pad."""
        # 20 turns with varying V2V
        turns = [
            VoiceTurnResult(
                turn_index=i, user_input="hi", assistant_text="hello",
                golden_text="hello",
                v2v_ms=float(1000 + i * 50),
                silence_pad_ms=float(40 + i),
            )
            for i in range(20)
        ]
        stats = compute_latency_stats(turns)
        assert "v2v_p95_ms" in stats
        assert "silence_pad_p95_ms" in stats
        # P95 of 20 values → index 19 (min(19*0.95=18.05→18, 19))
        sorted_v2v = sorted([1000 + i * 50 for i in range(20)])
        expected_p95 = sorted_v2v[min(int(20 * 0.95), 19)]
        assert stats["v2v_p95_ms"] == expected_p95


# ============================================================================
# Tests: Viewer data flow (JSON structure)
# ============================================================================

class TestViewerDataStructure:
    """Test that results JSON has the right structure for the viewer."""

    def test_turn_result_has_v2v_in_json(self):
        """Turn result JSON includes v2v_ms and silence_pad_ms."""
        tr = VoiceTurnResult(
            turn_index=0, user_input="hi", assistant_text="hello",
            golden_text="hello", v2v_ms=1050.0, silence_pad_ms=50.0,
        )
        data = tr.model_dump()
        assert "v2v_ms" in data
        assert "silence_pad_ms" in data
        assert data["v2v_ms"] == 1050.0
        assert data["silence_pad_ms"] == 50.0

    def test_latency_stats_structure(self):
        """Latency stats dict has expected keys for viewer."""
        tool_call = [{"name": "fn", "arguments": {}}]
        turns = [
            VoiceTurnResult(
                turn_index=0, user_input="hi", assistant_text="hello",
                golden_text="hello", ttfb_ms=100, latency_ms=200,
                v2v_ms=250.0, silence_pad_ms=50.0,
            ),
            VoiceTurnResult(
                turn_index=1, user_input="search", assistant_text="ok",
                golden_text="ok", ttfb_ms=150, latency_ms=400,
                v2v_ms=460.0, silence_pad_ms=60.0,
                tool_calls=tool_call,
            ),
        ]
        stats = compute_latency_stats(turns)

        # Keys the viewer can use
        expected_keys = [
            "ttfb_median_ms", "ttfb_mean_ms",
            "latency_median_ms", "latency_mean_ms",
            "v2v_median_ms", "v2v_mean_ms",
            "silence_pad_median_ms", "silence_pad_mean_ms",
            "tool_v2v_mean_ms",
            "non_tool_v2v_mean_ms",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
