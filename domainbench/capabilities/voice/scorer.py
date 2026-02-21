"""
Voice benchmark scorer — aggregation and pass-rate calculation.

Computes per-run and cross-run statistics following the aiewf-eval
methodology: binary pass/fail per dimension, aggregated as a percentage.
"""

import statistics
from typing import List, Dict, Any

from domainbench.capabilities.voice.config import (
    VoiceJudgment,
    VoiceTurnResult,
    VoiceRunResult,
)


def score_run(
    judgments: List[VoiceJudgment],
    include_turn_taking: bool = False,
) -> Dict[str, Any]:
    """
    Compute pass rate and per-dimension scores for a single run.

    Args:
        judgments: Per-turn binary judgments.
        include_turn_taking: Whether to include turn_taking in the overall
            pass rate (True for audio pipelines, False for text-only).

    Returns:
        Dict with ``pass_rate``, ``dimension_scores``, and ``per_turn``.
    """
    if not judgments:
        return {"pass_rate": 0.0, "dimension_scores": {}, "per_turn": []}

    n = len(judgments)

    # Dimension totals
    tool_correct = sum(1 for j in judgments if j.tool_use_correct)
    instruction = sum(1 for j in judgments if j.instruction_following)
    kb_ground = sum(1 for j in judgments if j.kb_grounding)
    turn_take = sum(1 for j in judgments if j.turn_taking)

    # Pass rate: sum of passing dimensions / (turns * scored_dimensions)
    if include_turn_taking:
        total_passing = tool_correct + instruction + kb_ground + turn_take
        total_possible = n * 4
    else:
        total_passing = tool_correct + instruction + kb_ground
        total_possible = n * 3

    pass_rate = (total_passing / total_possible * 100) if total_possible > 0 else 0.0

    dimension_scores = {
        "tool_use_correct": round(tool_correct / n * 100, 1) if n else 0.0,
        "instruction_following": round(instruction / n * 100, 1) if n else 0.0,
        "kb_grounding": round(kb_ground / n * 100, 1) if n else 0.0,
        "turn_taking": round(turn_take / n * 100, 1) if n else 0.0,
    }

    per_turn = [
        {
            "turn": j.turn_index,
            "tool_use_correct": j.tool_use_correct,
            "instruction_following": j.instruction_following,
            "kb_grounding": j.kb_grounding,
            "turn_taking": j.turn_taking,
            "reasoning": j.reasoning,
        }
        for j in judgments
    ]

    return {
        "pass_rate": round(pass_rate, 1),
        "dimension_scores": dimension_scores,
        "per_turn": per_turn,
    }


def _compute_group_stats(
    values: List[float], prefix: str
) -> Dict[str, float]:
    """Compute median/mean/p95/max for a list of values under a key prefix."""
    if not values:
        return {}
    sorted_v = sorted(values)
    p95_idx = min(int(len(sorted_v) * 0.95), len(sorted_v) - 1)
    return {
        f"{prefix}_median_ms": round(statistics.median(sorted_v), 1),
        f"{prefix}_mean_ms": round(statistics.mean(sorted_v), 1),
        f"{prefix}_p95_ms": round(sorted_v[p95_idx], 1),
        f"{prefix}_max_ms": round(max(sorted_v), 1),
    }


def compute_latency_stats(turn_results: List[VoiceTurnResult]) -> Dict[str, float]:
    """
    Compute latency statistics across all turns in a run.

    Returns median, p95, max, and mean for TTFB, latency, V2V, and silence
    padding — both overall and split by tool / non-tool turns.
    """
    ttfb_values = [
        tr.ttfb_ms for tr in turn_results if tr.ttfb_ms is not None
    ]
    latency_values = [
        tr.latency_ms for tr in turn_results if tr.latency_ms is not None
    ]

    stats: Dict[str, float] = {}

    if ttfb_values:
        stats.update(_compute_group_stats(ttfb_values, "ttfb"))

    if latency_values:
        stats.update(_compute_group_stats(latency_values, "latency"))

    # V2V (voice-to-voice) stats — overall
    v2v_values = [tr.v2v_ms for tr in turn_results if tr.v2v_ms is not None]
    if v2v_values:
        stats.update(_compute_group_stats(v2v_values, "v2v"))

    # Silence padding stats — overall
    sp_values = [tr.silence_pad_ms for tr in turn_results if tr.silence_pad_ms is not None]
    if sp_values:
        stats.update(_compute_group_stats(sp_values, "silence_pad"))

    # --- Tool vs Non-Tool split ---
    tool_turns = [tr for tr in turn_results if tr.tool_calls]
    non_tool_turns = [tr for tr in turn_results if not tr.tool_calls]

    # Tool turn V2V
    tool_v2v = [tr.v2v_ms for tr in tool_turns if tr.v2v_ms is not None]
    if tool_v2v:
        stats.update(_compute_group_stats(tool_v2v, "tool_v2v"))

    # Non-tool turn V2V
    non_tool_v2v = [tr.v2v_ms for tr in non_tool_turns if tr.v2v_ms is not None]
    if non_tool_v2v:
        stats.update(_compute_group_stats(non_tool_v2v, "non_tool_v2v"))

    # Tool turn latency
    tool_lat = [tr.latency_ms for tr in tool_turns if tr.latency_ms is not None]
    if tool_lat:
        stats.update(_compute_group_stats(tool_lat, "tool_latency"))

    # Non-tool turn latency
    non_tool_lat = [tr.latency_ms for tr in non_tool_turns if tr.latency_ms is not None]
    if non_tool_lat:
        stats.update(_compute_group_stats(non_tool_lat, "non_tool_latency"))

    return stats


def aggregate_runs(
    run_results: List[VoiceRunResult],
) -> Dict[str, Any]:
    """
    Aggregate results across multiple runs of the same scenario/model.

    Reports mean pass rate, median pass rate, per-dimension averages,
    and latency statistics.
    """
    if not run_results:
        return {}

    pass_rates = [r.pass_rate for r in run_results]
    model_name = run_results[0].model_name
    scenario_id = run_results[0].scenario_id
    pipeline_type = run_results[0].pipeline_type

    # Aggregate dimension scores
    dimension_keys = ["tool_use_correct", "instruction_following", "kb_grounding", "turn_taking"]
    dimension_avgs: Dict[str, float] = {}
    for key in dimension_keys:
        values = [r.dimension_scores.get(key, 0.0) for r in run_results]
        dimension_avgs[key] = round(statistics.mean(values), 1) if values else 0.0

    # Aggregate latency — collect median values from each run
    latency_keys = [
        "ttfb_median_ms", "latency_median_ms",
        "v2v_median_ms", "silence_pad_mean_ms",
        "tool_v2v_mean_ms", "non_tool_v2v_median_ms", "non_tool_v2v_max_ms",
        "tool_latency_mean_ms", "non_tool_latency_mean_ms",
    ]
    latency_agg: Dict[str, float] = {}
    for key in latency_keys:
        values = [r.latency_stats.get(key, 0.0) for r in run_results if r.latency_stats.get(key, 0.0) > 0]
        if values:
            latency_agg[key] = round(statistics.median(values), 1)

    return {
        "model_name": model_name,
        "scenario_id": scenario_id,
        "pipeline_type": pipeline_type,
        "num_runs": len(run_results),
        "pass_rate_mean": round(statistics.mean(pass_rates), 1),
        "pass_rate_median": round(statistics.median(pass_rates), 1),
        "pass_rate_min": round(min(pass_rates), 1),
        "pass_rate_max": round(max(pass_rates), 1),
        "dimension_scores": dimension_avgs,
        "latency": latency_agg,
        "per_run_pass_rates": [round(p, 1) for p in pass_rates],
    }
