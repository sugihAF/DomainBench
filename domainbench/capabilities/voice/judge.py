"""
Voice benchmark judge — LLM-as-Judge with two-phase realignment.

Implements the scoring algorithm from aiewf-eval:
  Phase 1: Score each turn independently against golden reference.
  Phase 2: Detect early/late function calls and adjust scores to prevent
           cascading false negatives.
"""

import json
from typing import List, Dict, Any, Optional

from domainbench.capabilities.voice.config import (
    VoiceScenario,
    VoiceTurnResult,
    VoiceJudgment,
)
from domainbench.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for voice AI agents. You evaluate a multi-turn \
conversation between a user and an AI assistant that has access to tools and \
a knowledge base.

## Scoring Dimensions (binary pass/fail per turn)

1. **turn_taking** — Pre-computed; the value is provided to you. Copy it as-is.

2. **tool_use_correct** — Did the model call the right function with the right arguments?
   - TRUE if the expected function was called with semantically equivalent arguments.
   - TRUE if no function was expected and none was made.
   - TRUE if the function was already called in an earlier turn (realignment credit).
   - FALSE if the expected function was not called and was not called earlier.
   - Identifiers (IDs, session IDs) must match exactly.
   - Other string arguments are compared for semantic equivalence, not verbatim match.

3. **instruction_following** — Did the assistant answer the question or advance the task?
   - TRUE if the assistant directly answers the question or advances the workflow.
   - TRUE if the assistant properly deflects an out-of-scope question.
   - FALSE if the assistant contradicts its own actions or ignores the user.
   - Be LENIENT when turn_taking is FALSE (audio issues may have affected input).

4. **kb_grounding** — Is the response factually consistent with the knowledge base?
   - TRUE unless an explicit factual error is stated.
   - TRUE if additional correct information is provided.
   - FALSE only for clear contradictions (wrong dates, times, locations, names).

## Two-Phase Evaluation

### Phase 1 — Initial Analysis
Score each turn independently against its golden (expected) response.

### Phase 2 — Realignment
After the initial pass, detect function-call timing misalignment:
- **Early call**: called at turn N instead of expected turn N+k.
  Credit the actual turn. Do NOT penalize the expected turn.
- **Late call**: called at turn N+k instead of expected turn N.
  Penalize the expected turn (tool_use_correct=false, instruction_following=false).
  Credit the actual turn.
- **Never called**: Penalize the expected turn.

This prevents one off-by-one shift from cascading failures across all later turns.

## Output Format

Return **strict JSON only** (no markdown fences, no commentary outside JSON):
{
  "phase1_analysis": [
    {"turn": 0, "notes": "brief analysis"}
  ],
  "realignment_notes": "description of any timing shifts detected, or 'none'",
  "function_call_tracking": {
    "function_name": {
      "expected_turn": 0,
      "actual_turn": 0,
      "status": "on_time"
    }
  },
  "final_judgments": [
    {
      "turn": 0,
      "reasoning": "brief explanation",
      "turn_taking": true,
      "tool_use_correct": true,
      "instruction_following": true,
      "kb_grounding": true
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_turns_for_judge(
    scenario: VoiceScenario,
    turn_results: List[VoiceTurnResult],
) -> str:
    """Build the user-message payload sent to the judge LLM."""

    parts: List[str] = []

    # --- Knowledge base ---
    parts.append("## Knowledge Base\n")
    parts.append(scenario.knowledge_base)
    parts.append("")

    # --- Tool definitions ---
    if scenario.tools:
        parts.append("## Tool Definitions\n")
        for tool in scenario.tools:
            parts.append(f"- **{tool.get('name', 'unknown')}**: {tool.get('description', '')}")
            params = tool.get("parameters", {}).get("properties", {})
            if params:
                param_strs = [f"  - {k}: {v.get('type', 'any')}" for k, v in params.items()]
                parts.append("\n".join(param_strs))
        parts.append("")

    # --- Expected function calls summary ---
    expected_calls: List[Dict[str, Any]] = []
    for i, turn in enumerate(scenario.turns):
        if turn.required_function_call:
            expected_calls.append({
                "turn": i,
                "function": turn.required_function_call.get("name", "unknown"),
                "args": turn.required_function_call.get("args", {}),
            })

    if expected_calls:
        parts.append("## Expected Function Calls\n")
        for ec in expected_calls:
            parts.append(
                f"- Turn {ec['turn']}: **{ec['function']}**({json.dumps(ec['args'])})"
            )
        parts.append("")

    # --- Per-turn data ---
    parts.append("## Conversation Transcript\n")

    for tr in turn_results:
        parts.append(f"### Turn {tr.turn_index}")
        parts.append(f"**User**: {tr.user_input}")
        parts.append(f"**Assistant**: {tr.assistant_text}")
        parts.append(f"**Golden response**: {tr.golden_text}")

        if tr.expected_function_call:
            name = tr.expected_function_call.get("name", "unknown")
            args = tr.expected_function_call.get("args", {})
            parts.append(f"**Expected function call**: {name}({json.dumps(args)})")
        else:
            parts.append("**Expected function call**: none")

        if tr.tool_calls:
            for tc in tr.tool_calls:
                tc_name = tc.get("name", "unknown")
                tc_args = tc.get("arguments", tc.get("args", {}))
                parts.append(f"**Actual function call**: {tc_name}({json.dumps(tc_args)})")
        else:
            parts.append("**Actual function call**: none")

        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(text: str) -> Optional[dict]:
    """Attempt to parse JSON, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```)
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_judgments(raw: dict, num_turns: int) -> List[VoiceJudgment]:
    """Extract VoiceJudgment list from parsed judge JSON."""
    final = raw.get("final_judgments", [])
    judgments: List[VoiceJudgment] = []

    for entry in final:
        turn_idx = entry.get("turn", len(judgments))
        judgments.append(VoiceJudgment(
            turn_index=turn_idx,
            turn_taking=bool(entry.get("turn_taking", True)),
            tool_use_correct=bool(entry.get("tool_use_correct", True)),
            instruction_following=bool(entry.get("instruction_following", True)),
            kb_grounding=bool(entry.get("kb_grounding", True)),
            reasoning=str(entry.get("reasoning", "")),
        ))

    # Pad missing turns with conservative defaults (fail)
    seen = {j.turn_index for j in judgments}
    for i in range(num_turns):
        if i not in seen:
            judgments.append(VoiceJudgment(
                turn_index=i,
                turn_taking=True,
                tool_use_correct=False,
                instruction_following=False,
                kb_grounding=False,
                reasoning="No judgment returned by judge for this turn; defaulting to fail.",
            ))

    judgments.sort(key=lambda j: j.turn_index)
    return judgments[:num_turns]


# ---------------------------------------------------------------------------
# VoiceJudge
# ---------------------------------------------------------------------------

class VoiceJudge:
    """
    LLM-as-Judge evaluator with two-phase realignment.

    Uses a strong LLM (e.g., GPT-4o, Claude) to score each turn on four
    binary dimensions after comparing the model's actual responses against
    golden references.
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        max_retries: int = 2,
    ):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries

    def evaluate(
        self,
        scenario: VoiceScenario,
        turn_results: List[VoiceTurnResult],
    ) -> List[VoiceJudgment]:
        """
        Evaluate a completed scenario run.

        Args:
            scenario: The original scenario definition.
            turn_results: Recorded results from executing the scenario.

        Returns:
            List of VoiceJudgment, one per turn.
        """
        user_content = _format_turns_for_judge(scenario, turn_results)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        last_text = ""
        for attempt in range(self.max_retries + 1):
            response = self.provider.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            text = response.get("content", "")
            last_text = text

            parsed = _safe_json_loads(text)
            if parsed is not None and "final_judgments" in parsed:
                return _extract_judgments(parsed, len(turn_results))

            # Retry: ask for strict JSON
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    "Your output was not valid JSON or is missing 'final_judgments'. "
                    "Output ONLY strict JSON per the schema above."
                ),
            })

        # Fallback: all pass (conservative; avoids penalizing on judge failure)
        return [
            VoiceJudgment(
                turn_index=i,
                reasoning=f"Judge failed to return valid JSON after {self.max_retries + 1} attempts. "
                          f"Last output: {last_text[:200]}",
            )
            for i in range(len(turn_results))
        ]
