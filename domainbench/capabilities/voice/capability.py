"""
Voice benchmark capability.

Thin wrapper that registers the voice capability with DomainBench's
capability system. The actual logic lives in engine.py, judge.py,
scorer.py, and generator.py.
"""

from typing import List, Dict, Any

from domainbench.capabilities.base import BaseCapability


class VoiceCapability(BaseCapability):
    """
    Multi-turn voice agent benchmark capability.

    Evaluates LLMs on four dimensions:
      - Tool use correctness
      - Instruction following
      - Knowledge base grounding
      - Turn-taking (audio pipelines only)

    Uses LLM-as-Judge with two-phase realignment scoring.
    """

    name: str = "voice"
    description: str = "Multi-turn voice agent benchmark with tool use and knowledge grounding"
    required_provider_features: List[str] = ["chat_completion"]

    def build_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """Not used directly — VoiceEngine handles sequential turn execution."""
        messages = [{"role": "system", "content": system_prompt}]
        for turn in test_case.get("turns", []):
            if isinstance(turn, dict):
                messages.append({"role": "user", "content": turn.get("input", "")})
            else:
                messages.append({"role": "user", "content": str(turn)})
        return messages

    def validate_test_case(self, test_case: Dict[str, Any]) -> bool:
        """Validate that a test case has the required voice scenario fields."""
        if "turns" not in test_case:
            return False
        if not isinstance(test_case["turns"], list) or len(test_case["turns"]) == 0:
            return False
        if "system_prompt" not in test_case:
            return False
        if "knowledge_base" not in test_case:
            return False
        return True

    def get_required_fields(self) -> List[str]:
        return ["id", "system_prompt", "knowledge_base", "turns"]

    def get_metrics(self) -> List[str]:
        return [
            "pass_rate",
            "tool_use_correct",
            "instruction_following",
            "kb_grounding",
            "turn_taking",
            "ttfb_ms",
            "latency_ms",
        ]
