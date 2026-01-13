"""
Function Calling Capability - Tool use benchmark for LLMs

Evaluates function/tool calling accuracy using AST-based validation
adapted from Berkeley Function Call Leaderboard (BFCL).

Supports categories:
- simple: Single function call
- parallel: Multiple function calls (order-independent)
- multiple: Same function called multiple times (order-dependent)
- multi_turn: Sequential conversation with state tracking
- agentic: Complex multi-step tasks with text response validation
"""

from typing import Any, Callable, Dict, List, Optional, Union

from domainbench.capabilities.base import BaseCapability
from domainbench.capabilities.function_calling.checkers import (
    validate_function_call,
    validate_parallel_calls,
    validate_multiple_calls,
    validate_multi_turn,
    validate_agentic_response,
    parse_function_call,
)


class FunctionCallingCapability(BaseCapability):
    """
    Function calling benchmark capability.

    Evaluates LLM tool use accuracy across different categories:
    - simple: Single function call validation
    - parallel: Multiple independent function calls
    - multiple: Repeated calls to same function
    - multi_turn: Sequential calls with state tracking
    - agentic: Complex tasks with text response validation

    Test cases should have:
    - 'query': User query that should trigger function call(s)
    - 'functions': List of available function definitions
    - 'ground_truth': Expected function call(s) as Python string(s)
    - 'category': (optional) Evaluation category (default: simple)

    For multi_turn:
    - 'turns': List of turn specifications
    - 'initial_state': (optional) Initial state for execution

    For agentic:
    - 'expected_response': Expected text response
    - 'match_mode': (optional) How to match (exact, contains, regex)
    """

    name = "function_calling"
    description = "Function calling benchmark - evaluates LLM tool use accuracy"
    required_provider_features = ["function_calling"]

    # Supported evaluation categories
    CATEGORIES = ["simple", "parallel", "multiple", "multi_turn", "agentic"]

    def __init__(
        self,
        category: str = "simple",
        execution_backend: Optional[Dict[str, Callable]] = None,
        strict_mode: bool = True,
    ):
        """
        Initialize function calling capability.

        Args:
            category: Default evaluation category
            execution_backend: Dict mapping function names to callables (for multi-turn)
            strict_mode: If True, require exact parameter matching
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {self.CATEGORIES}")

        self.category = category
        self.execution_backend = execution_backend or {}
        self.strict_mode = strict_mode

    def build_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Build messages for function calling request.

        Args:
            test_case: Test case with query and functions
            system_prompt: System prompt from domain config

        Returns:
            List of messages for LLM API
        """
        category = test_case.get("category", self.category)

        if category == "multi_turn":
            return self._build_multi_turn_messages(test_case, system_prompt)
        elif category == "agentic":
            return self._build_agentic_messages(test_case, system_prompt)
        else:
            return self._build_standard_messages(test_case, system_prompt)

    def _build_standard_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        """Build messages for simple/parallel/multiple categories."""
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # User query
        query = test_case.get("query", "")
        messages.append({
            "role": "user",
            "content": query,
        })

        return messages

    def _build_multi_turn_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        """Build initial messages for multi-turn category."""
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # First turn query
        turns = test_case.get("turns", [])
        if turns:
            messages.append({
                "role": "user",
                "content": turns[0].get("query", ""),
            })

        return messages

    def _build_agentic_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        """Build messages for agentic category (text response expected)."""
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # Add context if provided
        context = test_case.get("context", "")
        query = test_case.get("query", "")

        if context:
            content = f"{context}\n\n{query}"
        else:
            content = query

        messages.append({
            "role": "user",
            "content": content,
        })

        return messages

    def build_functions(self, test_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract function definitions from test case.

        Args:
            test_case: Test case with functions

        Returns:
            List of function definitions for provider
        """
        return test_case.get("functions", [])

    def validate_test_case(self, test_case: Dict[str, Any]) -> bool:
        """
        Validate test case has required fields.

        Args:
            test_case: Test case data

        Returns:
            True if valid, False otherwise
        """
        required = ["id", "query"]
        if not all(field in test_case for field in required):
            return False

        category = test_case.get("category", self.category)

        if category in ["simple", "parallel", "multiple"]:
            # Must have functions and ground_truth
            if "functions" not in test_case:
                return False
            if "ground_truth" not in test_case:
                return False

        elif category == "multi_turn":
            # Must have turns and functions
            if "turns" not in test_case or not test_case["turns"]:
                return False
            if "functions" not in test_case:
                return False

        elif category == "agentic":
            # Must have expected response
            if "expected_response" not in test_case:
                return False

        return True

    def get_required_fields(self) -> List[str]:
        """Required fields for function calling test cases."""
        return ["id", "query"]

    def get_metrics(self) -> List[str]:
        """Metrics collected by this capability."""
        return [
            "accuracy",
            "function_name_accuracy",
            "parameter_accuracy",
            "latency",
            "tokens",
        ]

    def evaluate_single(
        self,
        response: Dict[str, Any],
        ground_truth: Any,
        test_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single model's function call response.

        Args:
            response: Provider response with tool_calls or content
            ground_truth: Expected function call(s) or response
            test_case: Full test case for context

        Returns:
            Dict with evaluation metrics
        """
        category = test_case.get("category", self.category)
        functions = test_case.get("functions", [])

        if category == "simple":
            return self._evaluate_simple(response, ground_truth, functions)
        elif category == "parallel":
            return self._evaluate_parallel(response, ground_truth, functions)
        elif category == "multiple":
            return self._evaluate_multiple(response, ground_truth, functions)
        elif category == "multi_turn":
            return self._evaluate_multi_turn(response, ground_truth, test_case)
        elif category == "agentic":
            return self._evaluate_agentic(response, ground_truth, test_case)
        else:
            return {"is_correct": False, "score": 0.0, "errors": [f"Unknown category: {category}"]}

    def _evaluate_simple(
        self,
        response: Dict[str, Any],
        ground_truth: str,
        functions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate simple (single function call) category."""
        is_correct, errors, score = validate_function_call(
            response,
            ground_truth,
            functions,
            strict=self.strict_mode,
        )

        return {
            "is_correct": is_correct,
            "score": score * 100,  # Convert to percentage
            "errors": errors,
            "category": "simple",
        }

    def _evaluate_parallel(
        self,
        response: Dict[str, Any],
        ground_truth: Union[str, List[str]],
        functions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate parallel (multiple independent calls) category."""
        # Ensure ground_truth is a list
        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth

        is_correct, errors, score = validate_parallel_calls(
            response,
            ground_truths,
            functions,
            strict=self.strict_mode,
        )

        return {
            "is_correct": is_correct,
            "score": score * 100,
            "errors": errors,
            "category": "parallel",
        }

    def _evaluate_multiple(
        self,
        response: Dict[str, Any],
        ground_truth: Union[str, List[str]],
        functions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate multiple (repeated same function) category."""
        # Ensure ground_truth is a list
        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth

        is_correct, errors, score = validate_multiple_calls(
            response,
            ground_truths,
            functions,
            strict=self.strict_mode,
        )

        return {
            "is_correct": is_correct,
            "score": score * 100,
            "errors": errors,
            "category": "multiple",
        }

    def _evaluate_multi_turn(
        self,
        response: Union[Dict[str, Any], List[Dict[str, Any]]],
        ground_truth: Any,
        test_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate multi-turn category."""
        turns = test_case.get("turns", [])
        initial_state = test_case.get("initial_state", {})

        # Response should be a list of responses (one per turn)
        if isinstance(response, dict):
            responses = [response]
        else:
            responses = response

        is_correct, errors, score = validate_multi_turn(
            responses,
            turns,
            functions=self.execution_backend or None,
            initial_state=initial_state,
        )

        return {
            "is_correct": is_correct,
            "score": score * 100,
            "errors": errors,
            "category": "multi_turn",
        }

    def _evaluate_agentic(
        self,
        response: Dict[str, Any],
        ground_truth: Any,
        test_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate agentic (text response) category."""
        # Extract text content from response
        response_text = response.get("content", "")

        # Get expected response
        expected = test_case.get("expected_response", ground_truth)
        match_mode = test_case.get("match_mode", "contains")

        is_correct, errors, score = validate_agentic_response(
            response_text,
            expected,
            match_mode,
        )

        return {
            "is_correct": is_correct,
            "score": score * 100,
            "errors": errors,
            "category": "agentic",
        }

    def evaluate_pair(
        self,
        response_a: Dict[str, Any],
        response_b: Dict[str, Any],
        ground_truth: Any,
        test_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare two models' function calling performance.

        Args:
            response_a: Model A's response
            response_b: Model B's response
            ground_truth: Expected result
            test_case: Full test case

        Returns:
            Comparison result with winner
        """
        eval_a = self.evaluate_single(response_a, ground_truth, test_case)
        eval_b = self.evaluate_single(response_b, ground_truth, test_case)

        score_a = eval_a["score"]
        score_b = eval_b["score"]

        # Determine winner
        threshold = 2.0  # 2% threshold for tie
        diff = abs(score_a - score_b)

        if diff < threshold:
            winner = "tie"
        elif score_a > score_b:
            winner = "A"
        else:
            winner = "B"

        return {
            "winner": winner,
            "score_A": score_a,
            "score_B": score_b,
            "difference": round(diff, 2),
            "eval_A": eval_a,
            "eval_B": eval_b,
            "reasons": self._generate_comparison_reasons(eval_a, eval_b, winner),
        }

    def _generate_comparison_reasons(
        self,
        eval_a: Dict[str, Any],
        eval_b: Dict[str, Any],
        winner: str,
    ) -> List[str]:
        """Generate human-readable comparison reasons."""
        reasons = []

        score_a = eval_a["score"]
        score_b = eval_b["score"]
        category = eval_a.get("category", "simple")

        if winner == "tie":
            reasons.append(
                f"Models performed similarly on {category} task "
                f"(A: {score_a:.1f}%, B: {score_b:.1f}%)"
            )
        else:
            reasons.append(
                f"Model {winner} achieved higher accuracy on {category} task "
                f"({max(score_a, score_b):.1f}% vs {min(score_a, score_b):.1f}%)"
            )

        # Add error details
        errors_a = eval_a.get("errors", [])
        errors_b = eval_b.get("errors", [])

        if errors_a and not errors_b:
            reasons.append(f"Model A had {len(errors_a)} error(s)")
        elif errors_b and not errors_a:
            reasons.append(f"Model B had {len(errors_b)} error(s)")
        elif errors_a and errors_b:
            reasons.append(f"Model A: {len(errors_a)} error(s), Model B: {len(errors_b)} error(s)")

        return reasons


def calculate_category_scores(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy scores aggregated by category.

    Args:
        results: List of evaluation results

    Returns:
        Dict with per-category and overall scores
    """
    by_category: Dict[str, Dict[str, int]] = {}

    for result in results:
        cat = result.get("category", "simple")
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}

        by_category[cat]["total"] += 1
        if result.get("is_correct", False):
            by_category[cat]["correct"] += 1

    # Calculate percentages
    scores: Dict[str, Dict[str, float]] = {}
    for cat, stats in by_category.items():
        scores[cat] = {
            "accuracy": (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0.0,
            "correct": float(stats["correct"]),
            "total": float(stats["total"]),
        }

    # Overall
    total_correct = sum(s["correct"] for s in scores.values())
    total_count = sum(s["total"] for s in scores.values())

    scores["overall"] = {
        "accuracy": (total_correct / total_count * 100) if total_count > 0 else 0.0,
        "correct": total_correct,
        "total": total_count,
    }

    return scores
