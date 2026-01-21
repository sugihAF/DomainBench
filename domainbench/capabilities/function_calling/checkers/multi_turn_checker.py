"""
Multi-turn function call validation with state tracking.

Adapted from Berkeley Function Call Leaderboard (BFCL).
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

from domainbench.capabilities.function_calling.checkers.utils import (
    parse_function_call,
    parse_tool_calls_from_response,
    values_match,
)


class MultiTurnExecutor:
    """
    Executes function calls and tracks state across turns.

    Used for validating multi-turn function calling where each turn
    may modify some state that affects subsequent turns.
    """

    def __init__(
        self,
        functions: Optional[Dict[str, Callable]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize executor.

        Args:
            functions: Dict mapping function names to callable implementations
            initial_state: Initial state dictionary
        """
        self.functions = functions or {}
        self.state = copy.deepcopy(initial_state) if initial_state else {}
        self.call_history: List[Dict[str, Any]] = []

    def execute_call(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Execute a function call and update state.

        Args:
            name: Function name
            arguments: Function arguments

        Returns:
            Function result

        Raises:
            ValueError: If function is not registered
        """
        if name not in self.functions:
            raise ValueError(f"Unknown function: {name}")

        # Execute function with state
        result = self.functions[name](self.state, **arguments)

        self.call_history.append({
            "function": name,
            "arguments": arguments,
            "result": result,
        })

        return result

    def get_state(self) -> Dict[str, Any]:
        """Get current state (copy)."""
        return copy.deepcopy(self.state)

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset executor to initial state."""
        self.state = copy.deepcopy(initial_state) if initial_state else {}
        self.call_history = []


def validate_multi_turn(
    model_responses: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    functions: Optional[Dict[str, Callable]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], float]:
    """
    Validate multi-turn function calling sequence.

    Each turn specifies expected function calls and/or expected state
    after execution.

    Args:
        model_responses: List of model responses (one per turn)
        turns: List of turn specifications with expected outcomes
        functions: Optional executable function implementations for state tracking
        initial_state: Initial state for execution

    Returns:
        Tuple of (all_correct, errors, accuracy_score)
    """
    errors = []
    correct_turns = 0
    total_turns = len(turns)

    if len(model_responses) != len(turns):
        errors.append(
            f"Expected {len(turns)} turns, got {len(model_responses)} responses"
        )
        # Adjust to validate what we can
        total_turns = min(len(model_responses), len(turns))

    # Initialize executor if functions provided
    executor = None
    if functions:
        executor = MultiTurnExecutor(functions, initial_state)

    for i in range(total_turns):
        response = model_responses[i]
        turn = turns[i]
        turn_errors = []

        # Extract tool calls from response
        tool_calls = parse_tool_calls_from_response(response)

        # Get expected calls for this turn
        expected_calls_raw = turn.get("expected_calls", [])
        expected_calls = []
        for ec in expected_calls_raw:
            if isinstance(ec, str):
                parsed = parse_function_call(ec)
                if parsed:
                    expected_calls.append(parsed)
            elif isinstance(ec, dict):
                expected_calls.append(ec)

        # Validate call count
        if len(tool_calls) != len(expected_calls):
            turn_errors.append(
                f"Turn {i}: Expected {len(expected_calls)} calls, got {len(tool_calls)}"
            )

        # Execute model's calls (if executor available)
        if executor:
            for tc in tool_calls:
                try:
                    executor.execute_call(
                        tc.get("name", ""),
                        tc.get("arguments", {}),
                    )
                except Exception as e:
                    turn_errors.append(f"Turn {i}: Execution error - {e}")

        # Validate calls match expected
        min_calls = min(len(tool_calls), len(expected_calls))
        calls_matched = 0

        for j in range(min_calls):
            tc = tool_calls[j]
            ec = expected_calls[j]

            tc_name = tc.get("name", "")
            ec_name = ec.get("name", "")

            if tc_name != ec_name:
                turn_errors.append(
                    f"Turn {i}, Call {j}: Wrong function '{tc_name}', expected '{ec_name}'"
                )
                continue

            # Check arguments
            tc_args = tc.get("arguments", {})
            ec_args = ec.get("arguments", {})

            args_match = True
            for key, exp_val in ec_args.items():
                act_val = tc_args.get(key)
                if not values_match(act_val, exp_val, strict=False):
                    turn_errors.append(
                        f"Turn {i}, Call {j}: Param '{key}' mismatch: "
                        f"expected {repr(exp_val)}, got {repr(act_val)}"
                    )
                    args_match = False

            if args_match:
                calls_matched += 1

        # Check expected state (if provided)
        expected_state = turn.get("expected_state")
        if expected_state and executor:
            current_state = executor.get_state()
            state_match = validate_state(current_state, expected_state)
            if not state_match:
                turn_errors.append(
                    f"Turn {i}: State mismatch after execution"
                )

        # Determine if turn passed
        if not turn_errors:
            correct_turns += 1
        else:
            errors.extend(turn_errors)

    # Calculate accuracy
    accuracy = correct_turns / len(turns) if turns else 0.0

    return correct_turns == len(turns) and len(errors) == 0, errors, accuracy


def validate_state(
    actual_state: Dict[str, Any],
    expected_state: Dict[str, Any],
) -> bool:
    """
    Validate that actual state matches expected state.

    Args:
        actual_state: Current state after execution
        expected_state: Expected state specification

    Returns:
        True if states match
    """
    for key, exp_val in expected_state.items():
        if key not in actual_state:
            return False

        act_val = actual_state[key]

        # Deep comparison
        if isinstance(exp_val, dict):
            if not isinstance(act_val, dict):
                return False
            if not validate_state(act_val, exp_val):
                return False
        elif isinstance(exp_val, list):
            if not isinstance(act_val, list):
                return False
            if len(act_val) != len(exp_val):
                return False
            for a, e in zip(act_val, exp_val):
                if isinstance(e, dict):
                    if not isinstance(a, dict):
                        return False
                    if not validate_state(a, e):
                        return False
                elif not values_match(a, e, strict=False):
                    return False
        else:
            if not values_match(act_val, exp_val, strict=False):
                return False

    return True
