"""
AST-based function call validation for simple, parallel, and multiple categories.

Adapted from Berkeley Function Call Leaderboard (BFCL).
"""

from typing import Any, Dict, List, Optional, Tuple

from domainbench.capabilities.function_calling.checkers.utils import (
    parse_function_call,
    parse_tool_calls_from_response,
    values_match,
    get_function_schema,
    get_required_params,
)


def validate_function_call(
    model_output: Dict[str, Any],
    ground_truth: str,
    functions: List[Dict[str, Any]],
    strict: bool = True,
) -> Tuple[bool, List[str], float]:
    """
    Validate a single function call against ground truth.

    Args:
        model_output: Provider response with tool_calls
        ground_truth: Expected function call as Python string
        functions: Function definitions for parameter validation
        strict: Require exact parameter match

    Returns:
        Tuple of (is_correct, list_of_errors, score)
    """
    errors = []

    # Extract tool calls from response
    tool_calls = parse_tool_calls_from_response(model_output)

    if not tool_calls:
        return False, ["No function call in response"], 0.0

    # Parse ground truth
    expected = parse_function_call(ground_truth)
    if expected is None:
        return False, [f"Failed to parse ground truth: {ground_truth}"], 0.0

    # Get the first tool call (for simple category)
    actual = tool_calls[0]
    actual_name = actual.get("name", "")
    actual_args = actual.get("arguments", {})

    # Validate function name
    if actual_name != expected["name"]:
        errors.append(f"Wrong function: expected '{expected['name']}', got '{actual_name}'")
        return False, errors, 0.0

    # Get function schema for validation
    func_schema = get_function_schema(actual_name, functions)

    # Validate parameters
    param_errors, param_score = validate_parameters(
        actual_args,
        expected["arguments"],
        func_schema,
        strict,
    )
    errors.extend(param_errors)

    # Calculate overall score
    if errors:
        return False, errors, param_score
    return True, [], 1.0


def validate_parameters(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    func_schema: Optional[Dict[str, Any]],
    strict: bool,
) -> Tuple[List[str], float]:
    """
    Validate function parameters.

    Args:
        actual: Parameters from model response
        expected: Expected parameters from ground truth
        func_schema: Function definition for type validation
        strict: If True, require exact parameter match

    Returns:
        Tuple of (list_of_errors, partial_score)
    """
    errors = []
    total_params = len(expected)
    matched_params = 0

    # Check required parameters from schema
    if func_schema:
        required = get_required_params(func_schema)
        for param in required:
            if param not in actual and param in expected:
                errors.append(f"Missing required parameter: {param}")

    # Check expected parameters
    for key, exp_value in expected.items():
        if key not in actual:
            errors.append(f"Missing parameter: {key}")
            continue

        act_value = actual[key]
        if not values_match(act_value, exp_value, strict):
            errors.append(
                f"Parameter '{key}' mismatch: expected {repr(exp_value)}, got {repr(act_value)}"
            )
        else:
            matched_params += 1

    # Check for extra parameters in strict mode
    if strict:
        extra = set(actual.keys()) - set(expected.keys())
        if extra:
            errors.append(f"Extra parameters: {extra}")

    # Calculate partial score
    score = matched_params / total_params if total_params > 0 else 1.0

    return errors, score


def validate_parallel_calls(
    model_output: Dict[str, Any],
    ground_truths: List[str],
    functions: List[Dict[str, Any]],
    strict: bool = True,
) -> Tuple[bool, List[str], float]:
    """
    Validate parallel function calls (order-independent).

    The model should call all expected functions, but order doesn't matter.

    Args:
        model_output: Provider response with tool_calls
        ground_truths: List of expected function calls as Python strings
        functions: Function definitions
        strict: Require exact parameter match

    Returns:
        Tuple of (all_correct, errors, accuracy_score)
    """
    errors = []

    # Extract tool calls
    tool_calls = parse_tool_calls_from_response(model_output)

    if not tool_calls:
        return False, ["No function calls in response"], 0.0

    # Check count
    if len(tool_calls) != len(ground_truths):
        errors.append(
            f"Expected {len(ground_truths)} function calls, got {len(tool_calls)}"
        )

    # Parse all ground truths
    expected_calls = []
    for gt in ground_truths:
        parsed = parse_function_call(gt)
        if parsed is None:
            errors.append(f"Failed to parse ground truth: {gt}")
            continue
        expected_calls.append(parsed)

    # Match calls to ground truths using greedy best-match algorithm
    matched_expected = set()
    matched_actual = set()
    correct = 0

    # For each expected call, find the best matching actual call
    for exp_idx, expected in enumerate(expected_calls):
        best_match_idx = None
        best_score = 0

        for act_idx, actual in enumerate(tool_calls):
            if act_idx in matched_actual:
                continue

            # Check function name match
            if actual.get("name") != expected["name"]:
                continue

            # Calculate parameter match score
            _, param_score = validate_parameters(
                actual.get("arguments", {}),
                expected["arguments"],
                get_function_schema(expected["name"], functions),
                strict,
            )

            if param_score > best_score:
                best_score = param_score
                best_match_idx = act_idx

        if best_match_idx is not None and best_score > 0.5:
            matched_expected.add(exp_idx)
            matched_actual.add(best_match_idx)
            if best_score == 1.0:
                correct += 1
            else:
                errors.append(
                    f"Partial match for {expected['name']}: score {best_score:.2f}"
                )

    # Report unmatched expected calls
    for exp_idx, expected in enumerate(expected_calls):
        if exp_idx not in matched_expected:
            errors.append(f"No match found for: {expected['name']}({expected['arguments']})")

    # Report extra actual calls
    for act_idx, actual in enumerate(tool_calls):
        if act_idx not in matched_actual:
            errors.append(f"Unexpected function call: {actual.get('name')}")

    # Calculate accuracy
    accuracy = correct / len(expected_calls) if expected_calls else 0.0

    return correct == len(expected_calls) and len(errors) == 0, errors, accuracy


def validate_multiple_calls(
    model_output: Dict[str, Any],
    ground_truths: List[str],
    functions: List[Dict[str, Any]],
    strict: bool = True,
) -> Tuple[bool, List[str], float]:
    """
    Validate multiple calls of the same function (order-dependent).

    The model should call the same function multiple times with different
    arguments, in the specified order.

    Args:
        model_output: Provider response with tool_calls
        ground_truths: List of expected function calls as Python strings (ordered)
        functions: Function definitions
        strict: Require exact parameter match

    Returns:
        Tuple of (all_correct, errors, accuracy_score)
    """
    errors = []

    # Extract tool calls
    tool_calls = parse_tool_calls_from_response(model_output)

    if not tool_calls:
        return False, ["No function calls in response"], 0.0

    # Check count
    if len(tool_calls) != len(ground_truths):
        errors.append(
            f"Expected {len(ground_truths)} function calls, got {len(tool_calls)}"
        )
        # Adjust to compare what we can
        min_len = min(len(tool_calls), len(ground_truths))
    else:
        min_len = len(ground_truths)

    # Validate each call in order
    correct = 0
    for i in range(min_len):
        gt = ground_truths[i]
        tc = tool_calls[i]

        expected = parse_function_call(gt)
        if expected is None:
            errors.append(f"Call {i}: Failed to parse ground truth: {gt}")
            continue

        actual_name = tc.get("name", "")
        actual_args = tc.get("arguments", {})

        # Check function name
        if actual_name != expected["name"]:
            errors.append(
                f"Call {i}: Wrong function '{actual_name}', expected '{expected['name']}'"
            )
            continue

        # Check parameters
        param_errors, param_score = validate_parameters(
            actual_args,
            expected["arguments"],
            get_function_schema(expected["name"], functions),
            strict,
        )

        if param_errors:
            for err in param_errors:
                errors.append(f"Call {i}: {err}")
        else:
            correct += 1

    # Calculate accuracy
    total = len(ground_truths)
    accuracy = correct / total if total > 0 else 0.0

    return correct == total and len(errors) == 0, errors, accuracy
