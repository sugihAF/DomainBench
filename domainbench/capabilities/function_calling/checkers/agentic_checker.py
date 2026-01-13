"""
Agentic response validation - Text matching with standardization.

For multi-step agent tasks where the output is text rather than
structured function calls.

Adapted from Berkeley Function Call Leaderboard (BFCL).
"""

import re
from typing import List, Tuple, Union

from domainbench.capabilities.function_calling.checkers.utils import (
    standardize_string,
)


def validate_agentic_response(
    model_response: str,
    expected_response: Union[str, List[str]],
    match_mode: str = "contains",
) -> Tuple[bool, List[str], float]:
    """
    Validate agentic text response.

    Supports multiple match modes for flexible validation:
    - "exact": Response must exactly match expected (after standardization)
    - "contains": Expected text must be found somewhere in response
    - "regex": Expected is treated as regex pattern
    - "any": Response must contain any of the expected responses (for List)

    Args:
        model_response: Model's text output
        expected_response: Expected pattern, text, or list of acceptable responses
        match_mode: How to match responses

    Returns:
        Tuple of (is_correct, errors, score)
    """
    errors = []

    if not model_response:
        return False, ["Empty response from model"], 0.0

    # Handle list of possible responses
    if isinstance(expected_response, list):
        if match_mode == "any" or match_mode == "contains":
            # Check if any expected response is found
            for exp in expected_response:
                is_match, _, _ = _check_single_match(
                    model_response, exp, "contains"
                )
                if is_match:
                    return True, [], 1.0
            errors.append("Response did not match any expected answers")
            return False, errors, 0.0
        else:
            # For exact/regex, check against first expected
            expected_response = expected_response[0]

    return _check_single_match(model_response, expected_response, match_mode)


def _check_single_match(
    model_response: str,
    expected: str,
    match_mode: str,
) -> Tuple[bool, List[str], float]:
    """
    Check a single expected response against model output.

    Args:
        model_response: Model's text output
        expected: Expected text or pattern
        match_mode: Match mode (exact, contains, regex)

    Returns:
        Tuple of (is_correct, errors, score)
    """
    errors = []

    # Standardize both strings
    model_text = standardize_string(model_response)
    expected_text = standardize_string(expected)

    if match_mode == "exact":
        if model_text != expected_text:
            errors.append(
                f"Exact match failed: expected '{expected}', got response starting with '{model_response[:100]}...'"
            )
            # Calculate partial similarity score
            score = _calculate_similarity(model_text, expected_text)
            return False, errors, score
        return True, [], 1.0

    elif match_mode == "contains":
        # Check if expected text is contained in response
        if expected_text in model_text:
            return True, [], 1.0

        # Also try word boundary matching for more precise checking
        pattern = r'\b' + re.escape(expected_text) + r'\b'
        if re.search(pattern, model_text, re.IGNORECASE):
            return True, [], 1.0

        errors.append(f"Expected text '{expected}' not found in response")
        return False, errors, 0.0

    elif match_mode == "regex":
        try:
            if re.search(expected, model_response, re.IGNORECASE | re.DOTALL):
                return True, [], 1.0
            errors.append(f"Regex pattern '{expected}' not matched in response")
            return False, errors, 0.0
        except re.error as e:
            errors.append(f"Invalid regex pattern: {e}")
            return False, errors, 0.0

    else:
        errors.append(f"Unknown match mode: {match_mode}")
        return False, errors, 0.0


def _calculate_similarity(s1: str, s2: str) -> float:
    """
    Calculate simple similarity score between two strings.

    Uses a basic approach based on common word overlap.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0 and 1
    """
    if not s1 or not s2:
        return 0.0

    # Tokenize into words
    words1 = set(s1.split())
    words2 = set(s2.split())

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity
    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def extract_answer_from_response(
    response: str,
    answer_patterns: List[str] = None,
) -> str:
    """
    Extract the answer portion from a model response.

    Models often include explanations before/after the actual answer.
    This function tries to extract just the answer.

    Args:
        response: Full model response
        answer_patterns: Optional regex patterns to identify answer

    Returns:
        Extracted answer string
    """
    if not response:
        return ""

    # Default patterns for common answer formats
    if answer_patterns is None:
        answer_patterns = [
            r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
            r"(?:result:)\s*(.+?)(?:\.|$)",
            r"(?:therefore,?)\s*(.+?)(?:\.|$)",
        ]

    # Try each pattern
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: return first sentence or first line
    lines = response.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        # Remove common prefixes
        prefixes = ["Sure!", "Of course!", "Here's", "The"]
        for prefix in prefixes:
            if first_line.lower().startswith(prefix.lower()):
                first_line = first_line[len(prefix):].strip()
                if first_line.startswith(','):
                    first_line = first_line[1:].strip()
        return first_line

    return response.strip()
