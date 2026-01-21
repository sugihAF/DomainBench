"""
Shared utilities for function calling validation.

Adapted from Berkeley Function Call Leaderboard (BFCL).
"""

import ast
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union


def parse_function_call(call_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse a Python function call string into a dictionary.

    Examples:
        "get_weather(city='NYC')" -> {"name": "get_weather", "arguments": {"city": "NYC"}}
        "add_item(name='milk', quantity=2)" -> {"name": "add_item", "arguments": {"name": "milk", "quantity": 2}}

    Args:
        call_str: Python function call string

    Returns:
        Dict with "name" and "arguments", or None if parsing fails
    """
    if not call_str or not isinstance(call_str, str):
        return None

    call_str = call_str.strip()

    try:
        # Parse as Python expression
        tree = ast.parse(call_str, mode='eval')

        if not isinstance(tree.body, ast.Call):
            return None

        call = tree.body

        # Extract function name
        if isinstance(call.func, ast.Name):
            name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # Handle chained calls like obj.method()
            parts = []
            node = call.func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            name = '.'.join(reversed(parts))
        else:
            return None

        # Extract arguments
        arguments = {}

        # Handle positional arguments (convert to kwargs if possible)
        for i, arg in enumerate(call.args):
            try:
                value = ast.literal_eval(arg)
                arguments[f"arg_{i}"] = value
            except (ValueError, TypeError):
                pass

        # Handle keyword arguments
        for kw in call.keywords:
            if kw.arg is None:
                # **kwargs expansion - skip
                continue
            try:
                value = ast.literal_eval(kw.value)
                arguments[kw.arg] = value
            except (ValueError, TypeError):
                # Try to get string representation
                if isinstance(kw.value, ast.Constant):
                    arguments[kw.arg] = kw.value.value
                elif isinstance(kw.value, ast.Name):
                    arguments[kw.arg] = kw.value.id

        return {"name": name, "arguments": arguments}

    except (SyntaxError, ValueError, TypeError):
        pass

    # Fallback: Try regex-based parsing for simple cases
    try:
        match = re.match(r'(\w+)\s*\((.*)\)$', call_str, re.DOTALL)
        if match:
            name = match.group(1)
            args_str = match.group(2).strip()

            if not args_str:
                return {"name": name, "arguments": {}}

            # Parse key=value pairs
            arguments = {}
            # Simple regex for key=value, key='value', key="value", key=123
            pattern = r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\d+\.?\d*)|(\[[^\]]*\])|(\{[^\}]*\})|(\w+))"
            for m in re.finditer(pattern, args_str):
                key = m.group(1)
                # Find which group matched
                if m.group(2) is not None:
                    value = m.group(2)
                elif m.group(3) is not None:
                    value = m.group(3)
                elif m.group(4) is not None:
                    value = float(m.group(4)) if '.' in m.group(4) else int(m.group(4))
                elif m.group(5) is not None:
                    value = json.loads(m.group(5))
                elif m.group(6) is not None:
                    value = json.loads(m.group(6))
                elif m.group(7) is not None:
                    # Could be True, False, None, or variable name
                    val_str = m.group(7)
                    if val_str == 'True':
                        value = True
                    elif val_str == 'False':
                        value = False
                    elif val_str == 'None':
                        value = None
                    else:
                        value = val_str
                else:
                    continue
                arguments[key] = value

            return {"name": name, "arguments": arguments}
    except Exception:
        pass

    return None


def parse_tool_calls_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool calls from a provider response.

    Handles different provider formats:
    - OpenAI: response["tool_calls"] = [{"function": {"name": ..., "arguments": ...}}]
    - Anthropic: response["tool_use"] = [{"name": ..., "input": ...}]
    - Gemini: response["function_calls"] = [{"name": ..., "args": ...}]

    Args:
        response: Provider response dictionary

    Returns:
        Normalized list of tool calls: [{"name": str, "arguments": dict}]
    """
    tool_calls = []

    # OpenAI format
    if "tool_calls" in response and response["tool_calls"]:
        for tc in response["tool_calls"]:
            if "function" in tc:
                func = tc["function"]
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append({"name": name, "arguments": arguments})

    # Anthropic format
    elif "tool_use" in response and response["tool_use"]:
        for tu in response["tool_use"]:
            tool_calls.append({
                "name": tu.get("name", ""),
                "arguments": tu.get("input", {}),
            })

    # Gemini format
    elif "function_calls" in response and response["function_calls"]:
        for fc in response["function_calls"]:
            tool_calls.append({
                "name": fc.get("name", ""),
                "arguments": fc.get("args", {}),
            })

    return tool_calls


def standardize_string(s: str) -> str:
    """
    Standardize string for comparison.

    - Remove extra whitespace
    - Convert to lowercase
    - Remove common punctuation variations

    Args:
        s: Input string

    Returns:
        Standardized string
    """
    if not isinstance(s, str):
        return str(s).lower().strip()

    # Lowercase and strip
    s = s.lower().strip()
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    # Remove certain punctuation for fuzzy matching
    s = re.sub(r'[,.\-_*^()]', '', s)
    # Normalize quotes
    s = s.replace('"', "'")

    return s


def values_match(
    actual: Any,
    expected: Any,
    strict: bool = True,
) -> bool:
    """
    Compare two values with type coercion support.

    Args:
        actual: Value from model response
        expected: Expected value from ground truth
        strict: If True, require exact type match

    Returns:
        True if values match
    """
    # Direct equality
    if actual == expected:
        return True

    # None handling
    if expected is None:
        return actual is None
    if actual is None:
        return False

    # String comparison (case-insensitive, normalized)
    if isinstance(expected, str):
        if isinstance(actual, str):
            return standardize_string(actual) == standardize_string(expected)
        # Allow number-to-string comparison
        if not strict and isinstance(actual, (int, float)):
            return standardize_string(str(actual)) == standardize_string(expected)
        return False

    # Numeric comparison with tolerance
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        # Exact match for integers
        if isinstance(expected, int) and isinstance(actual, int):
            return actual == expected
        # Tolerance for floats
        return abs(float(actual) - float(expected)) < 0.001

    # Boolean comparison
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual == expected
        # Allow string "true"/"false" matching
        if isinstance(actual, str):
            return actual.lower() in ('true', '1') if expected else actual.lower() in ('false', '0')
        return False

    # List comparison
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if len(actual) != len(expected):
            return False
        # Order-dependent comparison
        return all(values_match(a, e, strict) for a, e in zip(actual, expected))

    # Dict comparison
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        # All expected keys must be present and match
        for key, exp_val in expected.items():
            if key not in actual:
                return False
            if not values_match(actual[key], exp_val, strict):
                return False
        return True

    return False


def get_function_schema(
    func_name: str,
    functions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Find function schema by name from list of function definitions.

    Args:
        func_name: Function name to find
        functions: List of function definitions

    Returns:
        Function schema dict or None if not found
    """
    for func in functions:
        if func.get("name") == func_name:
            return func
    return None


def get_required_params(func_schema: Dict[str, Any]) -> List[str]:
    """
    Extract required parameter names from function schema.

    Args:
        func_schema: Function definition with parameters

    Returns:
        List of required parameter names
    """
    if not func_schema:
        return []

    params = func_schema.get("parameters", {})
    return params.get("required", [])


def get_param_type(
    param_name: str,
    func_schema: Dict[str, Any],
) -> Optional[str]:
    """
    Get the expected type for a parameter from function schema.

    Args:
        param_name: Parameter name
        func_schema: Function definition

    Returns:
        Type string (e.g., "string", "integer", "array") or None
    """
    if not func_schema:
        return None

    params = func_schema.get("parameters", {})
    properties = params.get("properties", {})
    param_def = properties.get(param_name, {})

    return param_def.get("type")
