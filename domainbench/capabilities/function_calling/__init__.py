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

from domainbench.capabilities.function_calling.function_calling import (
    FunctionCallingCapability,
)
from domainbench.capabilities.function_calling.checkers import (
    parse_function_call,
    validate_function_call,
    validate_parallel_calls,
    validate_multiple_calls,
    validate_multi_turn,
    validate_agentic_response,
)
from domainbench.capabilities.function_calling.domain_creator import (
    create_domain_with_ai,
    validate_generated_domain,
    list_function_calling_domains,
    DEFAULT_CREATOR_MODEL,
    SUPPORTED_CATEGORIES,
)

__all__ = [
    "FunctionCallingCapability",
    "parse_function_call",
    "validate_function_call",
    "validate_parallel_calls",
    "validate_multiple_calls",
    "validate_multi_turn",
    "validate_agentic_response",
    "create_domain_with_ai",
    "validate_generated_domain",
    "list_function_calling_domains",
    "DEFAULT_CREATOR_MODEL",
    "SUPPORTED_CATEGORIES",
]
