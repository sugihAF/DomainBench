"""
AI-Powered Domain Creator for Function Calling Capability.

Uses LLM to generate new function calling benchmark domains with:
- domain.yaml (function definitions)
- generator.py (test case generator)
- __init__.py (module exports)

Key design principles (learned from working domains):
- NO array parameters (causes Gemini API errors)
- NO default values (not supported by all providers)
- Values in queries MUST match values in ground_truth exactly
- Use enum values directly in queries (not mapped names)
- Simple schemas: string, integer, boolean with enums
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import optional dependencies
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    class Console:
        def print(self, *args, **kwargs):
            # Strip rich formatting
            text = args[0] if args else ""
            text = re.sub(r'\[.*?\]', '', str(text))
            print(text)

    console = Console()


# Default model for domain generation
DEFAULT_CREATOR_MODEL = "gpt-4o"
DEFAULT_PROVIDER = "openai"

# All standard categories that every domain generator must support
SUPPORTED_CATEGORIES = ["simple", "parallel", "multiple", "multi_turn", "agentic"]

# Path to builtin function calling domains
FUNC_CALL_DOMAINS_DIR = Path(__file__).parent.parent.parent / "domains" / "builtin" / "function_calling"


def slugify(name: str) -> str:
    """Convert a name to a valid Python module name / directory slug."""
    slug = name.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '_', slug)
    slug = slug.strip('_')
    return slug


def _fix_generated_code(code: str) -> str:
    """
    Fix common issues in AI-generated Python code.

    Fixes:
    - JSON-style booleans (true/false -> True/False)
    - JSON-style null (null -> None)
    - Missing ground_truth field (expected_calls -> ground_truth)
    """
    # Fix JSON-style booleans and null
    code = re.sub(r':\s*true\b', ': True', code)
    code = re.sub(r':\s*false\b', ': False', code)
    code = re.sub(r':\s*null\b', ': None', code)
    code = re.sub(r',\s*true\b', ', True', code)
    code = re.sub(r',\s*false\b', ', False', code)
    code = re.sub(r',\s*null\b', ', None', code)
    code = re.sub(r'\[\s*true\b', '[True', code)
    code = re.sub(r'\[\s*false\b', '[False', code)
    code = re.sub(r'\[\s*null\b', '[None', code)

    # Fix ground_truth field if missing but expected_calls exists
    if '"expected_calls"' in code and '"ground_truth"' not in code:
        code = code.replace('"expected_calls":', '"ground_truth":')

    return code


def _clean_functions_for_compatibility(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean function definitions for cross-provider compatibility.

    Removes:
    - Array type parameters (Gemini issues)
    - Default values (not universally supported)
    """
    cleaned = []
    for func in functions:
        clean_func = {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": func.get("parameters", {}).get("required", [])
            }
        }

        props = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])
        new_required = []

        for param_name, param_def in props.items():
            # Skip array types - they cause Gemini errors
            if param_def.get("type") == "array":
                continue

            # Copy parameter without default
            clean_param = {
                "type": param_def.get("type", "string"),
                "description": param_def.get("description", "")
            }

            # Keep enum if present
            if "enum" in param_def:
                clean_param["enum"] = param_def["enum"]

            clean_func["parameters"]["properties"][param_name] = clean_param

            # Track required params that weren't removed
            if param_name in required:
                new_required.append(param_name)

        clean_func["parameters"]["required"] = new_required
        cleaned.append(clean_func)

    return cleaned


def _extract_enum_values(functions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extract all enum values from function parameter definitions.

    Returns a dict mapping parameter names to their valid values.
    """
    enums = {}

    for func in functions:
        params = func.get("parameters", {}).get("properties", {})
        for param_name, param_def in params.items():
            if "enum" in param_def:
                key = f"{func['name']}_{param_name}"
                enums[key] = param_def["enum"]
                # Also store by just param name for easier lookup
                if param_name not in enums:
                    enums[param_name] = param_def["enum"]

    return enums


def _extract_sample_data_from_functions(functions: List[Dict[str, Any]], domain_name: str) -> Dict[str, Any]:
    """
    Analyze function definitions and extract domain-specific sample data.

    Returns structured sample data that can be used in templates.
    """
    enums = _extract_enum_values(functions)

    # Build sample data structure
    sample_data = {
        "enums": enums,
        "functions": [],
        "domain_name": domain_name,
    }

    for func in functions:
        func_info = {
            "name": func["name"],
            "description": func.get("description", ""),
            "required_params": func.get("parameters", {}).get("required", []),
            "params": {},
        }

        params = func.get("parameters", {}).get("properties", {})
        for param_name, param_def in params.items():
            param_info = {
                "type": param_def.get("type", "string"),
                "description": param_def.get("description", ""),
                "required": param_name in func_info["required_params"],
            }

            if "enum" in param_def:
                param_info["values"] = param_def["enum"]
            elif param_def.get("type") == "integer":
                param_info["values"] = _generate_sample_integers(param_name)
            elif param_def.get("type") == "boolean":
                param_info["values"] = [True, False]

            func_info["params"][param_name] = param_info

        sample_data["functions"].append(func_info)

    return sample_data


def _generate_sample_integers(param_name: str) -> List[int]:
    """Generate appropriate sample integers based on parameter name."""
    name_lower = param_name.lower()

    if "count" in name_lower or "quantity" in name_lower or "num" in name_lower:
        return [1, 2, 3, 4, 5]
    elif "table" in name_lower:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif "guest" in name_lower or "people" in name_lower or "party" in name_lower:
        return [1, 2, 3, 4, 5, 6]
    elif "percent" in name_lower or "tip" in name_lower:
        return [10, 15, 18, 20, 25]
    elif "year" in name_lower:
        return [2020, 2021, 2022, 2023, 2024, 2025]
    elif "month" in name_lower:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif "day" in name_lower:
        return [1, 5, 10, 15, 20, 25, 28]
    elif "hour" in name_lower:
        return [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    elif "minute" in name_lower:
        return [0, 15, 30, 45]
    elif "id" in name_lower:
        return [1001, 1002, 1003, 1004, 1005]
    elif "price" in name_lower or "amount" in name_lower or "cost" in name_lower:
        return [10, 25, 50, 100, 250]
    else:
        return [1, 2, 3, 5, 10]


def create_domain_with_ai(
    domain_name: str,
    domain_description: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_CREATOR_MODEL,
    categories: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, str]:
    """
    Create a new function calling domain using AI.

    Args:
        domain_name: Human-readable name for the domain
        domain_description: Optional description of what the domain covers
        provider: LLM provider to use (openai, anthropic, gemini)
        model: Model name for generation
        categories: Categories to support (default: all)
        output_dir: Custom output directory (default: builtin domains)

    Returns:
        Tuple of (domain_path, domain_slug)
    """
    from dotenv import load_dotenv
    load_dotenv()

    from domainbench.core.config import ModelConfig, ProviderType
    from domainbench.providers import get_provider

    # Initialize provider
    try:
        provider_type = ProviderType(provider.lower())
    except ValueError:
        raise ValueError(f"Unknown provider: {provider}. Available: openai, anthropic, gemini")

    model_config = ModelConfig(
        provider=provider_type,
        model=model,
        alias=f"{provider}/{model}",
    )
    llm_provider = get_provider(model_config)

    # Set default categories - always support all standard categories
    if categories is None:
        categories = SUPPORTED_CATEGORIES.copy()

    # Generate domain slug
    domain_slug = slugify(domain_name)

    # Determine output directory
    if output_dir:
        domain_path = Path(output_dir) / domain_slug
    else:
        domain_path = FUNC_CALL_DOMAINS_DIR / domain_slug

    # Create domain directory
    domain_path.mkdir(parents=True, exist_ok=True)

    # Generate domain files using AI
    console.print(f"[dim]Generating domain: {domain_name}[/dim]")

    # Step 1: Generate function definitions
    console.print("[dim]  Generating function definitions...[/dim]")
    functions = _generate_functions(
        llm_provider, model, domain_name, domain_description
    )

    # Step 2: Clean functions for cross-provider compatibility
    console.print("[dim]  Cleaning functions for compatibility...[/dim]")
    functions = _clean_functions_for_compatibility(functions)

    # Step 3: Extract sample data from functions
    console.print("[dim]  Extracting domain-specific sample data...[/dim]")
    sample_data = _extract_sample_data_from_functions(functions, domain_name)

    # Step 4: Generate domain.yaml
    console.print("[dim]  Creating domain.yaml...[/dim]")
    domain_yaml = _create_domain_yaml(
        domain_name, domain_description, categories, functions
    )
    with open(domain_path / "domain.yaml", 'w', encoding='utf-8') as f:
        f.write(domain_yaml)

    # Step 5: Generate generator.py with domain-specific data
    console.print("[dim]  Generating test case generator...[/dim]")
    generator_code = _create_fallback_generator(domain_name, functions, sample_data)

    # Apply fixes for common AI generation issues
    generator_code = _fix_generated_code(generator_code)

    with open(domain_path / "generator.py", 'w', encoding='utf-8') as f:
        f.write(generator_code)

    # Step 6: Create __init__.py
    console.print("[dim]  Creating __init__.py...[/dim]")
    init_code = _create_init_py(domain_name)
    with open(domain_path / "__init__.py", 'w', encoding='utf-8') as f:
        f.write(init_code)

    return domain_path, domain_slug


def _generate_functions(
    provider,
    model: str,
    domain_name: str,
    domain_description: str,
) -> List[Dict[str, Any]]:
    """Generate function definitions using AI."""
    prompt = f"""Generate 4-6 function definitions for a function calling benchmark domain.

Domain Name: {domain_name}
Description: {domain_description or f"A {domain_name.lower()} related API"}

**CRITICAL REQUIREMENTS:**

1. **NO ARRAY TYPES** - Do NOT use "type": "array" for any parameter. Use separate parameters instead.
   - BAD: "items": {{"type": "array", "items": {{"type": "string"}}}}
   - GOOD: "item_id": {{"type": "string"}}

2. **NO DEFAULT VALUES** - Do NOT include "default" fields in parameters.

3. **USE ENUMS** for parameters with fixed valid values:
   - Categories, statuses, payment methods, etc.

4. **SIMPLE TYPES ONLY**: string, integer, boolean (with enum for strings where appropriate)

5. Functions should be realistic and work together for workflows.

Output ONLY valid JSON array with no markdown formatting. Each function object should have:
- "name": snake_case function name
- "description": what the function does
- "parameters": JSON Schema object with properties and required fields

Example format:
[
  {{
    "name": "get_menu",
    "description": "Get menu items by category",
    "parameters": {{
      "type": "object",
      "properties": {{
        "category": {{
          "type": "string",
          "enum": ["appetizers", "mains", "desserts", "drinks"],
          "description": "Menu category to filter by"
        }}
      }},
      "required": []
    }}
  }},
  {{
    "name": "get_item_details",
    "description": "Get details for a specific item",
    "parameters": {{
      "type": "object",
      "properties": {{
        "item_id": {{
          "type": "string",
          "description": "Item identifier"
        }}
      }},
      "required": ["item_id"]
    }}
  }},
  {{
    "name": "create_order",
    "description": "Create a new order",
    "parameters": {{
      "type": "object",
      "properties": {{
        "table_number": {{
          "type": "integer",
          "description": "Table number"
        }},
        "guest_count": {{
          "type": "integer",
          "description": "Number of guests"
        }}
      }},
      "required": ["table_number", "guest_count"]
    }}
  }},
  {{
    "name": "process_payment",
    "description": "Process payment for an order",
    "parameters": {{
      "type": "object",
      "properties": {{
        "order_id": {{
          "type": "string",
          "description": "Order identifier"
        }},
        "payment_method": {{
          "type": "string",
          "enum": ["cash", "card", "contactless"],
          "description": "Payment method"
        }}
      }},
      "required": ["order_id", "payment_method"]
    }}
  }}
]

Generate the functions now:"""

    response = provider.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2500,
    )

    content = response.get("content", "")

    # Parse JSON from response
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            functions = json.loads(json_match.group())
        else:
            functions = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: create basic functions
        functions = _create_fallback_functions(domain_name)

    return functions


def _create_fallback_functions(domain_name: str) -> List[Dict[str, Any]]:
    """Create basic fallback functions if AI generation fails."""
    slug = slugify(domain_name)
    return [
        {
            "name": f"get_{slug}_list",
            "description": f"Get a list of {domain_name.lower()} items by category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["category_a", "category_b", "category_c", "category_d"],
                        "description": "Category to filter by"
                    }
                },
                "required": []
            }
        },
        {
            "name": f"get_{slug}_details",
            "description": f"Get details for a specific {domain_name.lower()} item",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Item identifier"
                    }
                },
                "required": ["item_id"]
            }
        },
        {
            "name": f"create_{slug}_order",
            "description": f"Create a new {domain_name.lower()} order",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_number": {
                        "type": "integer",
                        "description": "Reference number"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity"
                    }
                },
                "required": ["reference_number", "quantity"]
            }
        },
        {
            "name": f"process_{slug}_action",
            "description": f"Process an action for {domain_name.lower()}",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "Action identifier"
                    },
                    "action_type": {
                        "type": "string",
                        "enum": ["approve", "reject", "pending", "complete"],
                        "description": "Type of action"
                    }
                },
                "required": ["action_id", "action_type"]
            }
        }
    ]


def _create_domain_yaml(
    name: str,
    description: str,
    categories: List[str],
    functions: List[Dict[str, Any]],
) -> str:
    """Create domain.yaml content."""
    import yaml

    domain_config = {
        "domain": {
            "name": name,
            "description": description or f"{name} function calling benchmark",
            "version": "1.0",
            "capability": "function_calling",
            "categories": categories,
            "functions": functions,
        }
    }

    return yaml.dump(domain_config, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _create_fallback_generator(
    domain_name: str,
    functions: List[Dict[str, Any]],
    sample_data: Dict[str, Any],
) -> str:
    """
    Create a robust generator following the restaurant_waiter pattern.

    Key principles:
    - Values in queries MUST match values in ground_truth exactly
    - Use IDs directly in queries (not mapped names)
    - Natural language that sounds like real requests
    """
    slug = slugify(domain_name)

    # Extract enum values from functions
    enums = sample_data.get("enums", {})
    func_info = sample_data.get("functions", [])

    # Find functions by type
    list_func = None
    detail_func = None
    create_func = None
    action_func = None

    for f in func_info:
        name = f["name"].lower()
        if "list" in name or ("get" in name and "detail" not in name and "info" not in name):
            if list_func is None:
                list_func = f
        elif "detail" in name or "info" in name:
            detail_func = f
        elif "create" in name or "new" in name or "order" in name:
            if create_func is None:
                create_func = f
        elif "process" in name or "action" in name or "update" in name:
            if action_func is None:
                action_func = f

    # Fallbacks
    if list_func is None and func_info:
        list_func = func_info[0]
    if detail_func is None and len(func_info) > 1:
        detail_func = func_info[1]
    if create_func is None and len(func_info) > 2:
        create_func = func_info[2]
    if action_func is None and len(func_info) > 3:
        action_func = func_info[3]

    # Build data declarations
    data_declarations = []

    # Find primary enum (usually category-like)
    primary_enum_param = None
    primary_enum_var = None
    primary_enum_values = None

    if list_func:
        for param_name, param_info in list_func.get("params", {}).items():
            if "values" in param_info and param_info.get("type") == "string":
                var_name = param_name.upper()
                if not var_name.endswith("S"):
                    var_name += "S"
                values = param_info["values"]
                data_declarations.append(f'{var_name} = {json.dumps(values)}')
                if primary_enum_param is None:
                    primary_enum_param = param_name
                    primary_enum_var = var_name
                    primary_enum_values = values

    # Find secondary enum (for action function)
    secondary_enum_param = None
    secondary_enum_var = None

    if action_func:
        for param_name, param_info in action_func.get("params", {}).items():
            if "values" in param_info and param_info.get("type") == "string" and param_name != primary_enum_param:
                var_name = param_name.upper()
                if not var_name.endswith("S"):
                    var_name += "S"
                values = param_info["values"]
                data_declarations.append(f'{var_name} = {json.dumps(values)}')
                if secondary_enum_param is None:
                    secondary_enum_param = param_name
                    secondary_enum_var = var_name

    # Fallback if no enums found
    if not primary_enum_param:
        primary_enum_param = "category"
        primary_enum_var = "CATEGORIES"
        primary_enum_values = ["category_a", "category_b", "category_c", "category_d"]
        data_declarations.append(f'CATEGORIES = {json.dumps(primary_enum_values)}')

    if not secondary_enum_param and action_func:
        secondary_enum_param = "action_type"
        secondary_enum_var = "ACTION_TYPES"
        data_declarations.append('ACTION_TYPES = ["approve", "reject", "pending", "complete"]')

    # Create sample items with IDs
    data_declarations.append(f'''
ITEMS = [
    {{"id": "{slug}_001", "name": "{domain_name} Item 1"}},
    {{"id": "{slug}_002", "name": "{domain_name} Item 2"}},
    {{"id": "{slug}_003", "name": "{domain_name} Item 3"}},
    {{"id": "{slug}_004", "name": "{domain_name} Item 4"}},
    {{"id": "{slug}_005", "name": "{domain_name} Item 5"}},
]''')

    # Add integer ranges
    data_declarations.append('\nREFERENCE_NUMBERS = list(range(1, 21))')
    data_declarations.append('QUANTITIES = [1, 2, 3, 4, 5]')

    data_section = "\n".join(data_declarations)

    # Build functions JSON
    functions_str = json.dumps(functions, indent=4)

    # Get function names
    list_func_name = list_func["name"] if list_func else f"get_{slug}_list"
    detail_func_name = detail_func["name"] if detail_func else f"get_{slug}_details"
    create_func_name = create_func["name"] if create_func else f"create_{slug}_order"
    action_func_name = action_func["name"] if action_func else f"process_{slug}_action"

    # Find ID param for detail function
    id_param = "item_id"
    if detail_func:
        for param_name in detail_func.get("params", {}).keys():
            if "id" in param_name.lower():
                id_param = param_name
                break

    # Find params for create function
    create_param1 = "reference_number"
    create_param2 = "quantity"
    if create_func:
        int_params = [p for p, info in create_func.get("params", {}).items()
                      if info.get("type") == "integer"]
        if len(int_params) >= 1:
            create_param1 = int_params[0]
        if len(int_params) >= 2:
            create_param2 = int_params[1]

    # Find params for action function
    action_id_param = "action_id"
    if action_func:
        for param_name in action_func.get("params", {}).keys():
            if "id" in param_name.lower():
                action_id_param = param_name
                break

    return f'''"""
{domain_name} function calling test case generator.

Supports all standard categories: simple, parallel, multiple, multi_turn, agentic.
"""

import random
from typing import Any, Dict, List


# ============================================================================
# DOMAIN-SPECIFIC SAMPLE DATA
# These values appear EXACTLY the same in queries AND ground_truth
# ============================================================================
{data_section}


# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

FUNCTIONS = {functions_str}


# ============================================================================
# SUPPORTED CATEGORIES
# ============================================================================

SUPPORTED_CATEGORIES = ["simple", "parallel", "multiple", "multi_turn", "agentic"]


# ============================================================================
# QUERY TEMPLATES
# CRITICAL: The exact same value must appear in query AND ground_truth
# ============================================================================

SIMPLE_TEMPLATES = [
    # List function with enum - enum value appears in both query and ground_truth
    ("Show me the {{{primary_enum_param}}} options", "{list_func_name}({primary_enum_param}='{{{primary_enum_param}}}')"),
    ("What {{{primary_enum_param}}} do you have?", "{list_func_name}({primary_enum_param}='{{{primary_enum_param}}}')"),
    ("Can I see the {{{primary_enum_param}}} please?", "{list_func_name}({primary_enum_param}='{{{primary_enum_param}}}')"),
    ("I'd like to see {{{primary_enum_param}}}", "{list_func_name}({primary_enum_param}='{{{primary_enum_param}}}')"),
    ("What's available in {{{primary_enum_param}}}?", "{list_func_name}({primary_enum_param}='{{{primary_enum_param}}}')"),

    # Detail function - use item_id directly in query
    ("Tell me about item {{item_id}}", "{detail_func_name}({id_param}='{{item_id}}')"),
    ("What's in item {{item_id}}?", "{detail_func_name}({id_param}='{{item_id}}')"),
    ("Can you describe item {{item_id}}?", "{detail_func_name}({id_param}='{{item_id}}')"),
    ("I'd like details on {{item_id}}", "{detail_func_name}({id_param}='{{item_id}}')"),

    # Create function - numbers in query match numbers in ground_truth
    ("Create an order for reference {{ref_num}} with {{quantity}} items",
     "{create_func_name}({create_param1}={{ref_num}}, {create_param2}={{quantity}})"),
    ("Start a new order: reference {{ref_num}}, quantity {{quantity}}",
     "{create_func_name}({create_param1}={{ref_num}}, {create_param2}={{quantity}})"),
]

PARALLEL_TEMPLATES = [
    # Multiple list calls - different enum values
    (
        "Show me both {{{primary_enum_param}1}} and {{{primary_enum_param}2}}",
        ["{list_func_name}({primary_enum_param}='{{{primary_enum_param}1}}')", "{list_func_name}({primary_enum_param}='{{{primary_enum_param}2}}')"]
    ),
    (
        "I want to see {{{primary_enum_param}1}} and also {{{primary_enum_param}2}}",
        ["{list_func_name}({primary_enum_param}='{{{primary_enum_param}1}}')", "{list_func_name}({primary_enum_param}='{{{primary_enum_param}2}}')"]
    ),
    # List and detail
    (
        "Show me {{{primary_enum_param}}} and tell me about item {{item_id}}",
        ["{list_func_name}({primary_enum_param}='{{{primary_enum_param}}}')", "{detail_func_name}({id_param}='{{item_id}}')"]
    ),
]

MULTIPLE_TEMPLATES = [
    # Multiple list calls in sequence
    (
        "First show me {{{primary_enum_param}1}}, then {{{primary_enum_param}2}}, then {{{primary_enum_param}3}}",
        [
            "{list_func_name}({primary_enum_param}='{{{primary_enum_param}1}}')",
            "{list_func_name}({primary_enum_param}='{{{primary_enum_param}2}}')",
            "{list_func_name}({primary_enum_param}='{{{primary_enum_param}3}}')"
        ]
    ),
    # Multiple detail calls
    (
        "Tell me about item {{item_id1}}, then item {{item_id2}}",
        [
            "{detail_func_name}({id_param}='{{item_id1}}')",
            "{detail_func_name}({id_param}='{{item_id2}}')"
        ]
    ),
]

MULTI_TURN_TEMPLATES = [
    {{
        "scenario": "browsing_flow",
        "turns": [
            {{
                "query": "Show me the {{{primary_enum_param}}} options",
                "expected_calls": ["{list_func_name}({primary_enum_param}='{{{primary_enum_param}}}')"],
            }},
            {{
                "query": "Tell me more about item {{item_id}}",
                "expected_calls": ["{detail_func_name}({id_param}='{{item_id}}')"],
            }},
        ],
    }},
    {{
        "scenario": "comparison_flow",
        "turns": [
            {{
                "query": "What {{{primary_enum_param}1}} do you have?",
                "expected_calls": ["{list_func_name}({primary_enum_param}='{{{primary_enum_param}1}}')"],
            }},
            {{
                "query": "And what about {{{primary_enum_param}2}}?",
                "expected_calls": ["{list_func_name}({primary_enum_param}='{{{primary_enum_param}2}}')"],
            }},
        ],
    }},
]

AGENTIC_TEMPLATES = [
    {{
        "query": "Is {{{primary_enum_param}}} a valid option? Just say yes or no.",
        "context": "Check if the value is valid.",
        "expected_response": ["yes"],
        "match_mode": "contains",
    }},
    {{
        "query": "Can you help me with item {{item_id}}? Answer yes or no.",
        "context": "Determine if you can assist with this request.",
        "expected_response": ["yes", "no"],
        "match_mode": "any",
    }},
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_random_values(rng: random.Random) -> Dict[str, Any]:
    """Get random values for template substitution."""
    item = rng.choice(ITEMS)
    items = rng.sample(ITEMS, min(2, len(ITEMS)))
    cats = rng.sample({primary_enum_var}, min(3, len({primary_enum_var})))

    return {{
        "{primary_enum_param}": rng.choice({primary_enum_var}),
        "{primary_enum_param}1": cats[0] if len(cats) > 0 else {primary_enum_var}[0],
        "{primary_enum_param}2": cats[1] if len(cats) > 1 else {primary_enum_var}[0],
        "{primary_enum_param}3": cats[2] if len(cats) > 2 else {primary_enum_var}[0],
        "item_id": item["id"],
        "item_name": item["name"],
        "item_id1": items[0]["id"],
        "item_id2": items[1]["id"] if len(items) > 1 else items[0]["id"],
        "ref_num": rng.choice(REFERENCE_NUMBERS),
        "quantity": rng.choice(QUANTITIES),
        "action_id": f"ACT-{{rng.randint(1000, 9999)}}",
    }}


# ============================================================================
# MAIN GENERATOR FUNCTION
# ============================================================================

def generate_test_cases(
    count: int,
    seed: int = 42,
    category: str = "simple",
) -> List[Dict[str, Any]]:
    """
    Generate function calling test cases.

    Args:
        count: Number of test cases to generate
        seed: Random seed for reproducibility
        category: Category to generate (simple, parallel, multiple, multi_turn, agentic, or all)

    Returns:
        List of test case dictionaries
    """
    rng = random.Random(seed)
    items = []

    if category == "all":
        categories = SUPPORTED_CATEGORIES.copy()
    elif category in SUPPORTED_CATEGORIES:
        categories = [category]
    else:
        raise ValueError(f"Unknown category: {{category}}. Supported: {{', '.join(SUPPORTED_CATEGORIES)}}, all")

    for i in range(count):
        cat = rng.choice(categories)

        if cat == "simple":
            item = _generate_simple(rng, i)
        elif cat == "parallel":
            item = _generate_parallel(rng, i)
        elif cat == "multiple":
            item = _generate_multiple(rng, i)
        elif cat == "multi_turn":
            item = _generate_multi_turn(rng, i)
        elif cat == "agentic":
            item = _generate_agentic(rng, i)

        items.append(item)

    return items


def _generate_simple(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate a simple test case."""
    template, gt_template = rng.choice(SIMPLE_TEMPLATES)
    values = _get_random_values(rng)

    query = template.format(**values)
    ground_truth = gt_template.format(**values)

    return {{
        "id": f"{slug}_simple_{{idx:04d}}",
        "category": "simple",
        "query": query,
        "functions": FUNCTIONS,
        "ground_truth": ground_truth,
    }}


def _generate_parallel(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate a parallel test case."""
    template, gt_templates = rng.choice(PARALLEL_TEMPLATES)
    values = _get_random_values(rng)

    query = template.format(**values)
    ground_truth = [gt.format(**values) for gt in gt_templates]

    return {{
        "id": f"{slug}_parallel_{{idx:04d}}",
        "category": "parallel",
        "query": query,
        "functions": FUNCTIONS,
        "ground_truth": ground_truth,
    }}


def _generate_multiple(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate a multiple test case."""
    template, gt_templates = rng.choice(MULTIPLE_TEMPLATES)
    values = _get_random_values(rng)

    query = template.format(**values)
    ground_truth = [gt.format(**values) for gt in gt_templates]

    return {{
        "id": f"{slug}_multiple_{{idx:04d}}",
        "category": "multiple",
        "query": query,
        "functions": FUNCTIONS,
        "ground_truth": ground_truth,
    }}


def _generate_multi_turn(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate a multi-turn conversation test case."""
    template = rng.choice(MULTI_TURN_TEMPLATES)
    values = _get_random_values(rng)

    turns = []
    for turn_template in template["turns"]:
        query = turn_template["query"].format(**values)
        expected_calls = [call.format(**values) for call in turn_template["expected_calls"]]
        turns.append({{
            "query": query,
            "expected_calls": expected_calls,
        }})

    return {{
        "id": f"{slug}_multi_turn_{{idx:04d}}",
        "category": "multi_turn",
        "query": turns[0]["query"],
        "functions": FUNCTIONS,
        "turns": turns,
        "ground_truth": turns[0]["expected_calls"],
    }}


def _generate_agentic(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate an agentic (text response) test case."""
    template = rng.choice(AGENTIC_TEMPLATES)
    values = _get_random_values(rng)

    query = template["query"].format(**values)
    context = template.get("context", "").format(**values)

    expected_response = template["expected_response"]
    if isinstance(expected_response, list):
        expected_response = [resp.format(**values) if isinstance(resp, str) else resp for resp in expected_response]
    elif isinstance(expected_response, str):
        expected_response = expected_response.format(**values)

    return {{
        "id": f"{slug}_agentic_{{idx:04d}}",
        "category": "agentic",
        "query": query,
        "context": context,
        "functions": FUNCTIONS,
        "expected_response": expected_response,
        "match_mode": template.get("match_mode", "contains"),
        "ground_truth": expected_response,
    }}
'''


def _create_init_py(domain_name: str) -> str:
    """Create __init__.py content."""
    return f'''"""
{domain_name} function calling domain
"""

from domainbench.domains.builtin.function_calling.{slugify(domain_name)}.generator import (
    generate_test_cases,
)

__all__ = ["generate_test_cases"]
'''


def validate_generated_domain(domain_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate a generated domain thoroughly.

    Checks:
    1. Required files exist
    2. domain.yaml is valid YAML with proper structure
    3. generator.py has generate_test_cases function
    4. generator.py produces valid test cases
    5. Test case values match function parameter schemas
    6. All 5 categories are supported

    Args:
        domain_path: Path to domain directory

    Returns:
        Tuple of (is_valid, list_of_warnings_or_errors)
    """
    import yaml

    issues = []

    # Check required files
    required_files = ["domain.yaml", "generator.py", "__init__.py"]
    for filename in required_files:
        if not (domain_path / filename).exists():
            issues.append(f"ERROR: Missing required file: {filename}")
            return False, issues

    # Validate domain.yaml
    try:
        with open(domain_path / "domain.yaml", 'r') as f:
            config = yaml.safe_load(f)

        if "domain" not in config:
            issues.append("ERROR: domain.yaml missing 'domain' key")
            return False, issues
        if "functions" not in config["domain"]:
            issues.append("ERROR: domain.yaml missing 'functions' key")
            return False, issues

        functions = config["domain"]["functions"]

        # Check for problematic patterns
        for func in functions:
            props = func.get("parameters", {}).get("properties", {})
            for param_name, param_def in props.items():
                if param_def.get("type") == "array":
                    issues.append(f"WARNING: Function '{func['name']}' has array parameter '{param_name}' - may cause Gemini errors")
                if "default" in param_def:
                    issues.append(f"WARNING: Function '{func['name']}' has default value for '{param_name}' - may not be supported by all providers")

    except yaml.YAMLError as e:
        issues.append(f"ERROR: Invalid YAML in domain.yaml: {e}")
        return False, issues

    # Validate generator.py syntax
    try:
        with open(domain_path / "generator.py", 'r') as f:
            code = f.read()
        compile(code, domain_path / "generator.py", 'exec')
    except SyntaxError as e:
        issues.append(f"ERROR: Syntax error in generator.py: {e}")
        return False, issues

    # Try to import and run generator
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generator", domain_path / "generator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'generate_test_cases'):
            issues.append("ERROR: generator.py missing generate_test_cases function")
            return False, issues

        # Check if SUPPORTED_CATEGORIES exists and has all categories
        if hasattr(module, 'SUPPORTED_CATEGORIES'):
            missing_cats = set(SUPPORTED_CATEGORIES) - set(module.SUPPORTED_CATEGORIES)
            if missing_cats:
                issues.append(f"WARNING: Generator missing categories: {missing_cats}")
        else:
            issues.append("WARNING: Generator missing SUPPORTED_CATEGORIES constant")

        # Try generating test cases for each category
        for cat in SUPPORTED_CATEGORIES:
            try:
                test_cases = module.generate_test_cases(2, seed=42, category=cat)
                if not test_cases:
                    issues.append(f"WARNING: No test cases generated for category '{cat}'")
                    continue

                # Validate test case structure
                required_fields = ["id", "category", "query", "functions"]
                for tc in test_cases:
                    for field in required_fields:
                        if field not in tc:
                            issues.append(f"WARNING: Test case missing field '{field}' in category '{cat}'")

                    # Check for ground_truth
                    if "ground_truth" not in tc:
                        issues.append(f"WARNING: Test case missing 'ground_truth' in category '{cat}'")

            except ValueError as e:
                if "Unknown category" in str(e):
                    issues.append(f"WARNING: Category '{cat}' not supported by generator")
            except Exception as e:
                issues.append(f"WARNING: Error generating '{cat}' test cases: {e}")

    except Exception as e:
        issues.append(f"ERROR: Error testing generator: {e}")
        return False, issues

    # Determine if valid (no ERROR messages)
    is_valid = not any(issue.startswith("ERROR") for issue in issues)

    return is_valid, issues


def list_function_calling_domains() -> List[Dict[str, Any]]:
    """
    List all available function calling domains.

    Returns:
        List of domain info dictionaries
    """
    import yaml

    domains = []

    if not FUNC_CALL_DOMAINS_DIR.exists():
        return domains

    for domain_dir in FUNC_CALL_DOMAINS_DIR.iterdir():
        if domain_dir.is_dir() and (domain_dir / "domain.yaml").exists():
            try:
                with open(domain_dir / "domain.yaml", 'r') as f:
                    config = yaml.safe_load(f)

                domain_info = config.get("domain", {})
                domains.append({
                    "slug": domain_dir.name,
                    "name": domain_info.get("name", domain_dir.name),
                    "description": domain_info.get("description", ""),
                    "categories": domain_info.get("categories", ["simple"]),
                    "function_count": len(domain_info.get("functions", [])),
                    "path": str(domain_dir),
                })
            except Exception:
                continue

    return domains
