"""
AI-Powered Domain Creator for Function Calling Capability.

Uses LLM to generate new function calling benchmark domains with:
- domain.yaml (function definitions)
- generator.py (test case generator)
- __init__.py (module exports)

Key improvements:
- Generates domain-specific sample data from function parameter enums
- Creates natural language query templates
- Produces ground_truth with valid parameter values
- Auto-fixes common AI generation issues
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
        # For simple/parallel/multiple, use ground_truth
        code = code.replace('"expected_calls":', '"ground_truth":')

    return code


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
            elif param_def.get("type") == "array" and "items" in param_def:
                items = param_def["items"]
                if "enum" in items:
                    key = f"{func['name']}_{param_name}"
                    enums[key] = items["enum"]
                    if param_name not in enums:
                        enums[param_name] = items["enum"]

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
            elif param_def.get("type") == "array" and "items" in param_def:
                if "enum" in param_def["items"]:
                    param_info["values"] = param_def["items"]["enum"]
            elif param_def.get("type") == "integer":
                # Generate sample integers based on context
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

    # Step 2: Extract sample data from functions
    console.print("[dim]  Extracting domain-specific sample data...[/dim]")
    sample_data = _extract_sample_data_from_functions(functions, domain_name)

    # Step 3: Generate domain.yaml
    console.print("[dim]  Creating domain.yaml...[/dim]")
    domain_yaml = _create_domain_yaml(
        domain_name, domain_description, categories, functions
    )
    with open(domain_path / "domain.yaml", 'w', encoding='utf-8') as f:
        f.write(domain_yaml)

    # Step 4: Generate generator.py with domain-specific data
    console.print("[dim]  Generating test case generator...[/dim]")
    generator_code = _generate_generator(
        llm_provider, model, domain_name, functions, categories, sample_data
    )

    # Apply fixes for common AI generation issues
    generator_code = _fix_generated_code(generator_code)

    with open(domain_path / "generator.py", 'w', encoding='utf-8') as f:
        f.write(generator_code)

    # Step 5: Create __init__.py
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

Requirements:
1. Functions should be realistic and useful for the domain
2. Each function should have clear parameters with types
3. Include a mix of required and optional parameters
4. **IMPORTANT**: Use enum types for parameters that have a fixed set of valid values
   - For example: category types, status values, payment methods, etc.
5. Use appropriate parameter types (string, integer, boolean, array)
6. Functions should work together for realistic workflows

Output ONLY valid JSON array with no markdown formatting. Each function object should have:
- "name": snake_case function name
- "description": what the function does
- "parameters": JSON Schema object with properties and required fields

Example format with enum:
[
  {{
    "name": "get_products",
    "description": "Get products by category",
    "parameters": {{
      "type": "object",
      "properties": {{
        "category": {{
          "type": "string",
          "enum": ["electronics", "clothing", "food", "books"],
          "description": "Product category to filter by"
        }},
        "in_stock": {{
          "type": "boolean",
          "description": "Only show in-stock items"
        }}
      }},
      "required": ["category"]
    }}
  }},
  {{
    "name": "get_product_details",
    "description": "Get details for a specific product",
    "parameters": {{
      "type": "object",
      "properties": {{
        "product_id": {{
          "type": "string",
          "description": "Product identifier"
        }}
      }},
      "required": ["product_id"]
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
            "description": f"Get a list of {domain_name.lower()} items",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["category_a", "category_b", "category_c"],
                        "description": "Category to filter by"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results"
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
            "name": f"create_{slug}",
            "description": f"Create a new {domain_name.lower()} item",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Item name"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["category_a", "category_b", "category_c"],
                        "description": "Item category"
                    }
                },
                "required": ["name", "category"]
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


def _generate_generator(
    provider,
    model: str,
    domain_name: str,
    functions: List[Dict[str, Any]],
    categories: List[str],
    sample_data: Dict[str, Any],
) -> str:
    """Generate test case generator code using AI with domain-specific data."""
    functions_json = json.dumps(functions, indent=2)
    sample_data_json = json.dumps(sample_data, indent=2)
    slug = slugify(domain_name)

    prompt = f"""Generate a Python test case generator for a function calling benchmark.

Domain: {domain_name}
Domain Slug: {slug}

Functions available:
{functions_json}

Extracted sample data from functions (USE THESE VALUES):
{sample_data_json}

**CRITICAL REQUIREMENTS:**

1. **USE DOMAIN-SPECIFIC SAMPLE DATA** - Extract values from function enums and use them directly:
   - If a function has `"category": {{"enum": ["electronics", "clothing", "food"]}}`, create:
     `CATEGORIES = ["electronics", "clothing", "food"]`
   - If a function has `"payment_method": {{"enum": ["cash", "card", "crypto"]}}`, create:
     `PAYMENT_METHODS = ["cash", "card", "crypto"]`

2. **CREATE NATURAL LANGUAGE TEMPLATES** - Queries should sound like real user requests:
   - BAD: "Please get_products with electronics"
   - GOOD: "Show me all electronics products"
   - GOOD: "What products do you have in the clothing category?"

3. **GROUND TRUTH MUST USE VALID PARAMETER VALUES** - The function calls in ground_truth must use values from the enums:
   - BAD: `get_products(category='item_1')`
   - GOOD: `get_products(category='electronics')`

4. **SUPPORT ALL 5 CATEGORIES**:
   - simple: Single function call
   - parallel: Multiple independent calls (order doesn't matter)
   - multiple: Same function called multiple times (order matters)
   - multi_turn: Sequential conversation with multiple turns
   - agentic: Complex tasks with yes/no text response validation

5. **USE PYTHON BOOLEANS**: Use `True` and `False`, NOT `true` and `false`

Generate a complete Python module with this EXACT structure:

```python
\"\"\"
{domain_name} function calling test case generator.
Supports all standard categories: simple, parallel, multiple, multi_turn, agentic.
\"\"\"

import random
from typing import Any, Dict, List


# ============================================================================
# DOMAIN-SPECIFIC SAMPLE DATA
# Extract these from function parameter enums - use REAL values, not generic ones
# ============================================================================

# Example: If function has category enum ["electronics", "clothing", "food"]
CATEGORIES = [...]  # Extract from function enums

# Example: If function has items with IDs
ITEMS = [
    {{"id": "item_001", "name": "Sample Item 1"}},
    {{"id": "item_002", "name": "Sample Item 2"}},
    ...
]

# Add more domain-specific data based on function parameters...


# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

FUNCTIONS = {functions_json}


# ============================================================================
# SUPPORTED CATEGORIES
# ============================================================================

SUPPORTED_CATEGORIES = ["simple", "parallel", "multiple", "multi_turn", "agentic"]


# ============================================================================
# QUERY TEMPLATES
# Use natural language that sounds like real user requests
# Variables use {{var}} format for string substitution
# ============================================================================

SIMPLE_TEMPLATES = [
    # (query_template, ground_truth_template)
    # Use natural language queries!
    ("Show me {{category}} items", "get_items(category='{{category}}')"),
    ("What's available in {{category}}?", "get_items(category='{{category}}')"),
    ("Tell me about item {{item_id}}", "get_item_details(item_id='{{item_id}}')"),
]

PARALLEL_TEMPLATES = [
    # (query_template, [ground_truth_templates])
    # Multiple independent function calls
    (
        "Show me both {{cat1}} and {{cat2}} items",
        ["get_items(category='{{cat1}}')", "get_items(category='{{cat2}}')"]
    ),
]

MULTIPLE_TEMPLATES = [
    # Same function called multiple times, order matters
    (
        "First show me {{cat1}}, then {{cat2}}, then {{cat3}}",
        [
            "get_items(category='{{cat1}}')",
            "get_items(category='{{cat2}}')",
            "get_items(category='{{cat3}}')"
        ]
    ),
]

MULTI_TURN_TEMPLATES = [
    {{
        "scenario": "browsing_flow",
        "turns": [
            {{
                "query": "Show me {{category}} items",
                "expected_calls": ["get_items(category='{{category}}')"],
            }},
            {{
                "query": "Tell me more about {{item_name}}",
                "expected_calls": ["get_item_details(item_id='{{item_id}}')"],
            }},
        ],
    }},
]

AGENTIC_TEMPLATES = [
    {{
        "query": "Is {{category}} a valid category? Just say yes or no.",
        "context": "Check if the category exists in the system.",
        "expected_response": ["yes", "no"],
        "match_mode": "any",
    }},
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_random_values(rng: random.Random) -> Dict[str, Any]:
    \"\"\"Get random values for template substitution.\"\"\"
    # Select random items from domain-specific data
    item = rng.choice(ITEMS)
    items = rng.sample(ITEMS, min(2, len(ITEMS)))
    cats = rng.sample(CATEGORIES, min(3, len(CATEGORIES)))

    return {{
        "category": rng.choice(CATEGORIES),
        "cat1": cats[0] if len(cats) > 0 else CATEGORIES[0],
        "cat2": cats[1] if len(cats) > 1 else CATEGORIES[0],
        "cat3": cats[2] if len(cats) > 2 else CATEGORIES[0],
        "item_id": item["id"],
        "item_name": item["name"],
        # Add more as needed...
    }}


# ============================================================================
# MAIN GENERATOR FUNCTION
# ============================================================================

def generate_test_cases(
    count: int,
    seed: int = 42,
    category: str = "simple",
) -> List[Dict[str, Any]]:
    \"\"\"
    Generate function calling test cases.

    Args:
        count: Number of test cases to generate
        seed: Random seed for reproducibility
        category: Category to generate (simple, parallel, multiple, multi_turn, agentic, or all)

    Returns:
        List of test case dictionaries
    \"\"\"
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
    \"\"\"Generate a simple test case.\"\"\"
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
    \"\"\"Generate a parallel test case.\"\"\"
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
    \"\"\"Generate a multiple test case.\"\"\"
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
    \"\"\"Generate a multi-turn conversation test case.\"\"\"
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
    \"\"\"Generate an agentic (text response) test case.\"\"\"
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
```

**IMPORTANT REMINDERS:**
- Use REAL enum values from the functions, not generic placeholders
- Create natural language queries, not function-like commands
- Ensure ground_truth uses valid parameter values
- Use Python True/False, not JSON true/false
- ALL 5 categories must be fully implemented

Generate the complete code now:"""

    response = provider.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8000,
    )

    content = response.get("content", "")

    # Extract code from markdown if present
    code_match = re.search(r'```python\s*([\s\S]*?)\s*```', content)
    if code_match:
        code = code_match.group(1)
    else:
        # Try to use content directly if it looks like Python
        if 'def generate_test_cases' in content:
            code = content
        else:
            # Fallback to a robust generator
            code = _create_fallback_generator(domain_name, functions, sample_data)

    return code


def _create_fallback_generator(
    domain_name: str,
    functions: List[Dict[str, Any]],
    sample_data: Dict[str, Any],
) -> str:
    """
    Create a robust fallback generator that uses actual function schemas.

    This generator extracts real values from function parameter enums
    to ensure test cases use valid parameter values.
    """
    slug = slugify(domain_name)

    # Extract enum values from functions
    enums = sample_data.get("enums", {})
    func_info = sample_data.get("functions", [])

    # Build domain-specific data declarations
    data_declarations = []
    value_mappings = {}

    # Find the primary list function and its category enum
    primary_categories = None
    primary_func = None
    detail_func = None

    for f in func_info:
        name = f["name"]
        if "list" in name or "get" in name and "detail" not in name:
            if primary_func is None:
                primary_func = f
        if "detail" in name or "info" in name:
            detail_func = f

    if primary_func is None and func_info:
        primary_func = func_info[0]
    if detail_func is None and len(func_info) > 1:
        detail_func = func_info[1]
    elif detail_func is None and func_info:
        detail_func = func_info[0]

    # Extract categories from primary function
    if primary_func:
        for param_name, param_info in primary_func.get("params", {}).items():
            if "values" in param_info and param_info.get("type") == "string":
                var_name = param_name.upper() + "S" if not param_name.upper().endswith("S") else param_name.upper()
                values = param_info["values"]
                data_declarations.append(f'{var_name} = {json.dumps(values)}')
                value_mappings[param_name] = var_name
                if primary_categories is None:
                    primary_categories = (param_name, var_name, values)

    # If no enums found, create generic but domain-specific categories
    if not data_declarations:
        data_declarations.append(f'CATEGORIES = ["category_a", "category_b", "category_c", "category_d"]')
        primary_categories = ("category", "CATEGORIES", ["category_a", "category_b", "category_c", "category_d"])

    # Create sample items based on domain
    domain_items = [
        {"id": f"{slug}_001", "name": f"{domain_name} Item 1"},
        {"id": f"{slug}_002", "name": f"{domain_name} Item 2"},
        {"id": f"{slug}_003", "name": f"{domain_name} Item 3"},
        {"id": f"{slug}_004", "name": f"{domain_name} Item 4"},
        {"id": f"{slug}_005", "name": f"{domain_name} Item 5"},
    ]
    data_declarations.append(f'ITEMS = {json.dumps(domain_items, indent=4)}')

    # Build functions JSON
    functions_str = json.dumps(functions, indent=4)

    # Get function names for templates
    primary_func_name = primary_func["name"] if primary_func else f"get_{slug}"
    detail_func_name = detail_func["name"] if detail_func else f"get_{slug}_details"

    # Get the main parameter names
    primary_param = primary_categories[0] if primary_categories else "category"
    primary_var = primary_categories[1] if primary_categories else "CATEGORIES"

    # Find the ID parameter for detail function
    id_param = "item_id"
    if detail_func:
        for param_name in detail_func.get("params", {}).keys():
            if "id" in param_name.lower():
                id_param = param_name
                break

    data_section = "\n".join(data_declarations)

    return f'''"""
{domain_name} function calling test case generator.

Supports all standard categories: simple, parallel, multiple, multi_turn, agentic.
"""

import random
from typing import Any, Dict, List


# ============================================================================
# DOMAIN-SPECIFIC SAMPLE DATA
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
# QUERY TEMPLATES - Natural language queries with domain-specific values
# ============================================================================

SIMPLE_TEMPLATES = [
    ("Show me the {{category}} options", "{primary_func_name}({primary_param}='{{category}}')"),
    ("What {{category}} do you have?", "{primary_func_name}({primary_param}='{{category}}')"),
    ("I'd like to see {{category}}", "{primary_func_name}({primary_param}='{{category}}')"),
    ("List all {{category}}", "{primary_func_name}({primary_param}='{{category}}')"),
    ("Tell me about {{item_name}}", "{detail_func_name}({id_param}='{{item_id}}')"),
    ("What are the details for {{item_name}}?", "{detail_func_name}({id_param}='{{item_id}}')"),
    ("Can you describe {{item_name}}?", "{detail_func_name}({id_param}='{{item_id}}')"),
]

PARALLEL_TEMPLATES = [
    (
        "Show me both {{cat1}} and {{cat2}}",
        ["{primary_func_name}({primary_param}='{{cat1}}')", "{primary_func_name}({primary_param}='{{cat2}}')"]
    ),
    (
        "I want to see {{cat1}} and also {{cat2}}",
        ["{primary_func_name}({primary_param}='{{cat1}}')", "{primary_func_name}({primary_param}='{{cat2}}')"]
    ),
    (
        "Show me {{category}} and tell me about {{item_name}}",
        ["{primary_func_name}({primary_param}='{{category}}')", "{detail_func_name}({id_param}='{{item_id}}')"]
    ),
]

MULTIPLE_TEMPLATES = [
    (
        "First show me {{cat1}}, then {{cat2}}, then {{cat3}}",
        [
            "{primary_func_name}({primary_param}='{{cat1}}')",
            "{primary_func_name}({primary_param}='{{cat2}}')",
            "{primary_func_name}({primary_param}='{{cat3}}')"
        ]
    ),
    (
        "Tell me about {{item1_name}}, then {{item2_name}}",
        [
            "{detail_func_name}({id_param}='{{item1_id}}')",
            "{detail_func_name}({id_param}='{{item2_id}}')"
        ]
    ),
]

MULTI_TURN_TEMPLATES = [
    {{
        "scenario": "browsing_flow",
        "turns": [
            {{
                "query": "Show me the {{category}} options",
                "expected_calls": ["{primary_func_name}({primary_param}='{{category}}')"],
            }},
            {{
                "query": "Tell me more about {{item_name}}",
                "expected_calls": ["{detail_func_name}({id_param}='{{item_id}}')"],
            }},
        ],
    }},
    {{
        "scenario": "comparison_flow",
        "turns": [
            {{
                "query": "What {{cat1}} do you have?",
                "expected_calls": ["{primary_func_name}({primary_param}='{{cat1}}')"],
            }},
            {{
                "query": "And what about {{cat2}}?",
                "expected_calls": ["{primary_func_name}({primary_param}='{{cat2}}')"],
            }},
        ],
    }},
]

AGENTIC_TEMPLATES = [
    {{
        "query": "Is {{category}} available? Just say yes or no.",
        "context": "Check if the category exists in the system.",
        "expected_response": ["yes", "no"],
        "match_mode": "any",
    }},
    {{
        "query": "Can you help me find {{item_name}}? Answer yes or no.",
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
    cats = rng.sample({primary_var}, min(3, len({primary_var})))

    return {{
        "category": rng.choice({primary_var}),
        "cat1": cats[0] if len(cats) > 0 else {primary_var}[0],
        "cat2": cats[1] if len(cats) > 1 else {primary_var}[0],
        "cat3": cats[2] if len(cats) > 2 else {primary_var}[0],
        "item_id": item["id"],
        "item_name": item["name"],
        "item1_id": items[0]["id"],
        "item1_name": items[0]["name"],
        "item2_id": items[1]["id"] if len(items) > 1 else items[0]["id"],
        "item2_name": items[1]["name"] if len(items) > 1 else items[0]["name"],
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
        enums = _extract_enum_values(functions)

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

                    # Check for ground_truth in non-agentic categories
                    if cat != "agentic" and "ground_truth" not in tc:
                        issues.append(f"WARNING: Test case missing 'ground_truth' in category '{cat}'")

                    # Validate that ground_truth uses valid parameter values
                    if "ground_truth" in tc and enums:
                        gt = tc["ground_truth"]
                        if isinstance(gt, str):
                            gt = [gt]
                        for call in gt:
                            if isinstance(call, str):
                                # Check for generic placeholder values
                                if any(generic in call for generic in ["item_1", "item_2", "option_a", "value_1"]):
                                    issues.append(f"WARNING: Ground truth may use generic values instead of domain-specific ones: {call[:50]}...")
                                    break

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
