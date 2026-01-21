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
- Templates MUST use actual parameter names from function definitions
- All required parameters MUST be provided in ground_truth
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import optional dependencies
try:
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    class Console:
        def print(self, *args, **kwargs):
            text = args[0] if args else ""
            text = re.sub(r'\[.*?\]', '', str(text))
            print(text)

    console = Console()


# Default model for domain generation
DEFAULT_CREATOR_MODEL = "gpt-5.2"
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
    """Fix common issues in AI-generated Python code."""
    code = re.sub(r':\s*true\b', ': True', code)
    code = re.sub(r':\s*false\b', ': False', code)
    code = re.sub(r':\s*null\b', ': None', code)
    code = re.sub(r',\s*true\b', ', True', code)
    code = re.sub(r',\s*false\b', ', False', code)
    code = re.sub(r',\s*null\b', ', None', code)
    return code


def _clean_functions_for_compatibility(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean function definitions for cross-provider compatibility.
    Removes array types and default values.
    """
    cleaned = []
    for func in functions:
        clean_func = {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        props = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])
        new_required = []

        for param_name, param_def in props.items():
            # Skip array types
            if param_def.get("type") == "array":
                continue

            clean_param = {
                "type": param_def.get("type", "string"),
                "description": param_def.get("description", "")
            }

            if "enum" in param_def:
                clean_param["enum"] = param_def["enum"]

            clean_func["parameters"]["properties"][param_name] = clean_param

            if param_name in required:
                new_required.append(param_name)

        clean_func["parameters"]["required"] = new_required
        cleaned.append(clean_func)

    return cleaned


def create_domain_with_ai(
    domain_name: str,
    domain_description: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_CREATOR_MODEL,
    categories: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, str]:
    """Create a new function calling domain using AI."""
    from dotenv import load_dotenv
    load_dotenv()

    from domainbench.core.config import ModelConfig, ProviderType
    from domainbench.providers import get_provider

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

    if categories is None:
        categories = SUPPORTED_CATEGORIES.copy()

    domain_slug = slugify(domain_name)

    if output_dir:
        domain_path = Path(output_dir) / domain_slug
    else:
        domain_path = FUNC_CALL_DOMAINS_DIR / domain_slug

    domain_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Generating domain: {domain_name}[/dim]")

    # Step 1: Generate function definitions
    console.print("[dim]  Generating function definitions...[/dim]")
    functions = _generate_functions(llm_provider, model, domain_name, domain_description)

    # Step 2: Clean functions for compatibility
    console.print("[dim]  Cleaning functions for compatibility...[/dim]")
    functions = _clean_functions_for_compatibility(functions)

    # Step 3: Generate domain.yaml
    console.print("[dim]  Creating domain.yaml...[/dim]")
    domain_yaml = _create_domain_yaml(domain_name, domain_description, categories, functions)
    with open(domain_path / "domain.yaml", 'w', encoding='utf-8') as f:
        f.write(domain_yaml)

    # Step 4: Generate generator.py
    console.print("[dim]  Generating test case generator...[/dim]")
    generator_code = _create_generator_from_functions(domain_name, functions)
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
    domain_slug = slugify(domain_name)

    prompt = f"""Generate 4 function definitions for a function calling benchmark domain.

Domain Name: {domain_name}
Description: {domain_description or f"A {domain_name.lower()} related API"}

**CRITICAL REQUIREMENTS:**

1. **NO ARRAY TYPES** - Do NOT use "type": "array"
2. **NO DEFAULT VALUES** - Do NOT include "default" fields
3. **USE ENUMS** for string parameters with fixed values (categories, statuses, types, etc.)
4. **SIMPLE TYPES ONLY**: string, integer, boolean

Generate exactly 4 functions:
1. A "get list" function with ONE optional enum parameter (like category)
2. A "get details" function with ONE required string ID parameter
3. A "create" function with 2 required integer parameters
4. A "process/update" function with ONE required string ID and ONE required enum parameter

Output ONLY valid JSON array:

[
  {{
    "name": "get_{domain_slug}_list",
    "description": "Get list of items, optionally filtered by category",
    "parameters": {{
      "type": "object",
      "properties": {{
        "category": {{
          "type": "string",
          "enum": ["type_a", "type_b", "type_c", "type_d", "type_e"],
          "description": "Category to filter by"
        }}
      }},
      "required": []
    }}
  }},
  {{
    "name": "get_{domain_slug}_details",
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
    "name": "create_{domain_slug}",
    "description": "Create a new item",
    "parameters": {{
      "type": "object",
      "properties": {{
        "reference_id": {{
          "type": "integer",
          "description": "Reference number"
        }},
        "quantity": {{
          "type": "integer",
          "description": "Quantity"
        }}
      }},
      "required": ["reference_id", "quantity"]
    }}
  }},
  {{
    "name": "process_{domain_slug}",
    "description": "Process or update an item",
    "parameters": {{
      "type": "object",
      "properties": {{
        "item_id": {{
          "type": "string",
          "description": "Item identifier"
        }},
        "action": {{
          "type": "string",
          "enum": ["approve", "reject", "pending"],
          "description": "Action to perform"
        }}
      }},
      "required": ["item_id", "action"]
    }}
  }}
]

Generate domain-specific function names and enum values that make sense for "{domain_name}".
Use "{domain_slug}" as the base for function names.
Output ONLY the JSON array, no other text:"""

    response = provider.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2500,
    )

    content = response.get("content", "")

    try:
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            functions = json.loads(json_match.group())
        else:
            functions = json.loads(content)
    except json.JSONDecodeError:
        functions = _create_fallback_functions(domain_name)

    return functions


def _create_fallback_functions(domain_name: str) -> List[Dict[str, Any]]:
    """Create basic fallback functions if AI generation fails."""
    slug = slugify(domain_name)
    return [
        {
            "name": f"get_{slug}_list",
            "description": f"Get list of {domain_name.lower()} items by category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["type_a", "type_b", "type_c", "type_d", "type_e"],
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
            "name": f"create_{slug}",
            "description": f"Create a new {domain_name.lower()} item",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_id": {
                        "type": "integer",
                        "description": "Reference number"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity"
                    }
                },
                "required": ["reference_id", "quantity"]
            }
        },
        {
            "name": f"process_{slug}",
            "description": f"Process a {domain_name.lower()} action",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Item identifier"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["approve", "reject", "pending"],
                        "description": "Action to perform"
                    }
                },
                "required": ["item_id", "action"]
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


def _analyze_function(func: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a function definition and extract useful info for template generation.
    """
    props = func.get("parameters", {}).get("properties", {})
    required = func.get("parameters", {}).get("required", [])

    params = []
    for param_name, param_def in props.items():
        param_info = {
            "name": param_name,
            "type": param_def.get("type", "string"),
            "required": param_name in required,
            "enum": param_def.get("enum"),
            "description": param_def.get("description", ""),
        }
        params.append(param_info)

    return {
        "name": func["name"],
        "description": func.get("description", ""),
        "params": params,
        "required_params": [p for p in params if p["required"]],
        "optional_params": [p for p in params if not p["required"]],
        "enum_params": [p for p in params if p["enum"]],
        "string_params": [p for p in params if p["type"] == "string" and not p["enum"]],
        "int_params": [p for p in params if p["type"] == "integer"],
    }


def _create_generator_from_functions(domain_name: str, functions: List[Dict[str, Any]]) -> str:
    """
    Create a generator that produces templates matching the actual function signatures.
    """
    slug = slugify(domain_name)

    # Analyze all functions
    analyzed = [_analyze_function(f) for f in functions]

    # Find functions by pattern
    list_func = None
    detail_func = None
    create_func = None
    process_func = None

    for f in analyzed:
        name_lower = f["name"].lower()
        if ("list" in name_lower or "get" in name_lower) and f["optional_params"] and f["enum_params"]:
            if list_func is None:
                list_func = f
        elif ("detail" in name_lower or "get" in name_lower) and f["string_params"] and not f["enum_params"]:
            if detail_func is None:
                detail_func = f
        elif "create" in name_lower and f["int_params"]:
            if create_func is None:
                create_func = f
        elif ("process" in name_lower or "update" in name_lower) and f["enum_params"]:
            if process_func is None:
                process_func = f

    # Fallbacks
    if list_func is None and analyzed:
        list_func = analyzed[0]
    if detail_func is None and len(analyzed) > 1:
        detail_func = analyzed[1]
    if create_func is None and len(analyzed) > 2:
        create_func = analyzed[2]
    if process_func is None and len(analyzed) > 3:
        process_func = analyzed[3]

    # Build sample data based on actual parameters
    sample_data_lines = []
    template_vars = {}

    # For list function's enum param
    if list_func and list_func["enum_params"]:
        enum_param = list_func["enum_params"][0]
        var_name = enum_param["name"].upper() + "S"
        sample_data_lines.append(f'{var_name} = {json.dumps(enum_param["enum"])}')
        template_vars["list_enum_var"] = var_name
        template_vars["list_enum_param"] = enum_param["name"]

    # For detail function's string param
    if detail_func and detail_func["string_params"]:
        id_param = detail_func["string_params"][0]
        var_name = "ITEM_IDS"
        sample_data_lines.append(f'{var_name} = ["{slug}_001", "{slug}_002", "{slug}_003", "{slug}_004", "{slug}_005"]')
        template_vars["detail_id_var"] = var_name
        template_vars["detail_id_param"] = id_param["name"]

    # For create function's int params
    if create_func and create_func["int_params"]:
        for i, int_param in enumerate(create_func["int_params"][:2]):
            var_name = int_param["name"].upper() + "S"
            sample_data_lines.append(f'{var_name} = list(range(1, 21))')
            template_vars[f"create_int_var_{i}"] = var_name
            template_vars[f"create_int_param_{i}"] = int_param["name"]

    # For process function's enum param
    if process_func and process_func["enum_params"]:
        enum_param = process_func["enum_params"][0]
        var_name = enum_param["name"].upper() + "S"
        if var_name not in [line.split(" = ")[0] for line in sample_data_lines]:
            sample_data_lines.append(f'{var_name} = {json.dumps(enum_param["enum"])}')
        template_vars["process_enum_var"] = var_name
        template_vars["process_enum_param"] = enum_param["name"]

    # For process function's string param
    if process_func and process_func["string_params"]:
        id_param = process_func["string_params"][0]
        template_vars["process_id_param"] = id_param["name"]

    sample_data = "\n".join(sample_data_lines)
    functions_json = json.dumps(functions, indent=4)

    # Build templates that use actual parameter names
    simple_templates = _build_simple_templates(list_func, detail_func, create_func, process_func, template_vars)
    parallel_templates = _build_parallel_templates(list_func, detail_func, template_vars)
    multiple_templates = _build_multiple_templates(list_func, detail_func, template_vars)
    multi_turn_templates = _build_multi_turn_templates(list_func, detail_func, template_vars)
    agentic_templates = _build_agentic_templates(template_vars)

    # Build _get_random_values function
    random_values = _build_random_values_func(template_vars)

    return f'''"""
{domain_name} function calling test case generator.

Supports all standard categories: simple, parallel, multiple, multi_turn, agentic.
"""

import random
from typing import Any, Dict, List


# ============================================================================
# DOMAIN-SPECIFIC SAMPLE DATA
# ============================================================================

{sample_data}


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
# ============================================================================

{simple_templates}

{parallel_templates}

{multiple_templates}

{multi_turn_templates}

{agentic_templates}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

{random_values}


# ============================================================================
# MAIN GENERATOR FUNCTION
# ============================================================================

def generate_test_cases(
    count: int,
    seed: int = 42,
    category: str = "simple",
) -> List[Dict[str, Any]]:
    """Generate function calling test cases."""
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
    template = rng.choice(MULTI_TURN_TEMPLATES)
    values = _get_random_values(rng)
    turns = []
    for turn_template in template["turns"]:
        query = turn_template["query"].format(**values)
        expected_calls = [call.format(**values) for call in turn_template["expected_calls"]]
        turns.append({{"query": query, "expected_calls": expected_calls}})
    return {{
        "id": f"{slug}_multi_turn_{{idx:04d}}",
        "category": "multi_turn",
        "query": turns[0]["query"],
        "functions": FUNCTIONS,
        "turns": turns,
        "ground_truth": turns[0]["expected_calls"],
    }}


def _generate_agentic(rng: random.Random, idx: int) -> Dict[str, Any]:
    template = rng.choice(AGENTIC_TEMPLATES)
    values = _get_random_values(rng)
    query = template["query"].format(**values)
    context = template.get("context", "").format(**values)
    expected_response = template["expected_response"]
    if isinstance(expected_response, list):
        expected_response = [r.format(**values) if isinstance(r, str) else r for r in expected_response]
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


def _build_simple_templates(list_func, detail_func, create_func, process_func, vars: dict) -> str:
    """Build SIMPLE_TEMPLATES using actual function signatures."""
    templates = []

    # List function templates
    if list_func and vars.get("list_enum_param"):
        func_name = list_func["name"]
        param = vars["list_enum_param"]
        templates.extend([
            f'    ("Show me the {{{param}}} options", "{func_name}({param}=\'{{{param}}}\')"),',
            f'    ("What {{{param}}} do you have?", "{func_name}({param}=\'{{{param}}}\')"),',
            f'    ("Can I see the {{{param}}} please?", "{func_name}({param}=\'{{{param}}}\')"),',
            f'    ("I\'d like to see {{{param}}}", "{func_name}({param}=\'{{{param}}}\')"),',
        ])

    # Detail function templates
    if detail_func and vars.get("detail_id_param"):
        func_name = detail_func["name"]
        param = vars["detail_id_param"]
        templates.extend([
            f'    ("Tell me about item {{{param}}}", "{func_name}({param}=\'{{{param}}}\')"),',
            f'    ("What\'s in item {{{param}}}?", "{func_name}({param}=\'{{{param}}}\')"),',
            f'    ("Can you describe item {{{param}}}?", "{func_name}({param}=\'{{{param}}}\')"),',
        ])

    # Create function templates
    if create_func and vars.get("create_int_param_0") and vars.get("create_int_param_1"):
        func_name = create_func["name"]
        p0 = vars["create_int_param_0"]
        p1 = vars["create_int_param_1"]
        templates.extend([
            f'    ("Create with {p0} {{{p0}}} and {p1} {{{p1}}}", "{func_name}({p0}={{{p0}}}, {p1}={{{p1}}})"),',
            f'    ("Start new: {p0} {{{p0}}}, {p1} {{{p1}}}", "{func_name}({p0}={{{p0}}}, {p1}={{{p1}}})"),',
        ])

    # Process function templates
    if process_func and vars.get("process_id_param") and vars.get("process_enum_param"):
        func_name = process_func["name"]
        id_p = vars["process_id_param"]
        enum_p = vars["process_enum_param"]
        templates.extend([
            f'    ("Process {{{id_p}}} with {enum_p} {{{enum_p}}}", "{func_name}({id_p}=\'{{{id_p}}}\', {enum_p}=\'{{{enum_p}}}\')"),',
            f'    ("Update {{{id_p}}} to {{{enum_p}}}", "{func_name}({id_p}=\'{{{id_p}}}\', {enum_p}=\'{{{enum_p}}}\')"),',
        ])

    return "SIMPLE_TEMPLATES = [\n" + "\n".join(templates) + "\n]"


def _build_parallel_templates(list_func, detail_func, vars: dict) -> str:
    """Build PARALLEL_TEMPLATES."""
    templates = []

    if list_func and vars.get("list_enum_param"):
        func_name = list_func["name"]
        param = vars["list_enum_param"]
        templates.append(f'''    (
        "Show me both {{{param}1}} and {{{param}2}}",
        ["{func_name}({param}='{{{param}1}}')", "{func_name}({param}='{{{param}2}}')"]
    ),''')

    if list_func and detail_func and vars.get("list_enum_param") and vars.get("detail_id_param"):
        list_name = list_func["name"]
        detail_name = detail_func["name"]
        enum_p = vars["list_enum_param"]
        id_p = vars["detail_id_param"]
        templates.append(f'''    (
        "Show me {{{enum_p}}} and tell me about item {{{id_p}}}",
        ["{list_name}({enum_p}='{{{enum_p}}}')", "{detail_name}({id_p}='{{{id_p}}}')"]
    ),''')

    return "PARALLEL_TEMPLATES = [\n" + "\n".join(templates) + "\n]"


def _build_multiple_templates(list_func, detail_func, vars: dict) -> str:
    """Build MULTIPLE_TEMPLATES."""
    templates = []

    if list_func and vars.get("list_enum_param"):
        func_name = list_func["name"]
        param = vars["list_enum_param"]
        templates.append(f'''    (
        "First show me {{{param}1}}, then {{{param}2}}, then {{{param}3}}",
        [
            "{func_name}({param}='{{{param}1}}')",
            "{func_name}({param}='{{{param}2}}')",
            "{func_name}({param}='{{{param}3}}')"
        ]
    ),''')

    if detail_func and vars.get("detail_id_param"):
        func_name = detail_func["name"]
        param = vars["detail_id_param"]
        templates.append(f'''    (
        "Tell me about item {{{param}1}}, then item {{{param}2}}",
        [
            "{func_name}({param}='{{{param}1}}')",
            "{func_name}({param}='{{{param}2}}')"
        ]
    ),''')

    return "MULTIPLE_TEMPLATES = [\n" + "\n".join(templates) + "\n]"


def _build_multi_turn_templates(list_func, detail_func, vars: dict) -> str:
    """Build MULTI_TURN_TEMPLATES."""
    templates = []

    if list_func and detail_func and vars.get("list_enum_param") and vars.get("detail_id_param"):
        list_name = list_func["name"]
        detail_name = detail_func["name"]
        enum_p = vars["list_enum_param"]
        id_p = vars["detail_id_param"]
        templates.append(f'''    {{
        "scenario": "browse_flow",
        "turns": [
            {{
                "query": "Show me the {{{enum_p}}} options",
                "expected_calls": ["{list_name}({enum_p}='{{{enum_p}}}')"],
            }},
            {{
                "query": "Tell me more about item {{{id_p}}}",
                "expected_calls": ["{detail_name}({id_p}='{{{id_p}}}')"],
            }},
        ],
    }},''')

    if list_func and vars.get("list_enum_param"):
        func_name = list_func["name"]
        param = vars["list_enum_param"]
        templates.append(f'''    {{
        "scenario": "compare_flow",
        "turns": [
            {{
                "query": "What {{{param}1}} do you have?",
                "expected_calls": ["{func_name}({param}='{{{param}1}}')"],
            }},
            {{
                "query": "And what about {{{param}2}}?",
                "expected_calls": ["{func_name}({param}='{{{param}2}}')"],
            }},
        ],
    }},''')

    return "MULTI_TURN_TEMPLATES = [\n" + "\n".join(templates) + "\n]"


def _build_agentic_templates(vars: dict) -> str:
    """Build AGENTIC_TEMPLATES."""
    templates = []

    if vars.get("list_enum_param"):
        param = vars["list_enum_param"]
        templates.append(f'''    {{
        "query": "Is {{{param}}} a valid option? Just say yes or no.",
        "context": "Check if the value is valid.",
        "expected_response": ["yes"],
        "match_mode": "contains",
    }},''')

    if vars.get("detail_id_param"):
        param = vars["detail_id_param"]
        templates.append(f'''    {{
        "query": "Can you help me with item {{{param}}}? Answer yes or no.",
        "context": "Determine if you can assist.",
        "expected_response": ["yes", "no"],
        "match_mode": "any",
    }},''')

    return "AGENTIC_TEMPLATES = [\n" + "\n".join(templates) + "\n]"


def _build_random_values_func(vars: dict) -> str:
    """Build _get_random_values function."""
    lines = ['def _get_random_values(rng: random.Random) -> Dict[str, Any]:',
             '    """Get random values for template substitution."""']

    # Sample from list enum
    if vars.get("list_enum_var"):
        var = vars["list_enum_var"]
        param = vars["list_enum_param"]
        lines.append(f'    {param}_samples = rng.sample({var}, min(3, len({var})))')

    # Sample from item IDs
    if vars.get("detail_id_var"):
        var = vars["detail_id_var"]
        param = vars["detail_id_param"]
        lines.append(f'    {param}_samples = rng.sample({var}, min(2, len({var})))')

    lines.append('')
    lines.append('    return {')

    # Add list enum values
    if vars.get("list_enum_var") and vars.get("list_enum_param"):
        var = vars["list_enum_var"]
        param = vars["list_enum_param"]
        lines.append(f'        "{param}": rng.choice({var}),')
        lines.append(f'        "{param}1": {param}_samples[0] if len({param}_samples) > 0 else {var}[0],')
        lines.append(f'        "{param}2": {param}_samples[1] if len({param}_samples) > 1 else {var}[0],')
        lines.append(f'        "{param}3": {param}_samples[2] if len({param}_samples) > 2 else {var}[0],')

    # Add detail ID values
    if vars.get("detail_id_var") and vars.get("detail_id_param"):
        var = vars["detail_id_var"]
        param = vars["detail_id_param"]
        lines.append(f'        "{param}": rng.choice({var}),')
        lines.append(f'        "{param}1": {param}_samples[0] if len({param}_samples) > 0 else {var}[0],')
        lines.append(f'        "{param}2": {param}_samples[1] if len({param}_samples) > 1 else {var}[0],')

    # Add create int values
    if vars.get("create_int_var_0") and vars.get("create_int_param_0"):
        var = vars["create_int_var_0"]
        param = vars["create_int_param_0"]
        lines.append(f'        "{param}": rng.choice({var}),')

    if vars.get("create_int_var_1") and vars.get("create_int_param_1"):
        var = vars["create_int_var_1"]
        param = vars["create_int_param_1"]
        lines.append(f'        "{param}": rng.choice({var}),')

    # Add process enum values
    if vars.get("process_enum_var") and vars.get("process_enum_param"):
        var = vars["process_enum_var"]
        param = vars["process_enum_param"]
        lines.append(f'        "{param}": rng.choice({var}),')

    # Add process ID (reuse detail ID)
    if vars.get("process_id_param") and vars.get("detail_id_var"):
        var = vars["detail_id_var"]
        param = vars["process_id_param"]
        if param != vars.get("detail_id_param"):
            lines.append(f'        "{param}": rng.choice({var}),')

    lines.append('    }')

    return '\n'.join(lines)


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
    """Validate a generated domain."""
    import yaml

    issues = []
    required_files = ["domain.yaml", "generator.py", "__init__.py"]

    for filename in required_files:
        if not (domain_path / filename).exists():
            issues.append(f"ERROR: Missing required file: {filename}")
            return False, issues

    try:
        with open(domain_path / "domain.yaml", 'r') as f:
            config = yaml.safe_load(f)

        if "domain" not in config:
            issues.append("ERROR: domain.yaml missing 'domain' key")
            return False, issues
        if "functions" not in config["domain"]:
            issues.append("ERROR: domain.yaml missing 'functions' key")
            return False, issues

    except yaml.YAMLError as e:
        issues.append(f"ERROR: Invalid YAML: {e}")
        return False, issues

    try:
        with open(domain_path / "generator.py", 'r') as f:
            code = f.read()
        compile(code, domain_path / "generator.py", 'exec')
    except SyntaxError as e:
        issues.append(f"ERROR: Syntax error: {e}")
        return False, issues

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generator", domain_path / "generator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'generate_test_cases'):
            issues.append("ERROR: Missing generate_test_cases function")
            return False, issues

        for cat in SUPPORTED_CATEGORIES:
            try:
                test_cases = module.generate_test_cases(2, seed=42, category=cat)
                if not test_cases:
                    issues.append(f"WARNING: No test cases for '{cat}'")
            except Exception as e:
                issues.append(f"WARNING: Error generating '{cat}': {e}")

    except Exception as e:
        issues.append(f"ERROR: {e}")
        return False, issues

    is_valid = not any(issue.startswith("ERROR") for issue in issues)
    return is_valid, issues


def list_function_calling_domains() -> List[Dict[str, Any]]:
    """List all available function calling domains."""
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
