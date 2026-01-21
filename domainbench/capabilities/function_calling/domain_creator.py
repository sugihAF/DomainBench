"""
AI-Powered Domain Creator for Function Calling Capability.

Uses LLM to generate new function calling benchmark domains with:
- domain.yaml (function definitions)
- generator.py (test case generator)
- __init__.py (module exports)
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

    # Step 2: Generate domain.yaml
    console.print("[dim]  Creating domain.yaml...[/dim]")
    domain_yaml = _create_domain_yaml(
        domain_name, domain_description, categories, functions
    )
    with open(domain_path / "domain.yaml", 'w', encoding='utf-8') as f:
        f.write(domain_yaml)

    # Step 3: Generate generator.py
    console.print("[dim]  Generating test case generator...[/dim]")
    generator_code = _generate_generator(
        llm_provider, model, domain_name, functions, categories
    )
    with open(domain_path / "generator.py", 'w', encoding='utf-8') as f:
        f.write(generator_code)

    # Step 4: Create __init__.py
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
    prompt = f"""Generate 3-5 function definitions for a function calling benchmark domain.

Domain Name: {domain_name}
Description: {domain_description or f"A {domain_name.lower()} related API"}

Requirements:
1. Functions should be realistic and useful for the domain
2. Each function should have clear parameters with types
3. Include a mix of required and optional parameters
4. Use appropriate parameter types (string, integer, boolean, array, enum)
5. Functions should work together for parallel and multi-call scenarios

Output ONLY valid JSON array with no markdown formatting. Each function object should have:
- "name": snake_case function name
- "description": what the function does
- "parameters": JSON Schema object with properties and required fields

Example format:
[
  {{
    "name": "example_function",
    "description": "Does something useful",
    "parameters": {{
      "type": "object",
      "properties": {{
        "param1": {{"type": "string", "description": "A string parameter"}},
        "param2": {{"type": "integer", "description": "An integer parameter"}}
      }},
      "required": ["param1"]
    }}
  }}
]

Generate the functions now:"""

    response = provider.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
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
        # Fallback: create a basic function
        functions = [
            {
                "name": f"{slugify(domain_name)}_action",
                "description": f"Perform an action in the {domain_name} domain",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The action to perform"
                        }
                    },
                    "required": ["action"]
                }
            }
        ]

    return functions


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
) -> str:
    """Generate test case generator code using AI."""
    functions_json = json.dumps(functions, indent=2)

    prompt = f"""Generate a Python test case generator for a function calling benchmark.

Domain: {domain_name}
Categories to support: simple, parallel, multiple, multi_turn, agentic (ALL 5 required)

Functions available:
{functions_json}

Generate a complete Python module that supports ALL 5 categories:

1. **simple**: Single function call
2. **parallel**: Multiple independent function calls (order doesn't matter)
3. **multiple**: Same function called multiple times (order matters)
4. **multi_turn**: Sequential conversation with multiple turns
5. **agentic**: Complex tasks with text response validation

Required structure:
```python
\"\"\"
{domain_name} function calling test case generator.
Supports all standard categories: simple, parallel, multiple, multi_turn, agentic.
\"\"\"

import random
from typing import Any, Dict, List

# Sample data relevant to the domain
SAMPLE_DATA = [...]

# Function definitions
FUNCTIONS = [...]

# All supported categories
SUPPORTED_CATEGORIES = ["simple", "parallel", "multiple", "multi_turn", "agentic"]

# Templates for each category
SIMPLE_TEMPLATES = [
    ("query {{var}}", "function(param='{{var}}')")
]

PARALLEL_TEMPLATES = [
    ("query {{var1}} and {{var2}}", ["function(param='{{var1}}')", "function(param='{{var2}}')"])
]

MULTIPLE_TEMPLATES = [
    ("do three times {{var}}", ["function(param='{{var}}')", "function(param='{{var}}')", "function(param='{{var}}')"])
]

MULTI_TURN_TEMPLATES = [
    {{
        "scenario": "name",
        "turns": [
            {{"query": "first query {{var}}", "expected_calls": ["function(param='{{var}}')"]}},
            {{"query": "follow up", "expected_calls": ["another_function()"]}}
        ]
    }}
]

AGENTIC_TEMPLATES = [
    {{
        "query": "question about {{var}}",
        "context": "You have access to functions.",
        "expected_response": ["yes", "no"],
        "match_mode": "any"
    }}
]

def generate_test_cases(count: int, seed: int = 42, category: str = "simple") -> List[Dict[str, Any]]:
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

def _generate_simple(rng, idx): ...
def _generate_parallel(rng, idx): ...
def _generate_multiple(rng, idx): ...
def _generate_multi_turn(rng, idx): ...
def _generate_agentic(rng, idx): ...
```

IMPORTANT:
- ALL 5 categories must be implemented
- Use SUPPORTED_CATEGORIES constant
- Raise ValueError for unknown categories
- Each _generate_* function must return proper test case dict

Generate the complete generator code:"""

    response = provider.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=6000,
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
            # Fallback to a basic generator
            code = _create_fallback_generator(domain_name, functions)

    return code


def _create_fallback_generator(
    domain_name: str,
    functions: List[Dict[str, Any]],
) -> str:
    """Create a basic fallback generator if AI generation fails. Supports all 5 categories."""
    slug = slugify(domain_name)
    func_names = [f["name"] for f in functions]
    first_func = functions[0] if functions else {"name": "action", "parameters": {"properties": {}, "required": []}}
    second_func = functions[1] if len(functions) > 1 else first_func

    # Get first required param
    first_param = ""
    if first_func.get("parameters", {}).get("required"):
        first_param = first_func["parameters"]["required"][0]
    elif first_func.get("parameters", {}).get("properties"):
        first_param = list(first_func["parameters"]["properties"].keys())[0]

    functions_str = json.dumps(functions, indent=4)
    func_name = first_func["name"]
    func_name_readable = first_func["name"].replace("_", " ")
    second_func_name = second_func["name"]

    return f'''"""
{domain_name} function calling test case generator.

Supports all standard categories: simple, parallel, multiple, multi_turn, agentic.
"""

import random
from typing import Any, Dict, List


# Sample data
SAMPLE_VALUES = [
    "item_1", "item_2", "item_3", "item_4", "item_5",
    "option_a", "option_b", "option_c", "option_d",
]

# Function definitions
FUNCTIONS = {functions_str}

# All supported categories
SUPPORTED_CATEGORIES = ["simple", "parallel", "multiple", "multi_turn", "agentic"]

# Query templates for simple category
SIMPLE_TEMPLATES = [
    ("Please {func_name_readable} with {{value}}", "{func_name}({first_param}='{{value}}')")
]

# Query templates for parallel category
PARALLEL_TEMPLATES = [
    (
        "Please {func_name_readable} with {{value1}} and {{value2}}",
        ["{func_name}({first_param}='{{value1}}')", "{func_name}({first_param}='{{value2}}')"]
    )
]

# Query templates for multiple category
MULTIPLE_TEMPLATES = [
    (
        "Please {func_name_readable} three times with {{value}}",
        [
            "{func_name}({first_param}='{{value}}')",
            "{func_name}({first_param}='{{value}}')",
            "{func_name}({first_param}='{{value}}')"
        ]
    )
]

# Multi-turn conversation templates
MULTI_TURN_TEMPLATES = [
    {{
        "scenario": "basic_interaction",
        "turns": [
            {{
                "query": "Please {func_name_readable} with {{value}}",
                "expected_calls": ["{func_name}({first_param}='{{value}}')"],
            }},
            {{
                "query": "Now do it again with {{value2}}",
                "expected_calls": ["{func_name}({first_param}='{{value2}}')"],
            }},
        ],
    }},
]

# Agentic templates - text response validation
AGENTIC_TEMPLATES = [
    {{
        "query": "Should I {func_name_readable} with {{value}}? Answer yes or no.",
        "context": "You have access to functions. Make a decision based on the request.",
        "expected_response": ["yes", "no"],
        "match_mode": "any",
    }},
    {{
        "query": "Is {{value}} a valid option for {func_name_readable}? Answer yes or no.",
        "context": "Check if the value is appropriate.",
        "expected_response": ["yes", "no"],
        "match_mode": "any",
    }},
]


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
    value = rng.choice(SAMPLE_VALUES)

    query = template.format(value=value)
    ground_truth = gt_template.format(value=value)

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
    values = rng.sample(SAMPLE_VALUES, 2)

    query = template.format(value1=values[0], value2=values[1])
    ground_truth = [gt.format(value1=values[0], value2=values[1]) for gt in gt_templates]

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
    value = rng.choice(SAMPLE_VALUES)

    query = template.format(value=value)
    ground_truth = [gt.format(value=value) for gt in gt_templates]

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
    values = rng.sample(SAMPLE_VALUES, 2)
    value, value2 = values

    turns = []
    for turn_template in template["turns"]:
        query = turn_template["query"].format(value=value, value2=value2)
        expected_calls = [
            call.format(value=value, value2=value2)
            for call in turn_template["expected_calls"]
        ]
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
    value = rng.choice(SAMPLE_VALUES)

    query = template["query"].format(value=value)
    context = template.get("context", "").format(value=value)

    expected_response = template["expected_response"]
    if isinstance(expected_response, list):
        expected_response = [resp.format(value=value) for resp in expected_response]
    else:
        expected_response = expected_response.format(value=value)

    return {{
        "id": f"{slug}_agentic_{{idx:04d}}",
        "category": "agentic",
        "query": query,
        "context": context,
        "functions": FUNCTIONS,
        "expected_response": expected_response,
        "match_mode": template.get("match_mode", "contains"),
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


def validate_generated_domain(domain_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate a generated domain.

    Checks:
    1. Required files exist
    2. domain.yaml is valid YAML
    3. generator.py has generate_test_cases function
    4. generator.py produces valid test cases

    Args:
        domain_path: Path to domain directory

    Returns:
        Tuple of (is_valid, error_message)
    """
    import yaml

    # Check required files
    required_files = ["domain.yaml", "generator.py", "__init__.py"]
    for filename in required_files:
        if not (domain_path / filename).exists():
            return False, f"Missing required file: {filename}"

    # Validate domain.yaml
    try:
        with open(domain_path / "domain.yaml", 'r') as f:
            config = yaml.safe_load(f)

        if "domain" not in config:
            return False, "domain.yaml missing 'domain' key"
        if "functions" not in config["domain"]:
            return False, "domain.yaml missing 'functions' key"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML in domain.yaml: {e}"

    # Validate generator.py syntax
    try:
        with open(domain_path / "generator.py", 'r') as f:
            code = f.read()
        compile(code, domain_path / "generator.py", 'exec')
    except SyntaxError as e:
        return False, f"Syntax error in generator.py: {e}"

    # Try to import and run generator
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generator", domain_path / "generator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'generate_test_cases'):
            return False, "generator.py missing generate_test_cases function"

        # Try generating a few test cases
        test_cases = module.generate_test_cases(3, seed=42)
        if not test_cases:
            return False, "generate_test_cases returned empty list"

        # Validate test case structure
        required_fields = ["id", "category", "query", "functions", "ground_truth"]
        for tc in test_cases:
            for field in required_fields:
                if field not in tc:
                    return False, f"Test case missing required field: {field}"

    except Exception as e:
        return False, f"Error testing generator: {e}"

    return True, None


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
