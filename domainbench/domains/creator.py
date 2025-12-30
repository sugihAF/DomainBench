"""
Domain Creator - AI-powered domain generation for DomainBench

Uses an LLM to generate domain.yaml and generator.py files based on 
a user's domain description (e.g., "doctor assistant", "banking customer service").
"""

import re
import os
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console

console = Console()

# The default model we use for domain creation
DEFAULT_CREATOR_MODEL = "gpt-5.2-2025-12-11"
DEFAULT_CREATOR_PROVIDER = "openai"

# Built-in domains directory
BUILTIN_DOMAINS_DIR = Path(__file__).parent / "builtin"


def get_domain_slug(domain_description: str) -> str:
    """Convert domain description to a valid folder name."""
    # Convert to lowercase, replace spaces with underscores
    slug = domain_description.lower().strip()
    slug = re.sub(r'[^a-z0-9\s]', '', slug)  # Remove special chars
    slug = re.sub(r'\s+', '_', slug)  # Replace spaces with underscores
    slug = slug[:50]  # Limit length
    return slug


def get_system_prompt_for_domain_creation() -> str:
    """
    Return the system prompt that instructs the AI how to create domains.
    This is our "framing" for domain generation.
    """
    return '''You are an expert at creating benchmark test case generators for LLM evaluation.

Your task is to create a complete domain definition for the DomainBench framework. Given a domain description, you will generate two files:
1. domain.yaml - Domain configuration with system prompt, personas, scenarios, and evaluation criteria
2. generator.py - Python code that generates diverse test cases

## CRITICAL REQUIREMENTS:

### For domain.yaml:
- Create a detailed system_prompt that defines the AI assistant's role
- Define 3-5 realistic personas that interact with this assistant
- Define 5-8 test scenarios with different categories and difficulties
- Define evaluation criteria with weights (must sum to 1.0)
- Include relevant functions if the domain involves actions

### For generator.py:
- Follow the EXACT structure of the restaurant_waiter generator
- Create 10-15 diverse categories covering the domain's key scenarios
- Create data pools (lists) relevant to the domain
- Create builder functions for each category that generate multi-turn conversations
- Each conversation should have 3-6 turns
- Include variability using random choices and conditional additions
- Generate realistic, challenging scenarios

## OUTPUT FORMAT:
You MUST respond with exactly two code blocks:

```yaml
# domain.yaml content here
```

```python
# generator.py content here
```

## EXAMPLE STRUCTURE FOR generator.py:

```python
"""
[Domain Name] test case generator
"""

import random
from typing import List, Dict, Any

# Categories for test scenarios
CATEGORIES = [
    "category_1",
    "category_2",
    # ... 10-15 categories
]

# Data pools for generation
DATA_POOL_1 = ["item1", "item2", "item3"]
DATA_POOL_2 = ["optionA", "optionB", "optionC"]

def pick(rng: random.Random, xs: List) -> Any:
    return rng.choice(xs)

def picks(rng: random.Random, xs: List, k: int) -> List:
    return rng.sample(xs, k)

def maybe(rng: random.Random, p: float) -> bool:
    return rng.random() < p

def scenario_id(idx: int) -> str:
    return f"tc_{idx:04d}"

# Turn builders for each category
def build_turns_category_1(rng: random.Random) -> List[str]:
    # Generate conversation turns
    t = [
        "First user message...",
        "Follow-up question...",
        "Additional context or request...",
    ]
    if maybe(rng, 0.4):
        t.append("Optional additional turn...")
    return t[:rng.randint(3, 4)]

# ... more builder functions ...

BUILDERS = {
    "category_1": build_turns_category_1,
    # ... map all categories
}

def generate_test_cases(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate n test cases for this domain."""
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []

    cat_cycle = []
    while len(cat_cycle) < n:
        cat_cycle.extend(CATEGORIES)
    cat_cycle = cat_cycle[:n]
    rng.shuffle(cat_cycle)

    for i in range(n):
        cat = cat_cycle[i]
        turns = BUILDERS[cat](rng)
        
        if len(turns) < 3:
            turns += ["Can you confirm?"]
        turns = turns[:rng.randint(3, 6)]

        items.append({
            "id": scenario_id(i + 1),
            "category": cat,
            "turns": turns,
            "meta": {},
        })

    return items
```

Remember: Generate realistic, diverse, and challenging test cases that thoroughly evaluate an AI assistant in this domain.'''


def get_user_prompt_for_domain(domain_description: str) -> str:
    """Create the user prompt for generating a specific domain."""
    return f'''Create a complete DomainBench domain for: "{domain_description}"

Requirements:
1. The AI assistant should be helpful, professional, and domain-appropriate
2. Test cases should cover common scenarios, edge cases, and challenging situations
3. Include safety-critical scenarios if applicable to the domain
4. Generate diverse and realistic multi-turn conversations
5. Evaluation criteria should reflect what matters most in this domain

Generate the domain.yaml and generator.py files now.'''


def create_domain_with_ai(
    domain_description: str,
    provider: str = DEFAULT_CREATOR_PROVIDER,
    model: str = DEFAULT_CREATOR_MODEL,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, str]:
    """
    Use AI to create a new domain based on description.
    
    Args:
        domain_description: Human description like "doctor assistant"
        provider: LLM provider to use (default: openai)
        model: Model to use (default: gpt-4o)
        output_dir: Where to save the domain (default: builtin domains)
        
    Returns:
        Tuple of (domain_path, domain_slug)
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    # Import provider
    if provider == "openai":
        from domainbench.providers.openai_provider import OpenAIProvider
        llm = OpenAIProvider()
    elif provider == "anthropic":
        from domainbench.providers.anthropic_provider import AnthropicProvider
        llm = AnthropicProvider()
    elif provider == "gemini":
        from domainbench.providers.gemini_provider import GeminiProvider
        llm = GeminiProvider()
    else:
        raise ValueError(f"Unsupported provider for domain creation: {provider}")
    
    # Build messages
    messages = [
        {"role": "system", "content": get_system_prompt_for_domain_creation()},
        {"role": "user", "content": get_user_prompt_for_domain(domain_description)},
    ]
    
    console.print(f"[dim]Calling {provider}/{model} to generate domain...[/dim]")
    
    # Call the LLM
    response = llm.chat_completion(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=8000,
    )
    
    content = response["content"]
    
    # Parse the response to extract yaml and python blocks
    domain_yaml, generator_py = parse_ai_response(content)
    
    if not domain_yaml or not generator_py:
        raise ValueError("Failed to parse AI response. Expected both yaml and python code blocks.")
    
    # Determine output directory
    domain_slug = get_domain_slug(domain_description)
    if output_dir is None:
        output_dir = BUILTIN_DOMAINS_DIR
    
    domain_path = output_dir / domain_slug
    domain_path.mkdir(parents=True, exist_ok=True)
    
    # Write files
    yaml_path = domain_path / "domain.yaml"
    py_path = domain_path / "generator.py"
    init_path = domain_path / "__init__.py"
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(domain_yaml)
    
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write(generator_py)
    
    # Create __init__.py to make it importable
    init_content = f'''"""
{domain_description.title()} domain for DomainBench
Auto-generated domain
"""

from .generator import generate_test_cases, CATEGORIES

__all__ = ["generate_test_cases", "CATEGORIES"]
'''
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    return domain_path, domain_slug


def parse_ai_response(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse AI response to extract yaml and python code blocks.
    
    Returns:
        Tuple of (domain_yaml, generator_py) or (None, None) if parsing fails
    """
    domain_yaml = None
    generator_py = None
    
    # Pattern for code blocks
    # Match ```yaml or ```yml
    yaml_pattern = r'```(?:yaml|yml)\n(.*?)```'
    python_pattern = r'```python\n(.*?)```'
    
    yaml_matches = re.findall(yaml_pattern, content, re.DOTALL)
    python_matches = re.findall(python_pattern, content, re.DOTALL)
    
    if yaml_matches:
        domain_yaml = yaml_matches[0].strip()
    
    if python_matches:
        generator_py = python_matches[0].strip()
    
    return domain_yaml, generator_py


def validate_generated_domain(domain_path: Path) -> Tuple[bool, str]:
    """
    Validate that a generated domain is correctly structured.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    yaml_path = domain_path / "domain.yaml"
    py_path = domain_path / "generator.py"
    
    # Check files exist
    if not yaml_path.exists():
        return False, "domain.yaml not found"
    
    if not py_path.exists():
        return False, "generator.py not found"
    
    # Validate YAML
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Handle nested 'domain' key (restaurant_waiter style)
        if 'domain' in data and isinstance(data['domain'], dict):
            data = data['domain']
        
        # Check for system_prompt - the most important field
        # Accept either 'system_prompt' or presence of 'name'/'description' as valid
        has_system_prompt = 'system_prompt' in data
        has_name_or_desc = 'name' in data or 'description' in data or 'domain' in data
        
        if not has_system_prompt and not has_name_or_desc:
            return False, "domain.yaml missing system_prompt or domain description"
    except Exception as e:
        return False, f"Invalid YAML: {e}"
    
    # Validate Python syntax
    try:
        with open(py_path, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, str(py_path), 'exec')
    except SyntaxError as e:
        return False, f"Python syntax error: {e}"
    
    # Try to import and check for generate_test_cases
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generator", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'generate_test_cases'):
            return False, "generator.py missing generate_test_cases function"
        
        if not hasattr(module, 'CATEGORIES'):
            return False, "generator.py missing CATEGORIES list"
            
    except Exception as e:
        return False, f"Failed to import generator: {e}"
    
    return True, ""


def list_domain_categories(domain_name: str) -> list:
    """List categories available in a domain's generator."""
    domain_path = BUILTIN_DOMAINS_DIR / domain_name
    py_path = domain_path / "generator.py"
    
    if not py_path.exists():
        return []
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generator", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'CATEGORIES'):
            return list(module.CATEGORIES)
    except Exception:
        pass
    
    return []
