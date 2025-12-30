"""
Domain loader - Load domains from built-in templates or user-defined paths
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


# Built-in domains directory
BUILTIN_DOMAINS_DIR = Path(__file__).parent / "builtin"


def list_builtin_domains() -> List[Dict[str, str]]:
    """
    List all available built-in domains.
    
    Returns:
        List of dicts with 'name' and 'description'
    """
    domains = []
    
    if not BUILTIN_DOMAINS_DIR.exists():
        return domains
    
    for domain_dir in BUILTIN_DOMAINS_DIR.iterdir():
        if domain_dir.is_dir():
            config_file = domain_dir / "domain.yaml"
            if config_file.exists():
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if 'domain' in data:
                    data = data['domain']
                
                domains.append({
                    "name": domain_dir.name,
                    "description": data.get("description", ""),
                })
    
    return domains


def load_domain(domain_name_or_path: str):
    """
    Load a domain configuration.
    
    Args:
        domain_name_or_path: Either a built-in domain name or path to domain.yaml
        
    Returns:
        DomainConfig instance
    """
    # Import here to avoid circular import
    from domainbench.core.config import DomainConfig
    
    # Check if it's a path to a file
    path = Path(domain_name_or_path)
    if path.exists():
        if path.is_file():
            return _load_domain_from_yaml(path)
        elif path.is_dir():
            # Look for domain.yaml in the directory
            config_file = path / "domain.yaml"
            if config_file.exists():
                return _load_domain_from_yaml(config_file)
    
    # Try to load as built-in domain
    builtin_path = BUILTIN_DOMAINS_DIR / domain_name_or_path / "domain.yaml"
    if builtin_path.exists():
        return _load_domain_from_yaml(builtin_path)
    
    raise ValueError(f"Domain not found: {domain_name_or_path}")


def _load_domain_from_yaml(path: Path):
    """Load domain config from YAML file"""
    import yaml
    from domainbench.core.config import DomainConfig
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Handle nested 'domain' key
    if 'domain' in data:
        data = data['domain']
    
    # Map 'context.system_prompt' to 'system_prompt' if needed
    if 'context' in data and 'system_prompt' in data['context']:
        data['system_prompt'] = data['context']['system_prompt']
        del data['context']
    
    return DomainConfig(**data)


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load a benchmark dataset from JSONL file.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of test case dictionaries
    """
    items = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    
    return items


def save_dataset(items: List[Dict[str, Any]], path: str) -> None:
    """
    Save a benchmark dataset to JSONL file.
    
    Args:
        items: List of test case dictionaries
        path: Output path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
