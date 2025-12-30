"""
Domain loading and management
"""

from domainbench.domains.loader import load_domain, load_dataset, list_builtin_domains
from domainbench.domains.schema import DomainSchema

# Note: creator module is imported lazily to avoid circular imports
# Use: from domainbench.domains.creator import create_domain_with_ai

__all__ = [
    "load_domain",
    "load_dataset",
    "list_builtin_domains",
    "DomainSchema",
]
