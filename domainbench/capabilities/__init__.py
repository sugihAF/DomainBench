"""
Capability plugins for different benchmark types
"""

from domainbench.capabilities.base import BaseCapability
from domainbench.capabilities.chat_completion import ChatCompletionCapability


def get_capability(name: str) -> BaseCapability:
    """
    Factory function to get a capability by name.
    
    Args:
        name: Capability name (e.g., "chat_completion")
        
    Returns:
        Initialized capability instance
    """
    capability_map = {
        "chat_completion": ChatCompletionCapability,
        "chat": ChatCompletionCapability,  # Alias
    }
    
    capability_class = capability_map.get(name.lower())
    if capability_class is None:
        raise ValueError(f"Unknown capability: {name}. Available: {list(capability_map.keys())}")
    
    return capability_class()


def list_capabilities() -> list:
    """List all available capabilities"""
    return [
        {"name": "chat_completion", "description": "Multi-turn chat conversation benchmark"},
    ]


__all__ = [
    "BaseCapability",
    "ChatCompletionCapability",
    "get_capability",
    "list_capabilities",
]
