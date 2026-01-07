"""
Capability plugins for different benchmark types
"""

from domainbench.capabilities.base import BaseCapability
from domainbench.capabilities.chat_completion import ChatCompletionCapability
from domainbench.capabilities.ocr import (
    OCRCapability,
    calculate_extraction_accuracy,
    compare_model_results,
    get_schema_config,
    MENU_EXTRACTION_SCHEMA,
    RECEIPT_EXTRACTION_SCHEMA,
    DOCUMENT_EXTRACTION_SCHEMA,
)


def get_capability(name: str, **kwargs) -> BaseCapability:
    """
    Factory function to get a capability by name.
    
    Args:
        name: Capability name (e.g., "chat_completion", "ocr")
        **kwargs: Additional arguments passed to capability constructor
        
    Returns:
        Initialized capability instance
    """
    capability_map = {
        "chat_completion": ChatCompletionCapability,
        "chat": ChatCompletionCapability,  # Alias
        "ocr": OCRCapability,
        "vision": OCRCapability,  # Alias
        "extraction": OCRCapability,  # Alias
    }
    
    capability_class = capability_map.get(name.lower())
    if capability_class is None:
        raise ValueError(f"Unknown capability: {name}. Available: {list(capability_map.keys())}")
    
    return capability_class(**kwargs) if kwargs else capability_class()


def list_capabilities() -> list:
    """List all available capabilities"""
    return [
        {"name": "chat_completion", "description": "Multi-turn chat conversation benchmark"},
        {"name": "ocr", "description": "Vision-based structured data extraction benchmark (OCR)"},
    ]


__all__ = [
    "BaseCapability",
    "ChatCompletionCapability",
    "OCRCapability",
    "get_capability",
    "list_capabilities",
    "calculate_extraction_accuracy",
    "compare_model_results",
    "get_schema_config",
    "MENU_EXTRACTION_SCHEMA",
    "RECEIPT_EXTRACTION_SCHEMA",
    "DOCUMENT_EXTRACTION_SCHEMA",
]
