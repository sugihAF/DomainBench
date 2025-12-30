"""
Provider adapters for LLM APIs
"""

from domainbench.providers.base import BaseProvider
from domainbench.providers.openai_provider import OpenAIProvider
from domainbench.providers.gemini_provider import GeminiProvider
from domainbench.providers.anthropic_provider import AnthropicProvider

from domainbench.core.config import ModelConfig, ProviderType


def get_provider(config: ModelConfig) -> BaseProvider:
    """
    Factory function to get the appropriate provider for a model config.
    
    Args:
        config: ModelConfig with provider type and settings
        
    Returns:
        Initialized provider instance
    """
    provider_map = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.GEMINI: GeminiProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
    }
    
    provider_class = provider_map.get(config.provider)
    if provider_class is None:
        raise ValueError(f"Unsupported provider: {config.provider}")
    
    return provider_class(api_key_env=config.api_key_env)


__all__ = [
    "BaseProvider",
    "OpenAIProvider", 
    "GeminiProvider",
    "AnthropicProvider",
    "get_provider",
]
