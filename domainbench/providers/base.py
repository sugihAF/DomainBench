"""
Base provider interface for LLM APIs
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseProvider(ABC):
    """
    Abstract base class for LLM provider adapters.
    
    Each provider must implement the core methods for different capabilities.
    """
    
    name: str = "base"
    supported_features: List[str] = ["chat_completion"]
    
    def __init__(self, api_key_env: Optional[str] = None):
        """
        Initialize the provider.
        
        Args:
            api_key_env: Environment variable name for API key (optional, uses default if not set)
        """
        self.api_key_env = api_key_env
        self._client = None
    
    def get_api_key(self, default_env: str) -> str:
        """Get API key from environment variable"""
        env_var = self.api_key_env or default_env
        key = os.environ.get(env_var, "").strip()
        if not key:
            raise ValueError(f"Missing required environment variable: {env_var}")
        return key
    
    @abstractmethod
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific options
            
        Returns:
            Dict with 'content' (str), 'usage' (dict), and 'raw' (original response)
        """
        pass
    
    def function_call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a function calling request.
        
        Args:
            model: Model identifier
            messages: List of message dicts
            functions: List of function definitions
            temperature: Sampling temperature
            **kwargs: Additional options
            
        Returns:
            Dict with function call info or text response
        """
        raise NotImplementedError(f"{self.name} does not support function calling")
    
    def structured_output(
        self,
        model: str,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Request structured JSON output conforming to a schema.
        
        Args:
            model: Model identifier
            messages: List of message dicts
            schema: JSON schema for expected output
            temperature: Sampling temperature
            **kwargs: Additional options
            
        Returns:
            Dict with parsed JSON output
        """
        raise NotImplementedError(f"{self.name} does not support structured output")
    
    def vision(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        images: List[str],
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a vision request with images.
        
        Args:
            model: Model identifier
            messages: List of message dicts (may include image references)
            images: List of image URLs or base64 data
            temperature: Sampling temperature
            **kwargs: Additional options
            
        Returns:
            Dict with text response
        """
        raise NotImplementedError(f"{self.name} does not support vision")
    
    def supports(self, feature: str) -> bool:
        """Check if provider supports a specific feature"""
        return feature in self.supported_features
