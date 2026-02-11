"""
OpenAI provider adapter
Based on the OpenAIChat class from waiterbench.py

Supported models include:
- GPT-4 series: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini
- GPT-4.1 series: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
- GPT-5 series: gpt-5, gpt-5-mini, gpt-5.2, gpt-5.2-mini
- GPT-5.2 variants: gpt-5.2-chat-latest, gpt-5.2-codex, gpt-5.2-pro,
                    gpt-5.2-2025-12-11, gpt-5.2-pro-2025-12-11
- O-series (reasoning): o1, o1-mini, o1-pro, o3, o3-mini, o4-mini

GPT-5.x models support additional parameters:
- reasoning: {"effort": "none"|"low"|"medium"|"high"|"xhigh"} - Controls reasoning depth
- text: {"verbosity": "low"|"medium"|"high"} - Controls output verbosity

Note: Reasoning models don't support custom temperature values:
- All GPT-5 series models (gpt-5, gpt-5.2, gpt-5.2-codex, gpt-5.2-pro, etc.)
- O-series models (o1, o3, o4)
The provider automatically omits the temperature parameter for these models.
"""

from typing import List, Dict, Any, Optional
from domainbench.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI API
    
    Automatically handles parameter differences between model generations:
    - Newer models (gpt-4.1+, gpt-5+, o1, o3, o4) use max_completion_tokens
    - Older models use max_tokens
    """
    
    name = "openai"
    supported_features = ["chat_completion", "function_calling", "structured_output", "vision"]
    
    def __init__(self, api_key_env: Optional[str] = None):
        super().__init__(api_key_env)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            from openai import OpenAI
            api_key = self.get_api_key("OPENAI_API_KEY")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    # Model prefixes that don't support custom temperature (reasoning models)
    NO_TEMPERATURE_PREFIXES = [
        "o1", "o3", "o4",  # O-series reasoning models
        "gpt-5",  # All GPT-5 series (gpt-5, gpt-5.2, gpt-5.2-codex, gpt-5.2-pro, etc.)
    ]

    def _supports_temperature(self, model: str) -> bool:
        """Check if model supports custom temperature values.

        Reasoning models (O-series, GPT-5 series) don't support custom temperature.
        """
        for prefix in self.NO_TEMPERATURE_PREFIXES:
            if model.startswith(prefix):
                return False
        return True

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenAI

        Args:
            model: Model identifier (e.g., gpt-4o, gpt-5.2-chat-latest)
            messages: List of message dicts with role and content
            temperature: Sampling temperature (0.0 to 2.0) - ignored for some models
            max_tokens: Maximum tokens in response
            reasoning_effort: For GPT-5.2 models: "none", "low", "medium", "high", "xhigh"
            verbosity: For GPT-5.2 models: "low", "medium", "high"
            **kwargs: Additional parameters passed to the API

        Returns:
            Dict with content, usage, and raw response
        """

        request_kwargs = {
            "model": model,
            "messages": messages,
        }

        # Some models don't support custom temperature
        if self._supports_temperature(model):
            request_kwargs["temperature"] = temperature

        if max_tokens is not None:
            # Newer models (gpt-4.1+, gpt-5+, o-series) use max_completion_tokens
            # Older models (gpt-4, gpt-4-turbo, gpt-4o) use max_tokens
            newer_model_prefixes = ["gpt-4.1", "gpt-5", "o1", "o3", "o4"]
            if any(model.startswith(prefix) for prefix in newer_model_prefixes):
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens

        # GPT-5.2 specific parameters
        if reasoning_effort is not None and model.startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": reasoning_effort}

        if verbosity is not None and model.startswith("gpt-5"):
            request_kwargs["text"] = {"verbosity": verbosity}

        # Add any additional kwargs
        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        
        content = response.choices[0].message.content or ""
        
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return {
            "content": content,
            "usage": usage,
            "raw": response,
        }
    
    def function_call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a function calling request to OpenAI"""

        # Convert functions to OpenAI tools format
        tools = [{"type": "function", "function": f} for f in functions]

        request_kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }

        # Some models don't support custom temperature
        if self._supports_temperature(model):
            request_kwargs["temperature"] = temperature

        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        
        message = response.choices[0].message
        content = message.content or ""
        
        # Extract tool calls if present
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })
        
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": usage,
            "raw": response,
        }
    
    def structured_output(
        self,
        model: str,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Request structured JSON output from OpenAI"""

        request_kwargs = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        # Some models don't support custom temperature
        if self._supports_temperature(model):
            request_kwargs["temperature"] = temperature

        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        
        content = response.choices[0].message.content or "{}"
        
        import json
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return {
            "content": content,
            "parsed": parsed,
            "usage": usage,
            "raw": response,
        }
    
    def vision(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        images: List[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a vision request with images.

        OpenAI's chat completion API natively handles vision - messages
        can include image_url content directly. This method also supports
        passing images as a separate list.

        Args:
            model: Model identifier (e.g., gpt-4o, gpt-5.2, gpt-5.2-codex)
            messages: List of message dicts (may include image_url content)
            images: Optional list of image URLs or base64 data to append
            temperature: Sampling temperature (ignored for GPT-5/O-series)
            max_tokens: Maximum tokens in response
            **kwargs: Additional options (reasoning_effort, verbosity for GPT-5.2)

        Returns:
            Dict with text response
        """
        # If images are passed separately, add them to the last user message
        if images:
            # Find the last user message or create one
            user_msg_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    user_msg_idx = i
                    break
            
            if user_msg_idx is not None:
                msg = messages[user_msg_idx]
                content = msg.get("content", "")
                
                # Convert to list format if it's a string
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                elif not isinstance(content, list):
                    content = [{"type": "text", "text": str(content)}]
                
                # Add images
                for image in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image}
                    })
                
                messages[user_msg_idx]["content"] = content
        
        # Use chat_completion which handles vision messages natively
        return self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
