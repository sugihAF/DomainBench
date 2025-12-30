"""
OpenAI provider adapter
Based on the OpenAIChat class from waiterbench.py
"""

from typing import List, Dict, Any, Optional
from domainbench.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI API"""
    
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
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenAI"""
        
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        
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
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            **kwargs,
        )
        
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
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            **kwargs,
        )
        
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
