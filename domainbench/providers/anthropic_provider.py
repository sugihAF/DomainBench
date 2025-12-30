"""
Anthropic Claude provider adapter
"""

from typing import List, Dict, Any, Optional
from domainbench.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider adapter for Anthropic Claude API"""
    
    name = "anthropic"
    supported_features = ["chat_completion", "function_calling", "vision"]
    
    def __init__(self, api_key_env: Optional[str] = None):
        super().__init__(api_key_env)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Anthropic client"""
        if self._client is None:
            import anthropic
            api_key = self.get_api_key("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a chat completion request to Anthropic Claude"""
        
        # Extract system message if present
        system_prompt = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        request_kwargs = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_prompt:
            request_kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**request_kwargs)
        
        # Extract content from response
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
        
        usage = {
            "prompt_tokens": response.usage.input_tokens if response.usage else 0,
            "completion_tokens": response.usage.output_tokens if response.usage else 0,
            "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
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
        """Send a tool use request to Anthropic Claude"""
        
        # Extract system message if present
        system_prompt = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        # Convert functions to Anthropic tools format
        tools = []
        for f in functions:
            tools.append({
                "name": f.get("name"),
                "description": f.get("description", ""),
                "input_schema": f.get("parameters", {}),
            })
        
        request_kwargs = {
            "model": model,
            "messages": chat_messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        if system_prompt:
            request_kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**request_kwargs)
        
        # Extract content and tool calls
        content = ""
        tool_calls = []
        
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    import json
                    tool_calls.append({
                        "id": block.id,
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        }
                    })
        
        usage = {
            "prompt_tokens": response.usage.input_tokens if response.usage else 0,
            "completion_tokens": response.usage.output_tokens if response.usage else 0,
            "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
        }
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": usage,
            "raw": response,
        }
