"""
Cerebras provider adapter

Uses the OpenAI SDK with a custom base URL since the Cerebras API is
OpenAI-compatible. No additional dependency is required.

Supported models:
- llama3.1-8b: 8B parameter model, ~2200 tok/s
- gpt-oss-120b: 120B parameter reasoning model, ~3000 tok/s
- qwen-3-235b-a22b-instruct-2507: 235B parameter preview model, ~1400 tok/s
- zai-glm-4.7: 355B parameter preview model, ~1000 tok/s

Notes:
- Uses max_completion_tokens (not max_tokens)
- Temperature range: 0 to 1.5
- gpt-oss-120b supports reasoning_effort ("low", "medium", "high")
- zai-glm-4.7 supports clear_thinking (boolean)
- tools and response_format cannot be used together in the same request
- No vision support
"""

from typing import List, Dict, Any, Optional
from domainbench.providers.base import BaseProvider

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Models that support reasoning parameters
REASONING_MODELS = ["gpt-oss-120b"]
CLEAR_THINKING_MODELS = ["zai-glm-4.7"]


class CerebrasProvider(BaseProvider):
    """Provider adapter for Cerebras Inference API

    Uses the OpenAI SDK with a custom base URL since the Cerebras API
    is OpenAI-compatible. Supports chat completion and function calling.
    """

    name = "cerebras"
    supported_features = ["chat_completion", "function_calling"]

    def __init__(self, api_key_env: Optional[str] = None):
        super().__init__(api_key_env)
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client pointed at Cerebras API"""
        if self._client is None:
            from openai import OpenAI
            api_key = self.get_api_key("CEREBRAS_API_KEY")
            self._client = OpenAI(
                api_key=api_key,
                base_url=CEREBRAS_BASE_URL,
            )
        return self._client

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a chat completion request to Cerebras

        Args:
            model: Model identifier (e.g., llama3.1-8b, gpt-oss-120b)
            messages: List of message dicts with role and content
            temperature: Sampling temperature (0.0 to 1.5)
            max_tokens: Maximum tokens in response
            reasoning_effort: For gpt-oss-120b: "low", "medium", "high"
            **kwargs: Additional parameters passed to the API

        Returns:
            Dict with content, usage, and raw response
        """
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_tokens

        if reasoning_effort is not None and model in REASONING_MODELS:
            request_kwargs["reasoning_effort"] = reasoning_effort

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
        """Send a function calling request to Cerebras

        Args:
            model: Model identifier
            messages: List of message dicts with role and content
            functions: List of function definitions
            temperature: Sampling temperature (0.0 to 1.5)
            **kwargs: Additional parameters passed to the API

        Returns:
            Dict with content, tool_calls, usage, and raw response
        """
        tools = [{"type": "function", "function": f} for f in functions]

        request_kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
        }

        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)

        message = response.choices[0].message
        content = message.content or ""

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
