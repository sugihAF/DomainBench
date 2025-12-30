"""
Google Gemini provider adapter
Based on the GeminiChat class from waiterbench.py
"""

from typing import List, Dict, Any, Optional
from domainbench.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Gemini API"""
    
    name = "gemini"
    supported_features = ["chat_completion"]
    
    def __init__(self, api_key_env: Optional[str] = None):
        super().__init__(api_key_env)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Gemini client"""
        if self._client is None:
            from google import genai
            api_key = self.get_api_key("GEMINI_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Gemini.
        
        Note: Gemini API has a different message format, so we convert
        the standard messages to a single prompt transcript.
        """
        
        # Convert messages to transcript format for Gemini
        # SYSTEM: ...
        # USER: ...
        # ASSISTANT: ...
        lines = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        
        prompt = "\n".join(lines)
        
        # Generate content
        # Note: config parameter handling varies by google-genai version
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")
        
        # Extract text from response
        text = getattr(response, "text", None)
        if text is None:
            # Fallback if response structure differs
            text = str(response)
        
        # Gemini doesn't provide token usage in the same way
        usage = {}
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(meta, "prompt_token_count", 0),
                "completion_tokens": getattr(meta, "candidates_token_count", 0),
                "total_tokens": getattr(meta, "total_token_count", 0),
            }
        
        return {
            "content": text,
            "usage": usage,
            "raw": response,
        }
