"""
Google Gemini provider adapter
Based on the GeminiChat class from waiterbench.py

Currently available stable models (as of January 2026):
- Gemini 2.5: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
- Gemini 2.0: gemini-2.0-flash, gemini-2.0-flash-lite
- Gemini 1.5: gemini-1.5-pro, gemini-1.5-flash

Gemini 3 models (currently in preview, requires -preview suffix):
- gemini-3-pro-preview
- gemini-3-flash-preview

Model aliases (auto-updated to latest):
- gemini-flash-latest
- gemini-pro-latest

Note: Preview models may have stricter rate limits and require the -preview suffix.
For production use, prefer stable models like gemini-2.5-pro or gemini-2.0-flash.
"""

import base64
from typing import List, Dict, Any, Optional
from domainbench.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Gemini API
    
    Supports Gemini 1.5, 2.0, 2.5, and 3.x (preview) models.
    """
    
    name = "gemini"
    supported_features = ["chat_completion", "function_calling", "vision"]
    
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
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Gemini.
        
        Handles both text-only and multimodal (vision) messages.
        For vision, messages should have content as a list with image_url items.
        """
        from google.genai import types
        
        # Check if this is a vision request (messages with image content)
        has_images = self._contains_images(messages)
        
        if has_images:
            return self._vision_completion(model, messages, temperature, max_tokens, **kwargs)
        
        # Convert messages to transcript format for Gemini (text-only)
        # SYSTEM: ...
        # USER: ...
        # ASSISTANT: ...
        lines = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Handle list content (extract text parts)
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                lines.append(f"{role}: {' '.join(text_parts)}")
        
        prompt = "\n".join(lines)
        
        # Generate content
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
            )
        except Exception as e:
            error_msg = str(e)
            # Provide helpful message for common errors
            if "404" in error_msg and "NOT_FOUND" in error_msg:
                helpful_hint = "\n\nHint: For Gemini 3 models, use 'gemini-3-pro-preview' or 'gemini-3-flash-preview' (with -preview suffix).\nFor stable models, use 'gemini-2.5-pro', 'gemini-2.5-flash', or 'gemini-2.0-flash'."
                raise RuntimeError(f"Gemini API error: {e}{helpful_hint}")
            raise RuntimeError(f"Gemini API error: {e}")
        
        # Extract text from response
        text = getattr(response, "text", None)
        if text is None:
            text = str(response)
        
        # Extract usage metadata
        usage = self._extract_usage(response)
        
        return {
            "content": text,
            "usage": usage,
            "raw": response,
        }
    
    def _contains_images(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if messages contain image content."""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False
    
    def _vision_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle vision requests with images."""
        from google.genai import types
        
        # Build content parts
        parts = []
        system_instruction = None
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Handle system prompt
            if role == "system":
                if isinstance(content, str):
                    system_instruction = content
                continue
            
            # Handle user/assistant messages
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append(types.Part.from_text(text=item))
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item.get("text", "")))
                        elif item.get("type") == "image_url":
                            image_data = self._extract_image_data(item.get("image_url", {}))
                            if image_data:
                                parts.append(types.Part.from_bytes(
                                    data=image_data["bytes"],
                                    mime_type=image_data["mime_type"]
                                ))
        
        # Create content
        contents = [types.Content(role="user", parts=parts)]
        
        # Build generation config
        gen_config = {"temperature": temperature}
        if max_tokens:
            gen_config["max_output_tokens"] = max_tokens
        
        try:
            # Generate with system instruction if provided
            if system_instruction:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        **gen_config
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**gen_config),
                )
        except Exception as e:
            raise RuntimeError(f"Gemini vision API error: {e}")
        
        # Extract text from response
        text = getattr(response, "text", None)
        if text is None:
            text = str(response)
        
        usage = self._extract_usage(response)
        
        return {
            "content": text,
            "usage": usage,
            "raw": response,
        }
    
    def _extract_image_data(self, image_url: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract image bytes and mime type from image_url dict."""
        url = image_url.get("url", "")
        
        # Handle base64 data URLs
        if url.startswith("data:"):
            try:
                # Parse data URL: data:image/png;base64,<data>
                header, data = url.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]
                image_bytes = base64.b64decode(data)
                return {"bytes": image_bytes, "mime_type": mime_type}
            except Exception:
                return None
        
        # For HTTP URLs, we'd need to fetch them (not implemented)
        # Could add requests-based fetching if needed
        return None
    
    def _extract_usage(self, response) -> Dict[str, int]:
        """Extract token usage from response."""
        usage = {}
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(meta, "prompt_token_count", 0),
                "completion_tokens": getattr(meta, "candidates_token_count", 0),
                "total_tokens": getattr(meta, "total_token_count", 0),
            }
        return usage
    
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
            messages: List of message dicts
            images: List of image URLs or base64 data strings
            temperature: Sampling temperature
            **kwargs: Additional options
            
        Returns:
            Dict with text response
        """
        from google.genai import types
        
        # Build parts
        parts = []
        system_instruction = None
        
        # Extract system prompt and text from messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                system_instruction = content if isinstance(content, str) else str(content)
            elif isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
        
        # Add images
        for image in images:
            if image.startswith("data:"):
                image_data = self._extract_image_data({"url": image})
                if image_data:
                    parts.append(types.Part.from_bytes(
                        data=image_data["bytes"],
                        mime_type=image_data["mime_type"]
                    ))
        
        # Create content
        contents = [types.Content(role="user", parts=parts)]
        
        gen_config = {"temperature": temperature}
        
        try:
            if system_instruction:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        **gen_config
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**gen_config),
                )
        except Exception as e:
            raise RuntimeError(f"Gemini vision API error: {e}")

        text = getattr(response, "text", None)
        if text is None:
            text = str(response)

        usage = self._extract_usage(response)

        return {
            "content": text,
            "usage": usage,
            "raw": response,
        }

    def function_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a function calling request to Gemini.

        Args:
            model: Model identifier
            messages: List of message dicts
            functions: List of function definitions (OpenAI format)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            **kwargs: Additional options

        Returns:
            Dict with content, tool_calls, usage, and raw response
        """
        import json
        from google.genai import types

        # Convert functions to Gemini tool format
        function_declarations = []
        for func in functions:
            # Convert OpenAI function schema to Gemini format
            params = func.get("parameters", {})

            # Build Gemini-compatible schema
            gemini_params = {}
            if params.get("type") == "object" and "properties" in params:
                gemini_params = {
                    "type": "OBJECT",
                    "properties": {},
                    "required": params.get("required", []),
                }
                for prop_name, prop_def in params.get("properties", {}).items():
                    prop_type = prop_def.get("type", "string").upper()
                    if prop_type == "INTEGER":
                        prop_type = "NUMBER"
                    gemini_params["properties"][prop_name] = {
                        "type": prop_type,
                        "description": prop_def.get("description", ""),
                    }
                    if "enum" in prop_def:
                        gemini_params["properties"][prop_name]["enum"] = prop_def["enum"]

            function_declarations.append(
                types.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=gemini_params if gemini_params else None,
                )
            )

        # Create tool with functions
        tool = types.Tool(function_declarations=function_declarations)

        # Build content parts from messages
        parts = []
        system_instruction = None

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                system_instruction = content if isinstance(content, str) else str(content)
            elif isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append(types.Part.from_text(text=item))
                    elif isinstance(item, dict) and item.get("type") == "text":
                        parts.append(types.Part.from_text(text=item.get("text", "")))

        # Create content
        contents = [types.Content(role="user", parts=parts)]

        # Build generation config
        gen_config = {
            "temperature": temperature,
        }
        if max_tokens:
            gen_config["max_output_tokens"] = max_tokens

        try:
            if system_instruction:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=[tool],
                        **gen_config,
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=[tool],
                        **gen_config,
                    ),
                )
        except Exception as e:
            raise RuntimeError(f"Gemini function call API error: {e}")

        # Extract function calls from response
        tool_calls = []
        content = ""

        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        # Check for function call
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            # Convert args to dict
                            args_dict = {}
                            if hasattr(fc, "args") and fc.args:
                                # fc.args might be a Struct or dict-like object
                                if hasattr(fc.args, "items"):
                                    args_dict = dict(fc.args.items())
                                elif isinstance(fc.args, dict):
                                    args_dict = fc.args
                                else:
                                    # Try to convert to dict
                                    try:
                                        args_dict = dict(fc.args)
                                    except (TypeError, ValueError):
                                        args_dict = {}

                            tool_calls.append({
                                "id": f"call_{len(tool_calls)}",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(args_dict),
                                },
                            })
                        # Check for text content
                        elif hasattr(part, "text") and part.text:
                            content += part.text

        usage = self._extract_usage(response)

        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": usage,
            "raw": response,
        }
