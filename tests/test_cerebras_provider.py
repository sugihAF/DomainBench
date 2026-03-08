"""
Tests for Cerebras provider implementation.

Tests:
1. Provider registration and factory
2. Chat completion with all 4 models
3. Function calling with supported models
4. CLI model parsing (cerebras/model format)
5. Integration with voice text pipeline patterns
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

# ── Unit tests (no API key needed) ──────────────────────────────────────────


def test_provider_type_enum():
    """CEREBRAS exists in the ProviderType enum."""
    from domainbench.core.config import ProviderType
    assert ProviderType.CEREBRAS == "cerebras"
    assert ProviderType("cerebras") == ProviderType.CEREBRAS


def test_provider_factory_registration():
    """get_provider returns CerebrasProvider for CEREBRAS type."""
    from domainbench.core.config import ProviderType, ModelConfig
    from domainbench.providers import get_provider, CerebrasProvider

    config = ModelConfig(
        provider=ProviderType.CEREBRAS,
        model="llama3.1-8b",
    )
    # Patch env to avoid needing real key for factory test
    with patch.dict(os.environ, {"CEREBRAS_API_KEY": "test-key"}):
        provider = get_provider(config)
    assert isinstance(provider, CerebrasProvider)
    assert provider.name == "cerebras"


def test_supported_features():
    """Cerebras supports chat_completion and function_calling."""
    from domainbench.providers.cerebras_provider import CerebrasProvider
    p = CerebrasProvider()
    assert p.supports("chat_completion")
    assert p.supports("function_calling")
    assert not p.supports("vision")
    assert not p.supports("structured_output")


def test_display_name_format():
    """ModelConfig.display_name renders as cerebras/model."""
    from domainbench.core.config import ProviderType, ModelConfig
    config = ModelConfig(provider=ProviderType.CEREBRAS, model="gpt-oss-120b")
    assert config.display_name == "cerebras/gpt-oss-120b"


def test_cli_model_parsing():
    """CLI-style model string 'cerebras/llama3.1-8b' parses correctly."""
    from domainbench.core.config import ProviderType
    model_str = "cerebras/llama3.1-8b"
    parts = model_str.split("/", 1)
    assert len(parts) == 2
    provider_str, model_name = parts
    provider_type = ProviderType(provider_str.lower())
    assert provider_type == ProviderType.CEREBRAS
    assert model_name == "llama3.1-8b"


def test_all_model_names_parse():
    """All Cerebras model names parse correctly with cerebras/ prefix."""
    from domainbench.core.config import ProviderType, ModelConfig
    models = [
        "llama3.1-8b",
        "gpt-oss-120b",
        "qwen-3-235b-a22b-instruct-2507",
        "zai-glm-4.7",
    ]
    for model_name in models:
        config = ModelConfig(provider=ProviderType.CEREBRAS, model=model_name)
        assert config.display_name == f"cerebras/{model_name}"


def test_base_url():
    """Cerebras provider uses the correct base URL."""
    from domainbench.providers.cerebras_provider import CEREBRAS_BASE_URL
    assert CEREBRAS_BASE_URL == "https://api.cerebras.ai/v1"


def test_vision_not_implemented():
    """Vision raises NotImplementedError."""
    from domainbench.providers.cerebras_provider import CerebrasProvider
    p = CerebrasProvider()
    with pytest.raises(NotImplementedError):
        p.vision(model="llama3.1-8b", messages=[], images=[])


def test_structured_output_not_implemented():
    """Structured output raises NotImplementedError."""
    from domainbench.providers.cerebras_provider import CerebrasProvider
    p = CerebrasProvider()
    with pytest.raises(NotImplementedError):
        p.structured_output(model="llama3.1-8b", messages=[], schema={})


# ── Live API tests (require CEREBRAS_API_KEY) ───────────────────────────────

CEREBRAS_PRODUCTION_MODELS = [
    "llama3.1-8b",
    "gpt-oss-120b",
]

# Preview models may require paid tier access
CEREBRAS_PREVIEW_MODELS = [
    "qwen-3-235b-a22b-instruct-2507",
    "zai-glm-4.7",
]

CEREBRAS_ALL_MODELS = CEREBRAS_PRODUCTION_MODELS + CEREBRAS_PREVIEW_MODELS

needs_api_key = pytest.mark.skipif(
    not os.environ.get("CEREBRAS_API_KEY"),
    reason="CEREBRAS_API_KEY not set",
)


@needs_api_key
@pytest.mark.parametrize("model", CEREBRAS_PRODUCTION_MODELS)
def test_chat_completion_production_models(model):
    """Chat completion works with production Cerebras models."""
    from domainbench.providers.cerebras_provider import CerebrasProvider

    provider = CerebrasProvider()
    # gpt-oss-120b is a reasoning model that needs more tokens
    # (reasoning tokens consume the budget before content tokens)
    max_tok = 256 if model == "gpt-oss-120b" else 32
    result = provider.chat_completion(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        temperature=0.0,
        max_tokens=max_tok,
    )

    assert "content" in result
    assert isinstance(result["content"], str)
    assert "usage" in result
    assert result["usage"]["total_tokens"] > 0
    print(f"  [{model}] response: {repr(result['content'][:80])}")
    print(f"  [{model}] tokens: {result['usage']}")


@needs_api_key
@pytest.mark.parametrize("model", CEREBRAS_PREVIEW_MODELS)
def test_chat_completion_preview_models(model):
    """Chat completion with preview models (may require paid tier)."""
    from domainbench.providers.cerebras_provider import CerebrasProvider

    provider = CerebrasProvider()
    try:
        result = provider.chat_completion(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            temperature=0.0,
            max_tokens=32,
        )
        assert "content" in result
        assert "usage" in result
        print(f"  [{model}] response: {repr(result['content'][:80])}")
    except Exception as e:
        if "404" in str(e) or "not_found" in str(e) or "does not exist" in str(e):
            pytest.skip(f"{model} not accessible on this account tier: {e}")
        raise


@needs_api_key
@pytest.mark.parametrize("model", ["llama3.1-8b", "gpt-oss-120b"])
def test_function_calling(model):
    """Function calling works with Cerebras models."""
    from domainbench.providers.cerebras_provider import CerebrasProvider

    provider = CerebrasProvider()
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    result = provider.function_call(
        model=model,
        messages=[
            {"role": "user", "content": "What is the weather in San Francisco?"}
        ],
        functions=functions,
        temperature=0.0,
    )

    assert "tool_calls" in result
    assert "content" in result
    assert "usage" in result

    if result["tool_calls"]:
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert "location" in args
        print(f"  [{model}] tool_call: {tc['function']['name']}({tc['function']['arguments']})")
    else:
        # Model may choose to respond with text instead - that's valid
        print(f"  [{model}] text response (no tool call): {result['content'][:80]}")

    print(f"  [{model}] tokens: {result['usage']}")


@needs_api_key
def test_multi_turn_conversation():
    """Multi-turn conversation works (as used by chat_completion benchmark)."""
    from domainbench.providers.cerebras_provider import CerebrasProvider

    provider = CerebrasProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
        {"role": "user", "content": "What is my name?"},
    ]

    result = provider.chat_completion(
        model="llama3.1-8b",
        messages=messages,
        temperature=0.0,
        max_tokens=64,
    )

    assert "alice" in result["content"].lower()
    print(f"  Multi-turn response: {result['content'][:120]}")


@needs_api_key
def test_voice_text_pipeline_pattern():
    """Simulates how the voice engine text pipeline calls the provider.

    The voice engine does:
      1. provider.function_call() if tools present
      2. Falls back to provider.chat_completion() on NotImplementedError
      3. provider.chat_completion() if no tools
    """
    from domainbench.providers.cerebras_provider import CerebrasProvider

    provider = CerebrasProvider()
    model = "llama3.1-8b"

    # Case 1: with tools (voice engine calls function_call)
    messages = [
        {"role": "system", "content": "You are a hotel concierge. Use tools when needed."},
        {"role": "user", "content": "Book me a room for tonight."},
    ]
    tools = [
        {
            "name": "book_room",
            "description": "Book a hotel room",
            "parameters": {
                "type": "object",
                "properties": {
                    "check_in": {"type": "string", "description": "Check-in date"},
                    "nights": {"type": "integer", "description": "Number of nights"},
                },
                "required": ["check_in", "nights"],
            },
        }
    ]

    result = provider.function_call(
        model=model,
        messages=messages,
        functions=tools,
        temperature=0.0,
    )
    assert "tool_calls" in result
    print(f"  Voice+tools: tool_calls={len(result['tool_calls'])}, content={result['content'][:60]}")

    # Case 2: no tools (voice engine calls chat_completion)
    messages2 = [
        {"role": "system", "content": "You are a hotel concierge."},
        {"role": "user", "content": "What time is breakfast?"},
    ]

    result2 = provider.chat_completion(
        model=model,
        messages=messages2,
        temperature=0.0,
        max_tokens=64,
    )
    assert len(result2["content"]) > 0
    print(f"  Voice no tools: {result2['content'][:80]}")


@needs_api_key
def test_reasoning_effort_gpt_oss_120b():
    """reasoning_effort parameter works for gpt-oss-120b."""
    from domainbench.providers.cerebras_provider import CerebrasProvider

    provider = CerebrasProvider()
    result = provider.chat_completion(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.0,
        max_tokens=32,
        reasoning_effort="low",
    )

    assert "content" in result
    assert len(result["content"]) > 0
    print(f"  reasoning_effort=low: {result['content'][:80]}")


@needs_api_key
def test_provider_via_get_provider():
    """End-to-end: create provider via factory and call API."""
    from domainbench.core.config import ProviderType, ModelConfig
    from domainbench.providers import get_provider

    config = ModelConfig(
        provider=ProviderType.CEREBRAS,
        model="llama3.1-8b",
        temperature=0.0,
    )
    provider = get_provider(config)

    result = provider.chat_completion(
        model=config.model,
        messages=[{"role": "user", "content": "Reply with just the word 'ok'."}],
        temperature=config.temperature,
        max_tokens=16,
    )

    assert "content" in result
    assert len(result["content"]) > 0
    print(f"  Factory provider result: {result['content'][:40]}")
