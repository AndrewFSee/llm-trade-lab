"""LLM client tests — JSON extraction + factory selection.
Provider integration tests are deferred (would need real API keys + cost)."""
from __future__ import annotations

import pytest

from llm_trade_lab.llm.client import (
    AnthropicClient,
    OllamaClient,
    default_client,
    extract_json,
)


def test_extract_json_direct() -> None:
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_extract_json_fenced() -> None:
    text = "Here you go:\n```json\n{\"name\": \"foo\", \"n\": 3}\n```\nThanks."
    assert extract_json(text) == {"name": "foo", "n": 3}


def test_extract_json_fenced_no_lang() -> None:
    text = "```\n{\"a\": 1}\n```"
    assert extract_json(text) == {"a": 1}


def test_extract_json_substring_fallback() -> None:
    text = "Sure, the answer is {\"verdict\": \"buy\"} based on my analysis."
    assert extract_json(text) == {"verdict": "buy"}


def test_extract_json_raises_when_no_json() -> None:
    with pytest.raises(ValueError, match="No parseable JSON"):
        extract_json("just some prose, no JSON here")


def test_extract_json_raises_on_malformed() -> None:
    with pytest.raises(ValueError):
        extract_json("{not valid json}")


def test_default_client_raises_when_nothing_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")  # nothing listening
    with pytest.raises(RuntimeError, match="No LLM provider available"):
        default_client()


def test_default_client_picks_anthropic_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    client = default_client()
    assert isinstance(client, AnthropicClient)
    assert client.provider == "anthropic"


def test_anthropic_client_default_model() -> None:
    # Construction with explicit fake key (Anthropic SDK accepts arbitrary string)
    c = AnthropicClient(api_key="sk-ant-test")
    assert c.model == "claude-haiku-4-5"
    assert c.provider == "anthropic"


def test_ollama_client_default_model() -> None:
    c = OllamaClient(host="http://127.0.0.1:1")  # not used at construction
    assert c.model == "qwen2.5:7b-instruct"
    assert c.provider == "ollama"
