"""LLM client abstraction with two backends.

Provider-agnostic `complete()` API consumed by the hypothesis generators.
Backends:
  - AnthropicClient (Claude) — prompt caching enabled on system block
  - OllamaClient (local) — free fallback if you have `ollama serve` running

`default_client()` picks Anthropic when ANTHROPIC_API_KEY is set, otherwise
probes the Ollama host. Models default to claude-haiku-4-5 (cheap + fast for
prompt iteration) and qwen2.5:7b-instruct respectively.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: dict[str, int] | None = None


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> LLMResponse: ...


# ------------------------------------------------------------- Anthropic

class AnthropicClient:
    """Claude API client. System prompts are prompt-cached (ephemeral) — reuse
    of the same system prompt within ~5 minutes is billed at ~10% input cost."""

    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
    ):
        from anthropic import Anthropic

        self.model = model
        # Anthropic() reads ANTHROPIC_API_KEY env when api_key is None.
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> LLMResponse:
        system_blocks = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        user_content = prompt
        if json_mode:
            user_content = (
                prompt + "\n\nReturn ONLY a single JSON object. No prose, no code fences."
            )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        u = resp.usage
        usage = {
            "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
            "cache_creation_input_tokens": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        }
        return LLMResponse(text=text, model=self.model, provider=self.provider, usage=usage)


# --------------------------------------------------------------- Ollama

class OllamaClient:
    """Local Ollama server client. Default host http://localhost:11434.

    Pull a model first: `ollama pull qwen2.5:7b-instruct`.
    """

    provider = "ollama"

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        host: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        r = self._client.post(f"{self.host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "") or ""
        usage = {
            "input_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(data.get("eval_count", 0) or 0),
        }
        return LLMResponse(text=text, model=self.model, provider=self.provider, usage=usage)


# ---------------------------------------------------------- factory

def default_client() -> LLMClient:
    """Pick a backend based on environment.

    1. ANTHROPIC_API_KEY set         -> AnthropicClient
    2. else probe Ollama at OLLAMA_HOST -> OllamaClient if reachable
    3. else raise with setup instructions
    """
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return AnthropicClient()

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        r = httpx.get(f"{host}/api/tags", timeout=2.0)
        if r.status_code == 200:
            return OllamaClient(host=host)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
        pass

    raise RuntimeError(
        "No LLM provider available. Either set ANTHROPIC_API_KEY in .env "
        "(free $5 credit at https://console.anthropic.com/), or run "
        "`ollama serve` locally with a model pulled "
        "(e.g., `ollama pull qwen2.5:7b-instruct`)."
    )


# ---------------------------------------------------------- json helper

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Extract a single JSON object from LLM output, tolerating fences and prose.

    Tries in order: direct parse, ```json fenced, first-{ to last-} substring.
    Raises ValueError if no parseable JSON object is found.
    """
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    first = s.find("{")
    last = s.rfind("}")
    if first >= 0 and last > first:
        try:
            return json.loads(s[first : last + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No parseable JSON object in LLM response: {text[:200]!r}")
