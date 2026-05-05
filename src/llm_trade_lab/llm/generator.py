"""Generate Hypothesis objects from an LLM.

The generator owns:
  - Building the user prompt from context (regime features / Event / memory)
  - Calling the LLM with the cached system prompt
  - Extracting + parsing JSON
  - Filling in fields the LLM should NOT control (type discriminator,
    generated_at timestamp, model_version_hash, generation_params)
  - Validating against the pydantic schema

Caller (the loop driver) is responsible for: deciding when to generate,
fetching retrieval context, persisting to ledger, scheduling backtests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import TypeAdapter, ValidationError

from llm_trade_lab.events.event_stream import Event
from llm_trade_lab.llm.client import LLMClient, default_client, extract_json
from llm_trade_lab.llm.prompts import (
    EVENT_DRIVEN_SYSTEM_PROMPT,
    PROMPT_VERSION,
    STATISTICAL_SYSTEM_PROMPT,
    build_event_driven_user_prompt,
    build_statistical_user_prompt,
)
from llm_trade_lab.schema.hypothesis import (
    EventDrivenHypothesis,
    Hypothesis,
    StatisticalHypothesis,
)

_HYPOTHESIS_ADAPTER: TypeAdapter[Hypothesis] = TypeAdapter(Hypothesis)


def _model_version_hash(client: LLMClient) -> str:
    return f"{client.provider}:{client.model}:p{PROMPT_VERSION}"


class HypothesisGenerationError(Exception):
    """Raised when the LLM output cannot be parsed/validated into a Hypothesis."""

    def __init__(self, message: str, *, raw_text: str = "", parsed: dict | None = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.parsed = parsed


def generate_statistical_hypothesis(
    *,
    today: str | None = None,
    universe_hint: list[str] | None = None,
    ticker_flags: dict[str, list[str]] | None = None,
    sector_contexts: dict[str, list[dict[str, str]]] | None = None,
    regime_features: dict[str, Any] | None = None,
    retrieved_memory: list[dict[str, Any]] | None = None,
    client: LLMClient | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> StatisticalHypothesis:
    """Generate one StatisticalHypothesis from the LLM."""
    client = client or default_client()
    today = today or datetime.now(timezone.utc).date().isoformat()
    user = build_statistical_user_prompt(
        today=today,
        universe_hint=universe_hint,
        ticker_flags=ticker_flags,
        sector_contexts=sector_contexts,
        regime_features=regime_features,
        retrieved_memory=retrieved_memory,
    )
    resp = client.complete(
        system=STATISTICAL_SYSTEM_PROMPT,
        prompt=user,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=True,
    )
    return _parse_and_validate(
        raw_text=resp.text,
        client=client,
        hypothesis_type="statistical",
        generation_params={"temperature": temperature, "max_tokens": max_tokens, "today": today},
    )  # type: ignore[return-value]


def generate_event_driven_hypothesis(
    *,
    event: Event,
    candidate_beneficiaries: list[dict[str, Any]] | None = None,
    retrieved_memory: list[dict[str, Any]] | None = None,
    client: LLMClient | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> EventDrivenHypothesis:
    """Generate one EventDrivenHypothesis grounded in a unified-stream Event.

    `candidate_beneficiaries` may be pre-enriched by the caller (with `flags`
    and `current_context` keys for richer prompting). If omitted, candidates
    are extracted from `event.suggested_beneficiaries` without enrichment.
    """
    client = client or default_client()
    if candidate_beneficiaries is None:
        candidate_beneficiaries = [
            {
                "ticker": b.ticker,
                "name": b.name,
                "mechanism": b.mechanism,
                "confidence": b.confidence,
                "matched_theme": b.matched_theme,
                "current_context": getattr(b, "current_context", ""),
                "flags": [],
            }
            for b in event.suggested_beneficiaries
        ]
    if not candidate_beneficiaries:
        raise HypothesisGenerationError(
            f"Event {event.event_id} has no theme-matched beneficiaries; "
            "expand configs/entities.yaml or pick a different event."
        )
    event_dict = {
        "source": event.source,
        "doc_id": event.raw_id,
        "event_type": event.event_type,
        "event_date": event.event_date.date().isoformat(),
        "title": event.title,
        "body": event.body,
    }
    user = build_event_driven_user_prompt(
        event=event_dict,
        candidate_beneficiaries=candidate_beneficiaries,
        retrieved_memory=retrieved_memory,
    )
    resp = client.complete(
        system=EVENT_DRIVEN_SYSTEM_PROMPT,
        prompt=user,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=True,
    )
    return _parse_and_validate(
        raw_text=resp.text,
        client=client,
        hypothesis_type="event_driven",
        generation_params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "event_id": event.event_id,
        },
    )  # type: ignore[return-value]


def _parse_and_validate(
    *,
    raw_text: str,
    client: LLMClient,
    hypothesis_type: str,
    generation_params: dict[str, Any],
) -> Hypothesis:
    try:
        payload = extract_json(raw_text)
    except ValueError as e:
        raise HypothesisGenerationError(
            f"Could not extract JSON from LLM output: {e}",
            raw_text=raw_text,
        ) from e

    # Server-controlled fields: do not trust the LLM for these.
    payload["type"] = hypothesis_type
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["model_version_hash"] = _model_version_hash(client)
    payload["generation_params"] = generation_params

    # Coerce common LLM mistakes (sizing=0, expected_horizon_days=0) to the
    # minimum allowed values rather than rejecting the whole generation.
    sizing = payload.get("sizing")
    if isinstance(sizing, (int, float)) and sizing <= 0:
        payload["sizing"] = 0.05
    if hypothesis_type == "event_driven":
        horizon = payload.get("expected_horizon_days")
        if isinstance(horizon, (int, float)) and horizon <= 0:
            payload["expected_horizon_days"] = 5

    try:
        return _HYPOTHESIS_ADAPTER.validate_python(payload)
    except ValidationError as e:
        raise HypothesisGenerationError(
            f"LLM output failed schema validation: {e}",
            raw_text=raw_text,
            parsed=payload,
        ) from e
