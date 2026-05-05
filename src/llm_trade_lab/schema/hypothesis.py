"""Pydantic schemas for trading hypotheses.

Two variants share a `type` discriminator:
  - StatisticalHypothesis: regime-conditioned price/fundamentals idea.
  - EventDrivenHypothesis: catalyst-conditioned (bill, 8-K, FOMC) with named
    beneficiaries and an explicit event-probability estimate.

The LLM emits these as JSON; downstream code parses them into a runnable backtest spec.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class EntryExitRule(BaseModel):
    """Reference to a registered strategy plus its parameters.

    `strategy_id` keys into `llm_trade_lab.backtest.engine.STRATEGY_REGISTRY`.
    """

    strategy_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class TriggerEvent(BaseModel):
    """Reference to the source event that motivated an event-driven hypothesis.

    `source` matches the EventSource literals in events/event_stream.py so a
    TriggerEvent can be constructed directly from a unified-stream Event.
    """

    source: Literal[
        "congress_bill",
        "edgar_8k",
        "form4",
        "federal_register",
        "fomc",
        "fred_release",
        "alphavantage_news",
    ]
    doc_id: str
    event_type: str
    event_date: date


class Beneficiary(BaseModel):
    ticker: str
    mechanism: str
    confidence: float = Field(ge=0.0, le=1.0)


class _HypothesisBase(BaseModel):
    name: str
    thesis_text: str
    universe: list[str]
    entry_rule: EntryExitRule
    exit_rule: EntryExitRule
    holding_period_days: int | None = None
    sizing: float = Field(default=1.0, gt=0.0, le=1.0)
    generated_at: datetime
    model_version_hash: str
    generation_params: dict[str, Any] = Field(default_factory=dict)
    market_regime_features: dict[str, float] = Field(default_factory=dict)


class StatisticalHypothesis(_HypothesisBase):
    type: Literal["statistical"] = "statistical"


class EventDrivenHypothesis(_HypothesisBase):
    type: Literal["event_driven"] = "event_driven"
    trigger_event: TriggerEvent
    beneficiaries: list[Beneficiary]
    event_probability: float = Field(ge=0.0, le=1.0)
    expected_horizon_days: int = Field(gt=0)
    confounders: list[str] = Field(default_factory=list)


Hypothesis = Annotated[
    StatisticalHypothesis | EventDrivenHypothesis,
    Field(discriminator="type"),
]
