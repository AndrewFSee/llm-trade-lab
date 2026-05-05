"""Generator tests with a mocked LLM client (no API calls)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from llm_trade_lab.events.event_stream import Event
from llm_trade_lab.llm.client import LLMResponse
from llm_trade_lab.llm.generator import (
    HypothesisGenerationError,
    generate_event_driven_hypothesis,
    generate_statistical_hypothesis,
)
from llm_trade_lab.schema.hypothesis import EventDrivenHypothesis, StatisticalHypothesis


class FakeClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_system: str | None = None
        self.last_prompt: str | None = None

    def complete(self, *, system, prompt, max_tokens=2048, temperature=0.7, json_mode=False):
        self.last_system = system
        self.last_prompt = prompt
        return LLMResponse(
            text=self.response_text,
            model=self.model,
            provider=self.provider,
            usage={"input_tokens": 100, "output_tokens": 50},
        )


# ----------------------------------------------------------- statistical

STAT_LLM_OK = """{
  "name": "aapl_breakout_20d",
  "thesis_text": "AAPL breaking above 20-day SMA with rising volume signals momentum continuation.",
  "universe": ["AAPL"],
  "entry_rule": {"strategy_id": "sma_cross", "params": {"fast": 10, "slow": 30}},
  "exit_rule":  {"strategy_id": "sma_cross", "params": {"fast": 10, "slow": 30}},
  "holding_period_days": null,
  "sizing": 0.8
}"""


def test_generate_statistical_happy_path() -> None:
    client = FakeClient(STAT_LLM_OK)
    h = generate_statistical_hypothesis(
        universe_hint=["AAPL"], client=client, today="2026-04-26"
    )
    assert isinstance(h, StatisticalHypothesis)
    assert h.type == "statistical"
    assert h.name == "aapl_breakout_20d"
    assert h.universe == ["AAPL"]
    assert h.entry_rule.strategy_id == "sma_cross"
    assert h.entry_rule.params == {"fast": 10, "slow": 30}
    assert h.sizing == 0.8
    # Server-filled fields:
    assert h.model_version_hash == "fake:fake-model:p7"
    assert h.generation_params["today"] == "2026-04-26"


def test_generate_statistical_handles_fenced_json() -> None:
    fenced = "```json\n" + STAT_LLM_OK + "\n```"
    client = FakeClient(fenced)
    h = generate_statistical_hypothesis(client=client)
    assert h.name == "aapl_breakout_20d"


def test_generate_statistical_raises_on_unparseable() -> None:
    client = FakeClient("Sorry, I can't help with that.")
    with pytest.raises(HypothesisGenerationError, match="Could not extract JSON"):
        generate_statistical_hypothesis(client=client)


def test_generate_statistical_raises_on_schema_violation() -> None:
    bad = '{"name": "x", "thesis_text": "y", "universe": []}'  # missing required fields
    client = FakeClient(bad)
    with pytest.raises(HypothesisGenerationError, match="schema validation"):
        generate_statistical_hypothesis(client=client)


# ----------------------------------------------------------- event-driven

EVT_LLM_OK = """{
  "name": "farm_bill_fertilizer",
  "thesis_text": "Farm bill subsidy for nitrogen fertilizer raises domestic demand for CF and MOS.",
  "universe": ["CF", "MOS"],
  "entry_rule": {"strategy_id": "event_hold", "params": {}},
  "exit_rule":  {"strategy_id": "event_hold", "params": {}},
  "holding_period_days": 60,
  "sizing": 0.5,
  "trigger_event": {
    "source": "congress_bill",
    "doc_id": "hr-1234-119",
    "event_type": "introduced",
    "event_date": "2025-03-01"
  },
  "beneficiaries": [
    {"ticker": "CF", "mechanism": "nitrogen producer benefits from subsidized corn fertilizer", "confidence": 0.7},
    {"ticker": "MOS", "mechanism": "potash demand uplift", "confidence": 0.6}
  ],
  "event_probability": 0.25,
  "expected_horizon_days": 90,
  "confounders": ["bill stalls in committee", "broad market drawdown"]
}"""


def _fake_event() -> Event:
    return Event(
        source="congress_bill",
        event_type="introduced",
        event_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        title="Farm bill providing fertilizer subsidies",
        body="A bill to provide subsidies for domestic fertilizer production...",
        raw_id="hr-1234-119",
        suggested_beneficiaries=[
            SimpleNamespace(
                ticker="CF", name="CF Industries", mechanism="nitrogen producer",
                confidence=0.9, matched_theme="fertilizer",
            ),
            SimpleNamespace(
                ticker="MOS", name="Mosaic", mechanism="phosphate / potash",
                confidence=0.9, matched_theme="fertilizer",
            ),
        ],
        matched_themes=["fertilizer"],
    )


def test_generate_event_driven_happy_path() -> None:
    client = FakeClient(EVT_LLM_OK)
    event = _fake_event()
    h = generate_event_driven_hypothesis(event=event, client=client)
    assert isinstance(h, EventDrivenHypothesis)
    assert h.type == "event_driven"
    assert h.universe == ["CF", "MOS"]
    assert h.event_probability == pytest.approx(0.25)
    assert h.expected_horizon_days == 90
    assert h.trigger_event.source == "congress_bill"
    assert {b.ticker for b in h.beneficiaries} == {"CF", "MOS"}


def test_generate_event_driven_raises_when_no_beneficiaries() -> None:
    event = Event(
        source="congress_bill",
        event_type="introduced",
        event_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        title="Some unrelated bill",
        body="...",
        raw_id="hr-9999-119",
        suggested_beneficiaries=[],  # entity resolver matched nothing
        matched_themes=[],
    )
    client = FakeClient(EVT_LLM_OK)
    with pytest.raises(HypothesisGenerationError, match="no theme-matched beneficiaries"):
        generate_event_driven_hypothesis(event=event, client=client)


def test_user_prompt_includes_event_context() -> None:
    client = FakeClient(EVT_LLM_OK)
    event = _fake_event()
    generate_event_driven_hypothesis(event=event, client=client)
    assert client.last_prompt is not None
    assert "Farm bill providing fertilizer subsidies" in client.last_prompt
    assert "CF" in client.last_prompt and "MOS" in client.last_prompt
    assert "fertilizer" in client.last_prompt
