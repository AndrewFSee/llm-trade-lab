"""RAG loop driver tests with a fake LLM client + fake encoder."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llm_trade_lab.events.event_stream import Event
from llm_trade_lab.llm.client import LLMResponse
from llm_trade_lab.llm.loop import (
    _format_outcome,
    _retrieve_memory,
    _seen_event_doc_ids,
    generate_event_driven_with_memory,
    generate_statistical_with_memory,
)
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.memory.retriever import HypothesisRetriever
from llm_trade_lab.schema.hypothesis import EventDrivenHypothesis, StatisticalHypothesis

DIM = 32


def _fake_encode(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8")).digest()
        for j in range(DIM):
            out[i, j] = (h[j] / 127.5) - 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


STAT_RESPONSE = """{
  "name": "test_stat",
  "thesis_text": "Test statistical hypothesis with memory.",
  "universe": ["AAPL"],
  "entry_rule": {"strategy_id": "sma_cross", "params": {"fast": 10, "slow": 30}},
  "exit_rule":  {"strategy_id": "sma_cross", "params": {"fast": 10, "slow": 30}},
  "holding_period_days": null,
  "sizing": 0.5
}"""

EVT_RESPONSE = """{
  "name": "test_event",
  "thesis_text": "Test event-driven hypothesis.",
  "universe": ["CF"],
  "entry_rule": {"strategy_id": "event_hold", "params": {}},
  "exit_rule":  {"strategy_id": "event_hold", "params": {}},
  "holding_period_days": 30,
  "sizing": 0.5,
  "trigger_event": {
    "source": "congress_bill",
    "doc_id": "hr-1234-119",
    "event_type": "introduced",
    "event_date": "2025-03-01"
  },
  "beneficiaries": [
    {"ticker": "CF", "mechanism": "fertilizer demand", "confidence": 0.7}
  ],
  "event_probability": 0.4,
  "expected_horizon_days": 30,
  "confounders": ["bill stalls"]
}"""


class FakeClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[dict] = []

    def complete(self, *, system, prompt, max_tokens=2048, temperature=0.7, json_mode=False):
        self.calls.append({"system": system, "prompt": prompt})
        return LLMResponse(
            text=self.response_text,
            model=self.model,
            provider=self.provider,
            usage={"input_tokens": 100, "output_tokens": 50},
        )


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.db")


@pytest.fixture
def retriever(tmp_path: Path, ledger: Ledger) -> HypothesisRetriever:
    return HypothesisRetriever(
        ledger, index_dir=tmp_path / "faiss", encode_fn=_fake_encode
    )


def _seed_stat_hypothesis(
    ledger: Ledger,
    retriever: HypothesisRetriever,
    name: str,
    *,
    ticker: str = "AAPL",
) -> str:
    from llm_trade_lab.schema.hypothesis import EntryExitRule

    h = StatisticalHypothesis(
        name=name,
        thesis_text=f"Seed hypothesis {name}",
        universe=[ticker],
        entry_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        exit_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        sizing=1.0,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="seed_v0",
    )
    hid = ledger.insert_hypothesis(h)
    retriever.add_hypothesis(hid, h)
    return hid


def test_format_outcome_no_results(ledger: Ledger) -> None:
    assert _format_outcome(ledger, "nonexistent") == "no realized outcome yet"


def test_format_outcome_aggregates(ledger: Ledger, retriever: HypothesisRetriever) -> None:
    hid = _seed_stat_hypothesis(ledger, retriever, "h1")
    from llm_trade_lab.backtest.engine import BacktestResult
    for ticker, ret in [("AAPL", 0.10), ("MSFT", -0.05), ("NVDA", 0.20)]:
        ledger.insert_backtest_result(
            hypothesis_id=hid,
            universe_ticker=ticker,
            window_start="2025-01-01",
            window_end="2025-02-01",
            result=BacktestResult(
                sharpe=1.0, sortino=1.5, return_pct=ret,
                max_drawdown_pct=-0.1, n_trades=1, win_rate_pct=0.0,
                cost_bps=5.0, cash=10000.0,
            ),
        )
    out = _format_outcome(ledger, hid)
    assert "+8.33%" in out  # avg of 10, -5, 20 = 8.33
    assert "3 ticker" in out
    assert "2 winner" in out


def test_retrieve_memory_returns_top_k(ledger: Ledger, retriever: HypothesisRetriever) -> None:
    # Distinct tickers because retrieval dedupes by primary ticker.
    for i, t in enumerate(["AAPL", "MSFT", "NVDA"]):
        _seed_stat_hypothesis(ledger, retriever, f"h{i}", ticker=t)
    memory, ids = _retrieve_memory(retriever, ledger, query="anything", k=2, filter_type="statistical")
    assert len(memory) == 2
    assert len(ids) == 2
    for m in memory:
        assert "name" in m and "thesis" in m and "outcome" in m


def test_retrieve_memory_dedupes_by_ticker(
    ledger: Ledger, retriever: HypothesisRetriever
) -> None:
    """3 hypotheses on the same ticker collapse to 1 retrieved entry — this is
    the anti-mode-collapse mechanism."""
    for i in range(3):
        _seed_stat_hypothesis(ledger, retriever, f"all_aapl_{i}")
    memory, ids = _retrieve_memory(
        retriever, ledger, query="anything", k=5, filter_type="statistical"
    )
    assert len(memory) == 1
    assert len(ids) == 1


def test_retrieve_memory_k_zero(ledger: Ledger, retriever: HypothesisRetriever) -> None:
    memory, ids = _retrieve_memory(retriever, ledger, query="x", k=0, filter_type=None)
    assert memory == [] and ids == []


def test_generate_statistical_with_memory_lineage(
    ledger: Ledger, retriever: HypothesisRetriever
) -> None:
    seed_id = _seed_stat_hypothesis(ledger, retriever, "seed")
    client = FakeClient(STAT_RESPONSE)
    run = generate_statistical_with_memory(
        universe_hint=["AAPL"],
        ledger=ledger,
        retriever=retriever,
        client=client,
        k=5,
        # Narrow custom window prevents the test from hitting yfinance for the
        # default 5-month walk-forward windows.
        walk_forward_windows=[("2024-07-01", "2024-07-02")],
    )
    assert run.error is None
    assert run.hypothesis is not None
    assert run.hypothesis.name == "test_stat"
    # Lineage: retrieved seed_id should appear in generation_params
    assert seed_id in run.retrieved_ids
    assert run.hypothesis.generation_params["retrieved_hypothesis_ids"] == run.retrieved_ids
    assert run.hypothesis.generation_params["retrieved_k"] == len(run.retrieved_ids)
    # Prompt should mention memory
    assert "Retrieved past hypotheses" in client.calls[0]["prompt"]


def test_generate_statistical_handles_llm_failure(
    ledger: Ledger, retriever: HypothesisRetriever
) -> None:
    client = FakeClient("not valid json")
    run = generate_statistical_with_memory(
        universe_hint=["AAPL"],
        ledger=ledger,
        retriever=retriever,
        client=client,
    )
    assert run.error is not None
    assert run.hypothesis is None


def test_seen_event_doc_ids(ledger: Ledger, retriever: HypothesisRetriever) -> None:
    # Insert one event-driven hypothesis with a known doc_id
    from llm_trade_lab.schema.hypothesis import (
        Beneficiary,
        EntryExitRule,
        TriggerEvent,
    )
    h = EventDrivenHypothesis(
        name="ed1",
        thesis_text="x",
        universe=["CF"],
        entry_rule=EntryExitRule(strategy_id="event_hold"),
        exit_rule=EntryExitRule(strategy_id="event_hold"),
        sizing=0.5,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="v0",
        trigger_event=TriggerEvent(
            source="congress_bill",
            doc_id="hr-9999-119",
            event_type="introduced",
            event_date=date(2025, 1, 1),
        ),
        beneficiaries=[Beneficiary(ticker="CF", mechanism="m", confidence=0.5)],
        event_probability=0.3,
        expected_horizon_days=30,
    )
    ledger.insert_hypothesis(h)
    seen = _seen_event_doc_ids(ledger)
    assert "hr-9999-119" in seen


def test_generate_event_driven_with_memory(
    ledger: Ledger, retriever: HypothesisRetriever
) -> None:
    event = Event(
        source="congress_bill",
        event_type="introduced",
        event_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        title="Farm bill fertilizer subsidy",
        body="A bill...",
        raw_id="hr-1234-119",
        suggested_beneficiaries=[
            SimpleNamespace(
                ticker="CF", name="CF", mechanism="n", confidence=0.9, matched_theme="fertilizer"
            )
        ],
        matched_themes=["fertilizer"],
    )
    client = FakeClient(EVT_RESPONSE)
    run = generate_event_driven_with_memory(
        event=event,
        ledger=ledger,
        retriever=retriever,
        client=client,
        today=date(2025, 4, 30),  # well past the 30-day horizon
    )
    assert run.error is None
    assert run.hypothesis is not None
    assert run.hypothesis.name == "test_event"
    assert run.hypothesis.generation_params["retrieved_k"] == 0  # empty index initially
