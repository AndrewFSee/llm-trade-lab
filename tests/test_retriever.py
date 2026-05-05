"""HypothesisRetriever tests using a deterministic fake encoder so we avoid
loading the real sentence-transformers model in offline tests."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.memory.retriever import HypothesisRetriever, _hypothesis_text
from llm_trade_lab.schema.hypothesis import (
    Beneficiary,
    EntryExitRule,
    EventDrivenHypothesis,
    StatisticalHypothesis,
    TriggerEvent,
)

DIM = 32


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Deterministic per-text 32-dim normalized embedding via hashing.

    Identical strings -> identical embeddings (perfect cosine 1.0).
    Substring overlap is roughly preserved by the byte-level hashing.
    """
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8")).digest()
        # Spread 32 bytes across 32 floats deterministically.
        for j in range(DIM):
            out[i, j] = (h[j] / 127.5) - 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _stat_hyp(name: str, thesis: str) -> StatisticalHypothesis:
    return StatisticalHypothesis(
        name=name,
        thesis_text=thesis,
        universe=["AAPL"],
        entry_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        exit_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        sizing=1.0,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="test_v0",
    )


def _evt_hyp(name: str, thesis: str) -> EventDrivenHypothesis:
    return EventDrivenHypothesis(
        name=name,
        thesis_text=thesis,
        universe=["CF"],
        entry_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 5, "slow": 20}),
        exit_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 5, "slow": 20}),
        sizing=0.5,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="test_v0",
        trigger_event=TriggerEvent(
            source="congress_bill",
            doc_id="hr-1234-119",
            event_type="bill_introduced",
            event_date=date(2025, 3, 1),
        ),
        beneficiaries=[Beneficiary(ticker="CF", mechanism="fertilizer", confidence=0.8)],
        event_probability=0.4,
        expected_horizon_days=60,
    )


@pytest.fixture
def retriever(tmp_path: Path) -> HypothesisRetriever:
    ledger = Ledger(tmp_path / "ledger.db")
    r = HypothesisRetriever(
        ledger,
        index_dir=tmp_path / "faiss",
        encode_fn=_fake_encode,
    )
    return r


def test_empty_search_returns_empty(retriever: HypothesisRetriever) -> None:
    assert retriever.search("anything") == []
    assert retriever.size == 0


def test_add_and_search_finds_exact_match(retriever: HypothesisRetriever) -> None:
    h = _stat_hyp("test1", "Apple momentum on positive earnings revisions.")
    hid = retriever.ledger.insert_hypothesis(h)
    retriever.add_hypothesis(hid, h)

    results = retriever.search(_hypothesis_text(h), k=1)
    assert len(results) == 1
    found_hid, score, found_h = results[0]
    assert found_hid == hid
    assert score == pytest.approx(1.0)
    assert found_h.name == "test1"


def test_add_is_idempotent(retriever: HypothesisRetriever) -> None:
    h = _stat_hyp("t", "x")
    hid = retriever.ledger.insert_hypothesis(h)
    retriever.add_hypothesis(hid, h)
    retriever.add_hypothesis(hid, h)
    assert retriever.size == 1


def test_filter_by_type(retriever: HypothesisRetriever) -> None:
    s1 = _stat_hyp("s1", "Statistical idea about momentum.")
    s2 = _stat_hyp("s2", "Statistical idea about mean reversion.")
    e1 = _evt_hyp("e1", "Event-driven idea about a fertilizer subsidy bill.")
    e2 = _evt_hyp("e2", "Event-driven idea about a tariff bill.")

    for h in (s1, s2, e1, e2):
        hid = retriever.ledger.insert_hypothesis(h)
        retriever.add_hypothesis(hid, h)
    assert retriever.size == 4

    stat_results = retriever.search("anything", k=4, filter_type="statistical")
    assert all(h.type == "statistical" for _, _, h in stat_results)
    assert len(stat_results) == 2

    evt_results = retriever.search("anything", k=4, filter_type="event_driven")
    assert all(h.type == "event_driven" for _, _, h in evt_results)
    assert len(evt_results) == 2


def test_save_and_reload_round_trip(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    r1 = HypothesisRetriever(ledger, index_dir=tmp_path / "faiss", encode_fn=_fake_encode)
    h = _stat_hyp("t", "Some thesis text.")
    hid = ledger.insert_hypothesis(h)
    r1.add_hypothesis(hid, h)
    r1.save()

    r2 = HypothesisRetriever(ledger, index_dir=tmp_path / "faiss", encode_fn=_fake_encode)
    assert r2.size == 1
    results = r2.search(_hypothesis_text(h), k=1)
    assert len(results) == 1
    assert results[0][0] == hid


def test_reindex_from_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    h1 = _stat_hyp("a", "first")
    h2 = _stat_hyp("b", "second")
    ledger.insert_hypothesis(h1)
    ledger.insert_hypothesis(h2)

    r = HypothesisRetriever(ledger, index_dir=tmp_path / "faiss", encode_fn=_fake_encode)
    n = r.reindex_from_ledger()
    assert n == 2
    assert r.size == 2
