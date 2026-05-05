"""Quality report tests with a hand-built ledger."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from llm_trade_lab.backtest.engine import BacktestResult
from llm_trade_lab.eval.quality import compare_with_without_memory, compute_quality_report
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.schema.hypothesis import (
    Beneficiary,
    EntryExitRule,
    EventDrivenHypothesis,
    StatisticalHypothesis,
    TriggerEvent,
)


def _stat(name: str, model_v: str = "m1", retrieved_k: int = 0) -> StatisticalHypothesis:
    return StatisticalHypothesis(
        name=name,
        thesis_text="x",
        universe=["AAPL", "MSFT"],
        entry_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        exit_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        sizing=1.0,
        generated_at=datetime.now(timezone.utc),
        model_version_hash=model_v,
        generation_params={"retrieved_k": retrieved_k},
    )


def _evt(name: str, prob: float, retrieved_k: int = 0) -> EventDrivenHypothesis:
    return EventDrivenHypothesis(
        name=name,
        thesis_text="x",
        universe=["CF"],
        entry_rule=EntryExitRule(strategy_id="event_hold"),
        exit_rule=EntryExitRule(strategy_id="event_hold"),
        sizing=0.5,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="m1",
        generation_params={"retrieved_k": retrieved_k},
        trigger_event=TriggerEvent(
            source="congress_bill",
            doc_id=f"doc-{name}",
            event_type="introduced",
            event_date=date(2025, 3, 1),
        ),
        beneficiaries=[Beneficiary(ticker="CF", mechanism="m", confidence=0.7)],
        event_probability=prob,
        expected_horizon_days=30,
    )


def _bt(ret: float, sharpe: float = 1.0) -> BacktestResult:
    return BacktestResult(
        sharpe=sharpe, sortino=sharpe * 1.2, return_pct=ret,
        max_drawdown_pct=-0.1, n_trades=2, win_rate_pct=0.5,
        cost_bps=5.0, cash=10000.0,
    )


def test_empty_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    r = compute_quality_report(ledger)
    assert r.n_hypotheses == 0
    assert r.testability_rate == 0.0
    assert r.mean_return_pct is None
    assert r.win_rate is None


def test_quality_with_data(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    # Two stat, one event-driven; some have backtests
    h1 = _stat("a"); h2 = _stat("b", model_v="m2"); h3 = _evt("c", 0.4)
    id1 = ledger.insert_hypothesis(h1)
    id2 = ledger.insert_hypothesis(h2)
    id3 = ledger.insert_hypothesis(h3)
    ledger.insert_backtest_result(id1, "AAPL", "2025-01-01", "2025-02-01", _bt(0.10, 1.5))
    ledger.insert_backtest_result(id1, "MSFT", "2025-01-01", "2025-02-01", _bt(-0.05, -0.3))
    ledger.insert_backtest_result(id3, "CF", "2025-03-01", "2025-04-01", _bt(0.20, 2.0))
    # h2 has no backtests

    r = compute_quality_report(ledger)
    assert r.n_hypotheses == 3
    assert r.n_statistical == 2
    assert r.n_event_driven == 1
    assert r.n_with_backtests == 2
    assert r.testability_rate == pytest.approx(2 / 3)
    assert r.n_backtest_results == 3
    # rets: 0.10, -0.05, 0.20 -> mean ~ 0.0833
    assert r.mean_return_pct == pytest.approx((0.10 - 0.05 + 0.20) / 3)
    assert r.win_rate == pytest.approx(2 / 3)  # 2 of 3 positive
    assert r.mean_event_probability == pytest.approx(0.4)
    assert r.by_model_version == {"m1": 2, "m2": 1}
    assert r.by_type == {"statistical": 2, "event_driven": 1}


def test_compare_with_without_memory(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    cold = _stat("cold", retrieved_k=0)
    warm = _stat("warm", retrieved_k=3)
    cold_id = ledger.insert_hypothesis(cold)
    warm_id = ledger.insert_hypothesis(warm)
    ledger.insert_backtest_result(cold_id, "AAPL", "2025-01-01", "2025-02-01", _bt(-0.05))
    ledger.insert_backtest_result(warm_id, "AAPL", "2025-01-01", "2025-02-01", _bt(0.10))

    cmp = compare_with_without_memory(ledger)
    assert cmp["with_memory"]["n_hypotheses"] == 1
    assert cmp["with_memory"]["mean_return"] == pytest.approx(0.10)
    assert cmp["with_memory"]["win_rate"] == pytest.approx(1.0)
    assert cmp["cold_start"]["n_hypotheses"] == 1
    assert cmp["cold_start"]["mean_return"] == pytest.approx(-0.05)
    assert cmp["cold_start"]["win_rate"] == pytest.approx(0.0)


def test_render_runs(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    h = _stat("a")
    hid = ledger.insert_hypothesis(h)
    ledger.insert_backtest_result(hid, "AAPL", "2025-01-01", "2025-02-01", _bt(0.05))
    r = compute_quality_report(ledger)
    text = r.render()
    assert "Quality report" in text
    assert "testability" in text
