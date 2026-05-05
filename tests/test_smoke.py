"""Smoke tests — no network, fast, run on every change."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_trade_lab.backtest.engine import run_backtest
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.schema.hypothesis import (
    Beneficiary,
    EntryExitRule,
    EventDrivenHypothesis,
    StatisticalHypothesis,
    TriggerEvent,
)


def _synthetic_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1_000_000, 10_000_000, n)
    idx = pd.date_range("2024-07-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _statistical_hypothesis() -> StatisticalHypothesis:
    return StatisticalHypothesis(
        name="test_sma_cross",
        thesis_text="SMA cross test.",
        universe=["TEST"],
        entry_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        exit_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        sizing=1.0,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="test_v0",
    )


def _event_driven_hypothesis() -> EventDrivenHypothesis:
    return EventDrivenHypothesis(
        name="test_farm_bill",
        thesis_text="Farm bill subsidizes fertilizer; CF/MOS benefit.",
        universe=["CF", "MOS"],
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
        beneficiaries=[
            Beneficiary(ticker="CF", mechanism="fertilizer subsidy demand", confidence=0.7),
            Beneficiary(ticker="MOS", mechanism="potash demand uplift", confidence=0.6),
        ],
        event_probability=0.35,
        expected_horizon_days=60,
        confounders=["bill stalls in committee", "OPEC price shock"],
    )


def test_backtest_runs_on_synthetic_data() -> None:
    h = _statistical_hypothesis()
    data = _synthetic_ohlcv()
    result = run_backtest(h, data)
    assert result.n_trades >= 0
    assert -1.0 <= result.max_drawdown_pct <= 0.0
    assert result.cost_bps == pytest.approx(5.0)


@pytest.mark.parametrize(
    "strategy_id,params",
    [
        ("sma_cross", {"fast": 10, "slow": 30}),
        ("rsi_reversion", {"period": 14, "buy_threshold": 30, "exit_threshold": 70}),
        ("donchian_breakout", {"entry_lookback": 20, "exit_lookback": 10}),
        ("bollinger_reversion", {"period": 20, "n_std": 2.0}),
    ],
)
def test_each_statistical_strategy_runs(strategy_id: str, params: dict) -> None:
    """Every registered statistical strategy must run on synthetic data without
    raising — guards against silent crashes during LLM-driven generation."""
    from llm_trade_lab.schema.hypothesis import EntryExitRule

    h = StatisticalHypothesis(
        name=f"smoke_{strategy_id}",
        thesis_text="smoke",
        universe=["TEST"],
        entry_rule=EntryExitRule(strategy_id=strategy_id, params=params),
        exit_rule=EntryExitRule(strategy_id=strategy_id, params=params),
        sizing=1.0,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="smoke_v0",
    )
    data = _synthetic_ohlcv()
    result = run_backtest(h, data)
    assert result.n_trades >= 0
    assert -1.0 <= result.max_drawdown_pct <= 0.0


def test_ledger_round_trip_statistical(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    h = _statistical_hypothesis()
    hid = ledger.insert_hypothesis(h)
    assert len(hid) == 16

    same = ledger.insert_hypothesis(h)
    assert same == hid  # idempotent

    retrieved = ledger.get_hypothesis(hid)
    assert retrieved is not None
    assert retrieved.type == "statistical"
    assert retrieved.name == h.name


def test_ledger_round_trip_event_driven(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    h = _event_driven_hypothesis()
    hid = ledger.insert_hypothesis(h)
    retrieved = ledger.get_hypothesis(hid)
    assert retrieved is not None
    assert retrieved.type == "event_driven"
    assert isinstance(retrieved, EventDrivenHypothesis)
    assert retrieved.event_probability == pytest.approx(0.35)
    assert {b.ticker for b in retrieved.beneficiaries} == {"CF", "MOS"}


def test_ledger_records_backtest_result(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    h = _statistical_hypothesis()
    hid = ledger.insert_hypothesis(h)
    data = _synthetic_ohlcv()
    result = run_backtest(h, data)
    bt_id = ledger.insert_backtest_result(
        hypothesis_id=hid,
        universe_ticker="TEST",
        window_start="2024-07-01",
        window_end="2025-08-01",
        result=result,
    )
    assert bt_id >= 1
    rows = ledger.query_results(hid)
    assert len(rows) == 1
    assert rows[0]["universe_ticker"] == "TEST"
    assert rows[0]["n_trades"] == result.n_trades
