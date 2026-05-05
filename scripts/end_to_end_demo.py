"""Smallest end-to-end slice through Phase 0.

Hand-written SMA-crossover hypothesis on AAPL -> fetch OHLCV -> backtest ->
log to SQLite ledger -> retrieve back. Validates the architecture before
building the rest of the data ingest layer.

Window starts 2024-07 to stay strictly after Qwen3's mid-2024 knowledge cutoff,
per the lookahead-bias mitigation in the project plan.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from llm_trade_lab.backtest.engine import run_backtest
from llm_trade_lab.data.yfinance_ingest import fetch_ohlcv
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.schema.hypothesis import EntryExitRule, StatisticalHypothesis

WINDOW_START = "2024-07-01"
WINDOW_END = "2026-04-25"
TICKER = "AAPL"
DB_PATH = Path("data/ledger.db")


def main() -> None:
    load_dotenv()
    hypothesis = StatisticalHypothesis(
        name="aapl_sma_20_50_cross",
        thesis_text=(
            "On AAPL, a 20-day SMA crossing above the 50-day SMA signals a momentum "
            "regime; cross-down exits the position."
        ),
        universe=[TICKER],
        entry_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 20, "slow": 50}),
        exit_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 20, "slow": 50}),
        holding_period_days=None,
        sizing=1.0,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="hand_written_v0",
        generation_params={},
        market_regime_features={},
    )

    print(f"Fetching {TICKER} OHLCV {WINDOW_START} -> {WINDOW_END} ...")
    data = fetch_ohlcv(TICKER, start=WINDOW_START, end=WINDOW_END)
    print(f"  -> {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")

    print("Running backtest ...")
    result = run_backtest(hypothesis, data)
    print(
        f"  -> trades={result.n_trades}  return={result.return_pct:+.2%}  "
        f"sharpe={result.sharpe:.2f}  maxDD={result.max_drawdown_pct:.2%}  "
        f"win={result.win_rate_pct:.2%}"
    )

    print(f"Logging to ledger at {DB_PATH} ...")
    ledger = Ledger(DB_PATH)
    hid = ledger.insert_hypothesis(hypothesis)
    bt_id = ledger.insert_backtest_result(
        hypothesis_id=hid,
        universe_ticker=TICKER,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        result=result,
    )
    print(f"  -> hypothesis_id={hid}  backtest_id={bt_id}")

    retrieved = ledger.get_hypothesis(hid)
    assert retrieved is not None and retrieved.name == hypothesis.name
    results = ledger.query_results(hid)
    assert len(results) >= 1
    print(f"Round-trip OK. Ledger now has {len(results)} backtest result(s) for {retrieved.name}.")


if __name__ == "__main__":
    main()
