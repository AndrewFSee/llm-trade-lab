"""Compare equal-weighted vs sizing-weighted vs full-conviction-weighted
aggregate portfolio stats across all backtest_result rows.

Tests whether the model's calibration discipline (sizing, P(event), beneficiary
confidence) sorts positions correctly at portfolio level. If full_conviction
beats equal on excess-vs-SPY, the model is adding value by knowing what to
weight up; if not, conviction is just noise we shouldn't trust.

Usage:
  uv run python scripts/portfolio_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm_trade_lab.data.yfinance_ingest import fetch_ohlcv
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.portfolio.sizing import (
    build_positions,
    compare_weighting_schemes,
)

LEDGER_PATH = Path("data/ledger.db")


def _spy_returns_for(positions) -> dict[tuple[str, str], float | None]:
    """Fetch SPY total return for every unique window in positions."""
    windows = sorted({(p.window_start, p.window_end) for p in positions})
    out: dict[tuple[str, str], float | None] = {}
    for start, end in windows:
        try:
            data = fetch_ohlcv("SPY", start=start, end=end)
            if len(data) < 2:
                out[(start, end)] = None
                continue
            out[(start, end)] = float(data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)
        except Exception:
            out[(start, end)] = None
    return out


def _fmt(v, fmt: str = "+.2%") -> str:
    return format(v, fmt) if v is not None else "n/a"


def main() -> None:
    if not LEDGER_PATH.exists():
        print(f"No ledger at {LEDGER_PATH}. Generate hypotheses first.")
        return

    ledger = Ledger(LEDGER_PATH)
    print("Building positions from backtest_result rows...")
    positions = build_positions(ledger)
    print(f"  {len(positions)} positions across {len({p.hypothesis_id for p in positions})} hypotheses")

    print("Fetching SPY benchmark per window (cached after first fetch)...")
    spy_returns = _spy_returns_for(positions)
    n_with_spy = sum(1 for v in spy_returns.values() if v is not None)
    print(f"  SPY data available for {n_with_spy} of {len(spy_returns)} unique windows")

    print()
    schemes = compare_weighting_schemes(positions, spy_returns)
    print(f"{'scheme':<18s}  {'n':>5s}  {'mean_raw':>10s}  {'mean_excess':>12s}  {'median_excess':>14s}  {'win_rate_vs_spy':>16s}  {'mean_weight':>12s}")
    print("-" * 100)
    for s in schemes:
        print(
            f"{s.name:<18s}  {s.n_positions:>5d}  "
            f"{_fmt(s.mean_raw_return):>10s}  "
            f"{_fmt(s.mean_excess):>12s}  "
            f"{_fmt(s.median_excess):>14s}  "
            f"{_fmt(s.win_rate_vs_spy, '.1%'):>16s}  "
            f"{_fmt(s.mean_weight, '.3f'):>12s}"
        )

    print()
    equal = next(s for s in schemes if s.name == "equal")
    full = next(s for s in schemes if s.name == "full_conviction")
    if equal.mean_excess is not None and full.mean_excess is not None:
        delta = full.mean_excess - equal.mean_excess
        verdict = (
            "Conviction adds value (full > equal on excess)"
            if delta > 0.005
            else "Conviction is noise or negative (full <= equal)"
            if delta < -0.005
            else "Indistinguishable"
        )
        print(f"full_conviction excess - equal excess = {delta:+.2%}    -> {verdict}")
    print()
    print("Caveats:")
    print("- This is aggregate mean comparison, NOT a true portfolio simulation")
    print("  (no trade timing, no cash management, no overlap accounting).")
    print("- Statistical hypotheses have no event_probability or beneficiary confidence;")
    print("  their 'full_conviction' weight equals their 'sizing_only' weight.")
    print("- A meaningful conviction edge requires both: (a) calibrated weights AND")
    print("  (b) those weights actually differentiating winners from losers.")


if __name__ == "__main__":
    main()
