"""Inspect hypotheses + backtest results with SPY benchmark + excess return.

The key question this answers: "did this hypothesis BEAT the market over its
window, or just ride it up?" Excess return = strategy_return - SPY_return over
the same window.

Usage:
  uv run python scripts/inspect_hypothesis.py
      Default: top 5 + bottom 5 by raw return, with summary stats.

  uv run python scripts/inspect_hypothesis.py --excess
      Sort by EXCESS return (strategy minus SPY) instead of raw.

  uv run python scripts/inspect_hypothesis.py --name nvda
      Show all hypotheses whose name contains "nvda".

  uv run python scripts/inspect_hypothesis.py --id 1af6c6
      Show hypothesis with this id prefix.

  uv run python scripts/inspect_hypothesis.py --type event_driven
      Filter to event-driven only.

  uv run python scripts/inspect_hypothesis.py --windows
      Show distribution of backtest windows (are all wins clustered in one period?).
"""
from __future__ import annotations

import argparse
import statistics
import sys

# LLM output sometimes contains unicode (sigma, plusminus, em-dash) that the
# default Windows console (cp1252) can't encode. Reconfigure to UTF-8 with
# replace fallback so we never crash mid-inspection.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter
from datetime import date
from pathlib import Path

from llm_trade_lab.data.yfinance_ingest import fetch_ohlcv
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.schema.hypothesis import EventDrivenHypothesis

LEDGER_PATH = Path("data/ledger.db")
_SPY_CACHE: dict[tuple[str, str], float | None] = {}


def _spy_return(start: str, end: str) -> float | None:
    key = (start, end)
    if key in _SPY_CACHE:
        return _SPY_CACHE[key]
    try:
        data = fetch_ohlcv("SPY", start=start, end=end)
        if len(data) < 2:
            ret = None
        else:
            ret = float(data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)
    except Exception:
        ret = None
    _SPY_CACHE[key] = ret
    return ret


def _print_hypothesis(ledger: Ledger, hid: str, h) -> None:
    print(f"\n{'-' * 88}")
    print(f"id: {hid}    type: {h.type}    name: {h.name}")
    print(f"model: {h.model_version_hash}    generated: {h.generated_at.date()}")
    print(f"\n  thesis: {h.thesis_text}")
    print(f"\n  universe: {h.universe}    entry: {h.entry_rule.strategy_id} {h.entry_rule.params}")
    if isinstance(h, EventDrivenHypothesis):
        ev = h.trigger_event
        print(
            f"\n  event:    {ev.source}/{ev.doc_id}  date={ev.event_date}  "
            f"P(event)={h.event_probability:.2f}  horizon={h.expected_horizon_days}d"
        )
        for b in h.beneficiaries:
            print(f"    {b.ticker:<6s} conf={b.confidence:.2f}  {b.mechanism[:80]}")
        if h.confounders:
            print(f"  confounders: {'; '.join(c[:60] for c in h.confounders[:3])}")
    if h.generation_params and "retrieved_k" in h.generation_params:
        rk = h.generation_params["retrieved_k"]
        print(f"\n  retrieved_k={rk}")

    rows = ledger.query_results(hid)
    if not rows:
        print("  (no backtest results)")
        return
    print(f"\n  backtests:")
    print(f"    {'tkr':<6s}  {'window':<24s}  {'return':>9s}  {'spy':>9s}  {'excess':>9s}  {'sharpe':>7s}")
    for r in rows:
        ret = r.get("return_pct")
        sharpe = r.get("sharpe")
        spy = _spy_return(r["window_start"], r["window_end"])
        ret_s = f"{ret:>+9.2%}" if ret is not None else "      n/a"
        sharpe_s = f"{sharpe:>+7.2f}" if sharpe is not None else "    n/a"
        if spy is None:
            spy_s = "      n/a"
            excess_s = "      n/a"
        elif ret is None:
            spy_s = f"{spy:>+9.2%}"
            excess_s = "      n/a"
        else:
            spy_s = f"{spy:>+9.2%}"
            excess_s = f"{(ret - spy):>+9.2%}"
        print(
            f"    {r['universe_ticker']:<6s}  "
            f"{r['window_start']}->{r['window_end']}  "
            f"{ret_s}  {spy_s}  {excess_s}  {sharpe_s}"
        )


def _summarize_excess(rows_with_excess: list[tuple[float, float | None]]) -> None:
    excess = [e for _, e in rows_with_excess if e is not None]
    raw = [r for r, _ in rows_with_excess]
    if not excess:
        print("  (no SPY data; can't compute excess)")
        return
    n = len(excess)
    mean_raw = statistics.fmean(raw)
    mean_excess = statistics.fmean(excess)
    median_excess = statistics.median(excess)
    win_raw = sum(1 for r in raw if r > 0) / len(raw)
    win_excess = sum(1 for e in excess if e > 0) / n
    print(f"\n{'=' * 88}")
    print("AGGREGATE: market-relative signal check")
    print(f"  backtests with SPY data: {n}")
    print(f"  mean RAW return:    {mean_raw:+.2%}    win rate vs zero: {win_raw:.1%}")
    print(f"  mean EXCESS return: {mean_excess:+.2%}    win rate vs SPY:  {win_excess:.1%}")
    print(f"  median EXCESS:      {median_excess:+.2%}")
    if mean_excess > 0.005 and win_excess > 0.55:
        verdict = "Possibly real edge — but small sample, watch for selection effects"
    elif mean_excess < -0.005:
        verdict = "Underperforms market on average"
    else:
        verdict = "Indistinguishable from buying SPY (no real edge in this sample)"
    print(f"\n  verdict: {verdict}")


def _show_window_distribution(rows: list[dict]) -> None:
    print(f"\n{'=' * 88}")
    print("BACKTEST WINDOW DISTRIBUTION")
    starts = [r["window_start"] for r in rows]
    by_month = Counter(s[:7] for s in starts)
    for ym in sorted(by_month):
        bar = "#" * by_month[ym]
        print(f"  {ym}  {by_month[ym]:>3d}  {bar}")
    if len(set(s[:7] for s in starts)) < 4:
        print("\n  Note: backtests cluster in <4 months. Returns may reflect a single regime.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--bottom", type=int, default=5)
    p.add_argument("--id", help="hypothesis id prefix")
    p.add_argument("--name", help="substring match on hypothesis name")
    p.add_argument("--type", choices=["statistical", "event_driven"], help="filter by type")
    p.add_argument("--model-version", help="substring filter on model_version_hash (e.g. 'p4' for prompt v4 only)")
    p.add_argument("--excess", action="store_true", help="sort by excess-vs-SPY instead of raw")
    p.add_argument("--windows", action="store_true", help="show backtest-window distribution")
    args = p.parse_args()

    if not LEDGER_PATH.exists():
        print(f"No ledger at {LEDGER_PATH}. Run a generation script first.")
        return
    ledger = Ledger(LEDGER_PATH)

    # specific lookups
    if args.id or args.name:
        if args.id:
            matches = [(hid, h) for hid, h in ledger.iter_hypotheses() if hid.startswith(args.id)]
        else:
            needle = args.name.lower()
            matches = [(hid, h) for hid, h in ledger.iter_hypotheses() if needle in h.name.lower()]
        if not matches:
            print("no matches")
            return
        for hid, h in matches:
            _print_hypothesis(ledger, hid, h)
        return

    # aggregate analysis: pull all backtest rows, score each by raw or excess
    hyps = ledger.iter_hypotheses(hypothesis_type=args.type)
    if args.model_version:
        needle = args.model_version
        hyps = [(hid, h) for hid, h in hyps if needle in h.model_version_hash]
        if not hyps:
            print(f"no hypotheses match model_version_hash containing {needle!r}")
            return
        print(f"filtered to {len(hyps)} hypotheses with model_version containing {needle!r}\n")
    all_results: list[tuple[float, float | None]] = []
    by_hyp: list[tuple[str, object, float, float, int]] = []  # (hid, h, score, mean_raw, n_bt)
    for hid, h in hyps:
        rows = ledger.query_results(hid)
        if not rows:
            continue
        per_row_raw = [r["return_pct"] for r in rows]
        per_row_excess = []
        for r in rows:
            spy = _spy_return(r["window_start"], r["window_end"])
            all_results.append((r["return_pct"], None if spy is None else r["return_pct"] - spy))
            per_row_excess.append(None if spy is None else r["return_pct"] - spy)
        mean_raw = statistics.fmean(per_row_raw)
        ex_vals = [e for e in per_row_excess if e is not None]
        mean_excess = statistics.fmean(ex_vals) if ex_vals else None
        score = mean_excess if (args.excess and mean_excess is not None) else mean_raw
        by_hyp.append((hid, h, score, mean_raw, len(rows)))

    if not by_hyp:
        print("ledger has no hypotheses with backtest results")
        return

    by_hyp.sort(key=lambda x: -x[2])

    sort_label = "EXCESS-vs-SPY" if args.excess else "raw return"
    print(f"\n{'TOP ' + str(args.top) + ' by ' + sort_label:=^88}")
    for hid, h, _, _, _ in by_hyp[: args.top]:
        _print_hypothesis(ledger, hid, h)

    if args.bottom > 0:
        print(f"\n{'BOTTOM ' + str(args.bottom) + ' by ' + sort_label:=^88}")
        for hid, h, _, _, _ in by_hyp[-args.bottom :][::-1]:
            _print_hypothesis(ledger, hid, h)

    _summarize_excess(all_results)

    if args.windows:
        all_rows: list[dict] = []
        for hid, _, _, _, _ in by_hyp:
            all_rows.extend(ledger.query_results(hid))
        _show_window_distribution(all_rows)


if __name__ == "__main__":
    main()
