"""Hypothesis-quality metrics computed from the ledger.

The plan's success criteria (no PnL chasing) live here:
  - testability_rate     % of generated hypotheses that produced >=1 backtest result
                         (parse + validate + at least one ticker had usable data)
  - by_type counts       balance of statistical vs. event_driven generations
  - return distribution  mean / median return across all (hypothesis, ticker) results
  - win_rate             % of (hypothesis, ticker) results with positive return
  - mean_sharpe          mean Sharpe across results (annualized, per backtesting.py)
  - by_model_version     count by `model_version_hash` so we can A/B prompt revisions
                         and (later) base vs. fine-tuned model versions
  - mean_event_probability  for event-driven only — gut check on calibration prior
                            to per-bin reliability scoring (which needs resolved events)
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.schema.hypothesis import EventDrivenHypothesis


@dataclass
class QualityReport:
    n_hypotheses: int
    n_statistical: int
    n_event_driven: int
    n_with_backtests: int
    testability_rate: float
    n_backtest_results: int
    mean_return_pct: float | None
    median_return_pct: float | None
    win_rate: float | None
    mean_sharpe: float | None
    median_sharpe: float | None
    mean_universe_size: float | None
    mean_event_probability: float | None
    by_model_version: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        def f(v: float | None, fmt: str = ".2%") -> str:
            return format(v, fmt) if v is not None else "n/a"

        lines = [
            "Quality report",
            f"  hypotheses:     {self.n_hypotheses} total ({self.n_statistical} stat, {self.n_event_driven} event)",
            f"  testability:    {self.n_with_backtests}/{self.n_hypotheses} = {self.testability_rate:.1%}",
            f"  backtest rows:  {self.n_backtest_results}",
            f"  return:         mean {f(self.mean_return_pct)}, median {f(self.median_return_pct)}",
            f"  win rate:       {f(self.win_rate, '.1%')}",
            # Median Sharpe is the trustworthy summary. Mean Sharpe is distorted
            # by backtesting.py's annualization on short event-hold windows
            # (a 7-day trade losing 1% can produce a Sharpe of -10).
            f"  Sharpe (median): {f(self.median_sharpe, '+.2f')}    <- preferred",
            f"  Sharpe (mean):   {f(self.mean_sharpe, '+.2f')}    <- distorted on short windows; ignore unless many bars/trade",
            f"  univ size:      mean {f(self.mean_universe_size, '.1f')}",
            f"  event prob:     mean {f(self.mean_event_probability, '.2f')}",
        ]
        if self.by_model_version:
            lines.append("  by model_version_hash:")
            for k, v in sorted(self.by_model_version.items()):
                lines.append(f"    {k:<40s}  {v}")
        return "\n".join(lines)


def _safe_mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _safe_median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def compute_quality_report(ledger: Ledger) -> QualityReport:
    hyps = ledger.iter_hypotheses()
    n_total = len(hyps)
    by_type: dict[str, int] = {}
    by_model: dict[str, int] = {}
    universe_sizes: list[int] = []
    event_probs: list[float] = []
    for _, h in hyps:
        by_type[h.type] = by_type.get(h.type, 0) + 1
        by_model[h.model_version_hash] = by_model.get(h.model_version_hash, 0) + 1
        universe_sizes.append(len(h.universe))
        if isinstance(h, EventDrivenHypothesis):
            event_probs.append(h.event_probability)

    results = ledger.query_all_results()
    rets = [r["return_pct"] for r in results if r.get("return_pct") is not None]
    sharpes = [r["sharpe"] for r in results if r.get("sharpe") is not None]
    wins = sum(1 for r in rets if r > 0)
    win_rate = (wins / len(rets)) if rets else None

    hypotheses_with_results = {r["hypothesis_id"] for r in results}
    n_with_bt = len(hypotheses_with_results)
    testability = (n_with_bt / n_total) if n_total else 0.0

    return QualityReport(
        n_hypotheses=n_total,
        n_statistical=by_type.get("statistical", 0),
        n_event_driven=by_type.get("event_driven", 0),
        n_with_backtests=n_with_bt,
        testability_rate=testability,
        n_backtest_results=len(results),
        mean_return_pct=_safe_mean(rets),
        median_return_pct=_safe_median(rets),
        win_rate=win_rate,
        mean_sharpe=_safe_mean(sharpes),
        median_sharpe=_safe_median(sharpes),
        mean_universe_size=_safe_mean([float(x) for x in universe_sizes]),
        mean_event_probability=_safe_mean(event_probs),
        by_model_version=by_model,
        by_type=by_type,
    )


def compare_with_without_memory(ledger: Ledger) -> dict[str, Any]:
    """Compare per-hypothesis stats split by whether retrieval was used.

    Reads `generation_params.retrieved_k` from each hypothesis. Hypotheses
    where retrieved_k > 0 had memory injected; ==0 (or missing) did not.
    Useful for the "did memory help?" question after a bootstrap batch.
    """
    hyps = ledger.iter_hypotheses()
    results = ledger.query_all_results()
    by_hid = {}
    for r in results:
        by_hid.setdefault(r["hypothesis_id"], []).append(r)

    with_mem_returns: list[float] = []
    cold_returns: list[float] = []
    with_mem_n = 0
    cold_n = 0
    for hid, h in hyps:
        rk = (h.generation_params or {}).get("retrieved_k", 0) or 0
        bucket_returns = [
            r["return_pct"]
            for r in by_hid.get(hid, [])
            if r.get("return_pct") is not None
        ]
        if rk > 0:
            with_mem_n += 1
            with_mem_returns.extend(bucket_returns)
        else:
            cold_n += 1
            cold_returns.extend(bucket_returns)

    return {
        "with_memory": {
            "n_hypotheses": with_mem_n,
            "n_results": len(with_mem_returns),
            "mean_return": _safe_mean(with_mem_returns),
            "median_return": _safe_median(with_mem_returns),
            "win_rate": (sum(1 for x in with_mem_returns if x > 0) / len(with_mem_returns))
            if with_mem_returns
            else None,
        },
        "cold_start": {
            "n_hypotheses": cold_n,
            "n_results": len(cold_returns),
            "mean_return": _safe_mean(cold_returns),
            "median_return": _safe_median(cold_returns),
            "win_rate": (sum(1 for x in cold_returns if x > 0) / len(cold_returns))
            if cold_returns
            else None,
        },
    }
