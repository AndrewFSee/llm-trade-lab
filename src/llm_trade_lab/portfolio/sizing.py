"""Cross-hypothesis portfolio sizing.

The honest question this answers: when the model says "low conviction" via small
sizing, low P(event), or low beneficiary confidence, does the trade actually
underperform high-conviction trades? If yes, **conviction-weighted aggregation
should beat equal-weighted aggregation** because we're putting more capital
behind the bets the model believes in.

This is the portfolio-level test of whether calibration discipline matters.

Three weighting schemes computed side by side:
  - equal:           weight = 1 for every position (baseline, no model info)
  - sizing_only:     weight = hypothesis.sizing (just the size decision)
  - full_conviction: weight = sizing × P(event) × beneficiary_confidence
                              (sizing for statistical; full chain for event_driven)

Aggregate metrics are computed across all (hypothesis, ticker, window) cells
in the ledger. This is NOT a true portfolio simulation (no trade-timing
overlap, no cash management) — it's an aggregate weighted-mean test that
answers "does conviction sort positions correctly?"
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.schema.hypothesis import EventDrivenHypothesis


@dataclass
class WeightedPosition:
    hypothesis_id: str
    hypothesis_type: str
    ticker: str
    window_start: str
    window_end: str
    return_pct: float
    sizing: float
    event_probability: float | None       # event_driven only
    beneficiary_confidence: float | None  # event_driven only

    @property
    def weight_equal(self) -> float:
        return 1.0

    @property
    def weight_sizing_only(self) -> float:
        return self.sizing

    @property
    def weight_full_conviction(self) -> float:
        w = self.sizing
        if self.event_probability is not None:
            w *= self.event_probability
        if self.beneficiary_confidence is not None:
            w *= self.beneficiary_confidence
        return w


def build_positions(ledger: Ledger) -> list[WeightedPosition]:
    """One WeightedPosition per backtest_result row, joined to its hypothesis."""
    rows = ledger.query_all_results()
    positions: list[WeightedPosition] = []
    for r in rows:
        if r.get("return_pct") is None:
            continue
        h = ledger.get_hypothesis(r["hypothesis_id"])
        if h is None:
            continue
        ticker = r["universe_ticker"]
        event_prob: float | None = None
        ben_conf: float | None = None
        if isinstance(h, EventDrivenHypothesis):
            event_prob = h.event_probability
            for b in h.beneficiaries:
                if b.ticker.upper() == ticker.upper():
                    ben_conf = b.confidence
                    break
        positions.append(
            WeightedPosition(
                hypothesis_id=r["hypothesis_id"],
                hypothesis_type=h.type,
                ticker=ticker,
                window_start=r["window_start"],
                window_end=r["window_end"],
                return_pct=float(r["return_pct"]),
                sizing=float(h.sizing),
                event_probability=event_prob,
                beneficiary_confidence=ben_conf,
            )
        )
    return positions


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    total = sum(weights)
    if total == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total


@dataclass
class SchemeStats:
    name: str
    n_positions: int
    mean_raw_return: float | None
    mean_excess: float | None
    median_excess: float | None
    win_rate_vs_spy: float | None
    mean_weight: float | None


def evaluate_scheme(
    positions: list[WeightedPosition],
    weights: list[float],
    spy_returns: dict[tuple[str, str], float],
    *,
    name: str,
) -> SchemeStats:
    """Compute weighted-mean stats for one weighting scheme."""
    rets = [p.return_pct for p in positions]
    raw_mean = _weighted_mean(rets, weights)

    excesses: list[float] = []
    excess_weights: list[float] = []
    win = 0
    for p, w in zip(positions, weights):
        spy = spy_returns.get((p.window_start, p.window_end))
        if spy is None:
            continue
        excess = p.return_pct - spy
        excesses.append(excess)
        excess_weights.append(w)
        if excess > 0:
            win += 1
    return SchemeStats(
        name=name,
        n_positions=len(positions),
        mean_raw_return=raw_mean,
        mean_excess=_weighted_mean(excesses, excess_weights),
        median_excess=statistics.median(excesses) if excesses else None,
        win_rate_vs_spy=(win / len(excesses)) if excesses else None,
        mean_weight=statistics.fmean(weights) if weights else None,
    )


def compare_weighting_schemes(
    positions: list[WeightedPosition],
    spy_returns: dict[tuple[str, str], float],
) -> list[SchemeStats]:
    """Run all three schemes and return their stats for side-by-side display."""
    return [
        evaluate_scheme(
            positions, [p.weight_equal for p in positions], spy_returns, name="equal"
        ),
        evaluate_scheme(
            positions, [p.weight_sizing_only for p in positions], spy_returns, name="sizing_only"
        ),
        evaluate_scheme(
            positions, [p.weight_full_conviction for p in positions], spy_returns, name="full_conviction"
        ),
    ]
