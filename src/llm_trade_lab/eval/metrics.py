"""Risk-adjusted return + probability calibration metrics.

Provides:
  - sharpe_ratio, sortino_ratio, max_drawdown          (basic)
  - probabilistic_sharpe_ratio                         (Bailey & Lopez de Prado)
  - deflated_sharpe_ratio                              (Bailey & Lopez de Prado)
  - brier_score, reliability_curve, expected_calibration_error  (event probability)

Why deflated Sharpe matters for this project: the LLM will generate many
hypotheses per round. Picking the best by raw Sharpe guarantees you'll
"discover" spurious edges from multiple-comparisons selection bias. DSR
adjusts the threshold for the number of trials, returning the probability
that the best observed Sharpe is genuinely above zero.

Why calibration matters: event_driven hypotheses include an explicit
event_probability field (e.g., P(bill passes) = 0.4). Brier score and
reliability curves measure whether those probabilities track reality.
A well-calibrated 0.7-confidence bucket should resolve true ~70% of the time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------- basics

def sharpe_ratio(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized Sharpe of a return series. Returns 0.0 for degenerate inputs."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(math.sqrt(periods_per_year) * r.mean() / sd)


def sortino_ratio(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized Sortino (downside-deviation only)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0:
        return 0.0
    sd_down = downside.std(ddof=1)
    if sd_down == 0:
        return 0.0
    return float(math.sqrt(periods_per_year) * r.mean() / sd_down)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Returns max drawdown as a negative fraction (e.g. -0.25 = -25%)."""
    e = np.asarray(equity_curve, dtype=float)
    if len(e) == 0:
        return 0.0
    running_max = np.maximum.accumulate(e)
    dd = (e - running_max) / running_max
    return float(dd.min())


# ---------------------------------------------------------- Sharpe inference

def _sample_skew(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    sd = x.std(ddof=1)
    if sd == 0:
        return 0.0
    m = x.mean()
    return float(((x - m) ** 3).mean() / sd ** 3)


def _sample_kurtosis(x: np.ndarray) -> float:
    """Raw (non-excess) kurtosis; normal distribution = 3.0."""
    n = len(x)
    if n < 4:
        return 3.0
    sd = x.std(ddof=1)
    if sd == 0:
        return 3.0
    m = x.mean()
    return float(((x - m) ** 4).mean() / sd ** 4)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n_observations: int,
    *,
    sr_benchmark: float = 0.0,
    returns: Sequence[float] | None = None,
) -> float:
    """PSR: probability that the true Sharpe exceeds `sr_benchmark`.

    If `returns` is provided, skew + raw kurtosis are estimated from it
    (more accurate for fat-tailed return distributions). Otherwise assumes
    normal returns. `observed_sr` should be in the same units as the
    benchmark — typically annualized.

    Reference: Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier."
    """
    if n_observations < 2:
        return 0.5
    if returns is not None:
        r = np.asarray(returns, dtype=float)
        skew = _sample_skew(r)
        kurt = _sample_kurtosis(r)
    else:
        skew = 0.0
        kurt = 3.0
    denom_sq = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr ** 2
    if denom_sq <= 0:
        return 0.5
    z = (observed_sr - sr_benchmark) * math.sqrt(n_observations - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def expected_max_sr(n_trials: int, var_sr_trials: float) -> float:
    """Expected maximum Sharpe across `n_trials` trials with variance `var_sr_trials`.

    Used by deflated_sharpe_ratio as the deflated benchmark.
    """
    if n_trials <= 1:
        return 0.0
    sigma = math.sqrt(max(var_sr_trials, 0.0))
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * math.e))
    return float(sigma * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_observations: int,
    n_trials: int,
    *,
    var_sr_trials: float = 1.0,
    returns: Sequence[float] | None = None,
) -> float:
    """DSR: probability the best observed Sharpe is genuinely above zero,
    after adjusting for multiple-trial selection bias.

    Args:
        observed_sr: Sharpe of the (best) selected strategy.
        n_observations: number of return periods in the selected strategy.
        n_trials: total trials evaluated (the multiple-comparisons count).
        var_sr_trials: variance of Sharpes across all trials. If you don't
            have this, 1.0 is a conservative default; smaller values make
            DSR more lenient.
        returns: if provided, used to estimate skew + kurt for the PSR step.

    Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014).
    """
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(
            observed_sr, n_observations, sr_benchmark=0.0, returns=returns
        )
    sr_benchmark = expected_max_sr(n_trials, var_sr_trials)
    return probabilistic_sharpe_ratio(
        observed_sr, n_observations, sr_benchmark=sr_benchmark, returns=returns
    )


# ----------------------------------------------------------- calibration

@dataclass
class ReliabilityBin:
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    fraction_positive: float
    count: int


def brier_score(
    predicted_probs: Sequence[float],
    outcomes: Sequence[int | float],
) -> float:
    """Mean squared error between predicted probability and binary outcome.

    Range [0, 1]; lower is better. Perfect prediction = 0; constant 0.5
    on uniform outcomes ~= 0.25.
    """
    p = np.asarray(predicted_probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return 0.0
    return float(((p - y) ** 2).mean())


def reliability_curve(
    predicted_probs: Sequence[float],
    outcomes: Sequence[int | float],
    *,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bin predictions by predicted probability; return per-bin calibration stats.

    A perfectly calibrated model has fraction_positive == mean_predicted
    in every bin. Diagonal y=x is the calibration target.
    """
    p = np.asarray(predicted_probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        out.append(
            ReliabilityBin(
                bin_lower=lo,
                bin_upper=hi,
                mean_predicted=float(p[mask].mean()),
                fraction_positive=float(y[mask].mean()),
                count=int(mask.sum()),
            )
        )
    return out


def expected_calibration_error(
    predicted_probs: Sequence[float],
    outcomes: Sequence[int | float],
    *,
    n_bins: int = 10,
) -> float:
    """ECE: weighted-by-bin-count mean of |fraction_positive - mean_predicted|.

    Range [0, 1]; lower is better. Perfect calibration = 0.
    """
    bins = reliability_curve(predicted_probs, outcomes, n_bins=n_bins)
    n_total = sum(b.count for b in bins)
    if n_total == 0:
        return 0.0
    return float(
        sum(
            (b.count / n_total) * abs(b.fraction_positive - b.mean_predicted)
            for b in bins
        )
    )
