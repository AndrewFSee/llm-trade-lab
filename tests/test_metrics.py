"""Tests for risk-adjusted return + calibration metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from llm_trade_lab.eval.metrics import (
    brier_score,
    deflated_sharpe_ratio,
    expected_calibration_error,
    expected_max_sr,
    max_drawdown,
    probabilistic_sharpe_ratio,
    reliability_curve,
    sharpe_ratio,
    sortino_ratio,
)


# -------------------------------------------------------------- Sharpe basics

def test_sharpe_handles_short_or_constant() -> None:
    assert sharpe_ratio([]) == 0.0
    assert sharpe_ratio([0.01]) == 0.0
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_positive_for_positive_drift() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=2520)
    sr = sharpe_ratio(r)
    # Theoretical: sqrt(252) * 0.001 / 0.01 ~= 1.59; empirical tolerance.
    assert 1.0 < sr < 2.5


def test_sortino_excludes_upside_volatility() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=2520)
    s = sharpe_ratio(r)
    so = sortino_ratio(r)
    # Sortino on near-symmetric returns is roughly sqrt(2) * Sharpe.
    assert so > s


def test_max_drawdown_known_curve() -> None:
    eq = [100.0, 110.0, 120.0, 90.0, 95.0, 100.0]
    # peak 120 -> trough 90 = -25%
    assert max_drawdown(eq) == pytest.approx(-0.25)


def test_max_drawdown_monotone_increasing_is_zero() -> None:
    assert max_drawdown([100.0, 101.0, 102.0]) == pytest.approx(0.0)


# -------------------------------------------------------------- PSR

def test_psr_when_sr_equals_benchmark_returns_half() -> None:
    p = probabilistic_sharpe_ratio(0.5, n_observations=252, sr_benchmark=0.5)
    assert p == pytest.approx(0.5, abs=1e-6)


def test_psr_increases_with_more_observations() -> None:
    p_short = probabilistic_sharpe_ratio(1.0, n_observations=30, sr_benchmark=0.0)
    p_long = probabilistic_sharpe_ratio(1.0, n_observations=2520, sr_benchmark=0.0)
    assert p_long > p_short


def test_psr_uses_returns_for_skew_kurt() -> None:
    """With positive per-period SR and synthetic left-tailed returns,
    passing `returns` (which estimates skew + kurt) should yield strictly
    lower PSR than the normal-distribution assumption.

    Uses per-period (daily) SR computed directly from the series to keep
    PSR unsaturated — annualized SR with ~60 obs saturates the formula.
    """
    r = np.array([0.001] * 60, dtype=float)
    r[5] = -0.02
    r[30] = -0.025
    daily_sr = float(r.mean() / r.std(ddof=1))
    p_normal = probabilistic_sharpe_ratio(daily_sr, n_observations=60)
    p_with_returns = probabilistic_sharpe_ratio(daily_sr, n_observations=60, returns=r)
    assert 0.05 < p_normal < 0.95
    assert p_with_returns < p_normal


# -------------------------------------------------------------- DSR

def test_dsr_identical_to_psr_when_n_trials_one() -> None:
    p = probabilistic_sharpe_ratio(1.0, n_observations=500)
    d = deflated_sharpe_ratio(1.0, n_observations=500, n_trials=1)
    assert p == pytest.approx(d)


def test_dsr_decreases_with_more_trials() -> None:
    d_few = deflated_sharpe_ratio(2.0, n_observations=500, n_trials=2, var_sr_trials=1.0)
    d_many = deflated_sharpe_ratio(2.0, n_observations=500, n_trials=200, var_sr_trials=1.0)
    assert d_many < d_few


def test_expected_max_sr_grows_with_trials() -> None:
    e10 = expected_max_sr(10, var_sr_trials=1.0)
    e1000 = expected_max_sr(1000, var_sr_trials=1.0)
    assert e1000 > e10


# -------------------------------------------------------------- calibration

def test_brier_perfect_predictions_is_zero() -> None:
    assert brier_score([0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1]) == 0.0


def test_brier_constant_half_on_uniform_outcomes_is_quarter() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=10_000)
    p = [0.5] * len(y)
    assert brier_score(p, y) == pytest.approx(0.25, abs=0.005)


def test_reliability_curve_perfect_calibration() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < p).astype(int)  # outcomes drawn from predicted
    bins = reliability_curve(p, y, n_bins=10)
    for b in bins:
        # Allow some sampling noise; tolerance scales with 1/sqrt(count).
        tol = 4.0 / math.sqrt(b.count)
        assert abs(b.fraction_positive - b.mean_predicted) <= tol, b


def test_reliability_curve_empty_bins_are_skipped() -> None:
    bins = reliability_curve([0.05, 0.06, 0.07], [0, 0, 1], n_bins=10)
    # All predictions in bin 0; only one bin should appear.
    assert len(bins) == 1
    assert bins[0].count == 3


def test_ece_perfect_calibration_low() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < p).astype(int)
    e = expected_calibration_error(p, y, n_bins=10)
    assert e < 0.03


def test_ece_overconfident_model_is_high() -> None:
    # Model predicts 0.9 always; true outcomes are 50/50.
    rng = np.random.default_rng(0)
    n = 1000
    p = [0.9] * n
    y = rng.integers(0, 2, size=n)
    e = expected_calibration_error(p, y, n_bins=10)
    assert e > 0.3
