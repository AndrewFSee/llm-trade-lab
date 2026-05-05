"""Tests for portfolio sizing weighting schemes."""
from __future__ import annotations

import pytest

from llm_trade_lab.portfolio.sizing import (
    SchemeStats,
    WeightedPosition,
    _weighted_mean,
    compare_weighting_schemes,
    evaluate_scheme,
)


def _stat_pos(*, ticker="AAPL", ret=0.10, sizing=1.0, ws="2024-07-01", we="2024-12-01") -> WeightedPosition:
    return WeightedPosition(
        hypothesis_id="h1", hypothesis_type="statistical",
        ticker=ticker, window_start=ws, window_end=we,
        return_pct=ret, sizing=sizing,
        event_probability=None, beneficiary_confidence=None,
    )


def _evt_pos(*, ticker="CF", ret=0.05, sizing=0.5, p_event=0.4, conf=0.7,
             ws="2025-03-01", we="2025-06-01") -> WeightedPosition:
    return WeightedPosition(
        hypothesis_id="h2", hypothesis_type="event_driven",
        ticker=ticker, window_start=ws, window_end=we,
        return_pct=ret, sizing=sizing,
        event_probability=p_event, beneficiary_confidence=conf,
    )


def test_weight_equal_is_one() -> None:
    p = _stat_pos()
    assert p.weight_equal == 1.0


def test_weight_sizing_only_for_statistical() -> None:
    p = _stat_pos(sizing=0.6)
    assert p.weight_sizing_only == 0.6
    assert p.weight_full_conviction == 0.6  # no event_prob/conf


def test_weight_full_conviction_for_event_driven() -> None:
    p = _evt_pos(sizing=0.5, p_event=0.4, conf=0.7)
    # 0.5 * 0.4 * 0.7 = 0.14
    assert p.weight_full_conviction == pytest.approx(0.14)


def test_weighted_mean_handles_empty() -> None:
    assert _weighted_mean([], []) is None
    assert _weighted_mean([1.0, 2.0], [0.0, 0.0]) is None


def test_weighted_mean_known_values() -> None:
    # 0.1 * 0.5 + 0.2 * 0.5 = 0.15
    assert _weighted_mean([0.1, 0.2], [0.5, 0.5]) == pytest.approx(0.15)
    # weighted toward second
    assert _weighted_mean([0.1, 0.2], [0.0, 1.0]) == pytest.approx(0.2)


def test_evaluate_scheme_basic() -> None:
    positions = [
        _stat_pos(ret=0.10, ws="2024-07-01", we="2024-12-01"),
        _stat_pos(ret=-0.05, ws="2024-12-01", we="2025-05-01"),
    ]
    spy = {("2024-07-01", "2024-12-01"): 0.05, ("2024-12-01", "2025-05-01"): 0.02}
    weights = [1.0, 1.0]
    s = evaluate_scheme(positions, weights, spy, name="equal")
    # excesses: 0.10-0.05=0.05, -0.05-0.02=-0.07; mean = -0.01
    assert s.name == "equal"
    assert s.n_positions == 2
    assert s.mean_excess == pytest.approx((0.05 + -0.07) / 2)
    assert s.win_rate_vs_spy == pytest.approx(0.5)


def test_evaluate_scheme_skips_missing_spy() -> None:
    positions = [
        _stat_pos(ret=0.10, ws="2024-07-01", we="2024-12-01"),
        _stat_pos(ret=-0.05, ws="missing", we="missing"),
    ]
    spy = {("2024-07-01", "2024-12-01"): 0.05}
    weights = [1.0, 1.0]
    s = evaluate_scheme(positions, weights, spy, name="equal")
    # Only 1 position contributes to excess (other has no SPY)
    assert s.mean_excess == pytest.approx(0.05)


def test_compare_returns_three_schemes() -> None:
    positions = [_stat_pos(), _evt_pos()]
    spy = {
        ("2024-07-01", "2024-12-01"): 0.05,
        ("2025-03-01", "2025-06-01"): 0.02,
    }
    schemes = compare_weighting_schemes(positions, spy)
    names = [s.name for s in schemes]
    assert names == ["equal", "sizing_only", "full_conviction"]
    for s in schemes:
        assert isinstance(s, SchemeStats)
        assert s.n_positions == 2


def test_full_conviction_weights_event_lower_when_low_prob() -> None:
    # High-conviction stat vs low-P(event) event_driven: full_conviction should
    # weight the stat much higher than the event_driven.
    high_conv_stat = _stat_pos(ret=0.10, sizing=1.0)
    low_p_event = _evt_pos(ret=0.10, sizing=0.5, p_event=0.05, conf=0.5)
    assert high_conv_stat.weight_full_conviction > low_p_event.weight_full_conviction
    # 1.0 vs 0.5 * 0.05 * 0.5 = 0.0125
    assert low_p_event.weight_full_conviction == pytest.approx(0.0125)
