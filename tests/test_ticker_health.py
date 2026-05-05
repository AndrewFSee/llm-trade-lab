"""Ticker health flag tests with synthetic price data + monkeypatched fetch."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from llm_trade_lab.data import ticker_health


def _synthetic_close(start_price: float, end_price: float, n: int = 300, peak_at: int | None = None) -> pd.DataFrame:
    """Construct a price series that linearly drifts from start to end,
    optionally with a peak at index `peak_at` (used to engineer drawdown)."""
    if peak_at is not None and 0 < peak_at < n:
        before = np.linspace(start_price, max(start_price, end_price) * 1.5, peak_at)
        after = np.linspace(before[-1], end_price, n - peak_at)
        close = np.concatenate([before, after])
    else:
        close = np.linspace(start_price, end_price, n)
    idx = pd.date_range("2025-04-30", periods=n, freq="B")[::-1].sort_values()
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000},
        index=idx,
    )


@pytest.fixture(autouse=True)
def reset_cache():
    ticker_health.clear_cache()
    yield
    ticker_health.clear_cache()


def test_one_year_return_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _synthetic_close(start_price=100.0, end_price=60.0, n=260)
    monkeypatch.setattr(ticker_health, "fetch_ohlcv", lambda t, start, end: df)
    th = ticker_health.compute_ticker_health("FOO", as_of=date(2026, 4, 28))
    assert th.one_year_return is not None
    assert -0.45 < th.one_year_return < -0.35  # roughly -40%


def test_one_year_return_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _synthetic_close(start_price=100.0, end_price=130.0, n=260)
    monkeypatch.setattr(ticker_health, "fetch_ohlcv", lambda t, start, end: df)
    th = ticker_health.compute_ticker_health("FOO", as_of=date(2026, 4, 28))
    assert th.one_year_return is not None
    assert 0.25 < th.one_year_return < 0.35


def test_drawdown_from_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _synthetic_close(start_price=100.0, end_price=60.0, n=260, peak_at=130)
    monkeypatch.setattr(ticker_health, "fetch_ohlcv", lambda t, start, end: df)
    th = ticker_health.compute_ticker_health("FOO", as_of=date(2026, 4, 28))
    assert th.drawdown_from_peak is not None
    assert th.drawdown_from_peak < -0.4  # significant drawdown


def test_distance_from_200dma_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    # Linear ramp; current price > 200d SMA by definition
    df = _synthetic_close(start_price=100.0, end_price=200.0, n=260)
    monkeypatch.setattr(ticker_health, "fetch_ohlcv", lambda t, start, end: df)
    th = ticker_health.compute_ticker_health("FOO", as_of=date(2026, 4, 28))
    assert th.distance_from_200dma is not None
    assert th.distance_from_200dma > 0


def test_excess_vs_sector(monkeypatch: pytest.MonkeyPatch) -> None:
    ticker_df = _synthetic_close(start_price=100.0, end_price=110.0, n=260)  # +10%
    sector_df = _synthetic_close(start_price=100.0, end_price=130.0, n=260)  # +30%

    def fake_fetch(t, start, end):
        if t.upper() == "FOO":
            return ticker_df
        if t.upper() == "XLV":
            return sector_df
        raise ValueError(t)

    monkeypatch.setattr(ticker_health, "fetch_ohlcv", fake_fetch)
    th = ticker_health.compute_ticker_health("FOO", as_of=date(2026, 4, 28), sector_etf="XLV")
    assert th.excess_vs_sector is not None
    assert -0.25 < th.excess_vs_sector < -0.15  # roughly -20pp underperformance


def test_flags_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _synthetic_close(start_price=100.0, end_price=60.0, n=260, peak_at=130)
    monkeypatch.setattr(ticker_health, "fetch_ohlcv", lambda t, start, end: df)
    th = ticker_health.compute_ticker_health("FOO", as_of=date(2026, 4, 28))
    flag_str = " | ".join(th.flags())
    assert "1y return" in flag_str
    assert "drawdown" in flag_str
    assert "200d SMA" in flag_str


def test_returns_empty_when_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(t, start, end):
        raise ValueError("delisted")

    monkeypatch.setattr(ticker_health, "fetch_ohlcv", fake_fetch)
    th = ticker_health.compute_ticker_health("DEAD", as_of=date(2026, 4, 28))
    assert th.flags() == []
    assert th.one_year_return is None


def test_caches_within_session(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _synthetic_close(start_price=100.0, end_price=110.0, n=260)
    call_count = {"n": 0}

    def fake_fetch(t, start, end):
        call_count["n"] += 1
        return df

    monkeypatch.setattr(ticker_health, "fetch_ohlcv", fake_fetch)
    th1 = ticker_health.compute_ticker_health("FOO", as_of=date(2026, 4, 28))
    th2 = ticker_health.compute_ticker_health("FOO", as_of=date(2026, 4, 28))
    assert th1 is th2  # same cached instance
    assert call_count["n"] == 1
