"""Thin wrapper around backtesting.py.

A strategy registry maps `EntryExitRule.strategy_id` strings to backtesting.py
Strategy classes. New strategies register themselves at module import.

Cost defaults: 5 bps commission. backtesting.py models slippage implicitly via
trade-on-next-bar; for explicit slippage modelling, increase commission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from llm_trade_lab.schema.hypothesis import (
    EventDrivenHypothesis,
    StatisticalHypothesis,
)

DEFAULT_COMMISSION = 0.0005  # 5 bps each side
DEFAULT_CASH = 10_000.0


@dataclass
class BacktestResult:
    sharpe: float
    sortino: float
    return_pct: float
    max_drawdown_pct: float
    n_trades: int
    win_rate_pct: float
    cost_bps: float
    cash: float
    raw_stats: dict[str, Any] = field(default_factory=dict)


def _sma(values: pd.Series, n: int) -> pd.Series:
    return pd.Series(values).rolling(n).mean()


class SmaCross(Strategy):
    fast: int = 20
    slow: int = 50

    def init(self) -> None:
        self.sma_fast = self.I(_sma, self.data.Close, self.fast)
        self.sma_slow = self.I(_sma, self.data.Close, self.slow)

    def next(self) -> None:
        if crossover(self.sma_fast, self.sma_slow):
            self.position.close()
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()


class EventHold(Strategy):
    """Buy on the first available bar, hold to end of the supplied data window.

    Designed for event-driven hypotheses: caller slices OHLCV to
    [event_date, event_date + horizon_days] before running, so "hold to end"
    is equivalent to "exit at horizon." DO NOT USE on the statistical track
    over long windows — it degenerates to "buy and hold the universe" and
    produces meaningless returns.
    """

    def init(self) -> None:
        pass

    def next(self) -> None:
        if not self.position:
            self.buy()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI without division-by-zero pathology."""
    series = pd.Series(close)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


class RsiReversion(Strategy):
    """RSI mean-reversion. Buy when RSI(period) drops below `buy_threshold`;
    exit when RSI rises above `exit_threshold`."""

    period: int = 14
    buy_threshold: int = 30
    exit_threshold: int = 70

    def init(self) -> None:
        self.rsi = self.I(_rsi, self.data.Close, self.period)

    def next(self) -> None:
        if not self.position and self.rsi[-1] < self.buy_threshold:
            self.buy()
        elif self.position and self.rsi[-1] > self.exit_threshold:
            self.position.close()


def _rolling_max(values: pd.Series, n: int) -> pd.Series:
    return pd.Series(values).rolling(n).max()


def _rolling_min(values: pd.Series, n: int) -> pd.Series:
    return pd.Series(values).rolling(n).min()


class DonchianBreakout(Strategy):
    """Donchian channel breakout (trend-following).
    Buy when today's close > yesterday's `entry_lookback`-day high;
    exit when today's close < yesterday's `exit_lookback`-day low.
    Uses prior-bar bands to avoid intra-bar lookahead."""

    entry_lookback: int = 20
    exit_lookback: int = 10

    def init(self) -> None:
        self.upper = self.I(_rolling_max, self.data.High, self.entry_lookback)
        self.lower = self.I(_rolling_min, self.data.Low, self.exit_lookback)

    def next(self) -> None:
        if len(self.upper) < 2 or len(self.lower) < 2:
            return
        c = self.data.Close[-1]
        prior_upper = self.upper[-2]
        prior_lower = self.lower[-2]
        if not self.position and c > prior_upper:
            self.buy()
        elif self.position and c < prior_lower:
            self.position.close()


def _rolling_mean(values: pd.Series, n: int) -> pd.Series:
    return pd.Series(values).rolling(n).mean()


def _rolling_std(values: pd.Series, n: int) -> pd.Series:
    return pd.Series(values).rolling(n).std()


class BollingerReversion(Strategy):
    """Bollinger Band mean-reversion.
    Buy when close < (SMA - n_std × std); exit when close > SMA."""

    period: int = 20
    n_std: float = 2.0

    def init(self) -> None:
        close = self.data.Close
        self.sma = self.I(_rolling_mean, close, self.period)
        self.std = self.I(_rolling_std, close, self.period)

    def next(self) -> None:
        if pd.isna(self.std[-1]) or pd.isna(self.sma[-1]):
            return
        c = self.data.Close[-1]
        lower = self.sma[-1] - self.n_std * self.std[-1]
        if not self.position and c < lower:
            self.buy()
        elif self.position and c > self.sma[-1]:
            self.position.close()


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "sma_cross": SmaCross,
    "event_hold": EventHold,
    "rsi_reversion": RsiReversion,
    "donchian_breakout": DonchianBreakout,
    "bollinger_reversion": BollingerReversion,
}


def run_backtest(
    hypothesis: StatisticalHypothesis | EventDrivenHypothesis,
    data: pd.DataFrame,
    *,
    cash: float = DEFAULT_CASH,
    commission: float = DEFAULT_COMMISSION,
) -> BacktestResult:
    """Run a backtest on one ticker's OHLCV using the strategy referenced by
    `hypothesis.entry_rule.strategy_id`.

    Works for both StatisticalHypothesis and EventDrivenHypothesis. For
    multi-ticker hypotheses, call once per ticker and aggregate. For
    event-driven hypotheses, slice `data` to the event window before calling.
    """
    strategy_id = hypothesis.entry_rule.strategy_id
    if strategy_id not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy_id={strategy_id!r}. "
            f"Registered: {sorted(STRATEGY_REGISTRY)}"
        )
    strat_cls = STRATEGY_REGISTRY[strategy_id]

    bt = Backtest(data, strat_cls, cash=cash, commission=commission, finalize_trades=True)
    stats = bt.run(**hypothesis.entry_rule.params)

    return BacktestResult(
        sharpe=float(stats.get("Sharpe Ratio", 0.0) or 0.0),
        sortino=float(stats.get("Sortino Ratio", 0.0) or 0.0),
        return_pct=float(stats.get("Return [%]", 0.0)) / 100.0,
        max_drawdown_pct=float(stats.get("Max. Drawdown [%]", 0.0)) / 100.0,
        n_trades=int(stats.get("# Trades", 0)),
        win_rate_pct=float(stats.get("Win Rate [%]", 0.0) or 0.0) / 100.0,
        cost_bps=commission * 10_000,
        cash=cash,
        raw_stats={k: _jsonable(v) for k, v in stats.items() if not k.startswith("_")},
    )


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)
