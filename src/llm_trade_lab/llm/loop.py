"""RAG-with-memory loop driver.

Per-hypothesis flow:
  pick context -> retrieve top-k similar past hypotheses + outcomes from ledger
  -> generate via LLM with memory injected -> validate -> insert ledger
  -> add to FAISS -> backtest universe -> log results

Lineage: each generated hypothesis records `retrieved_hypothesis_ids` in its
`generation_params` so we can later answer "did the retrieved memory actually
shape this hypothesis, and did memory-conditioned generations outperform
cold-start ones?"
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from llm_trade_lab.backtest.engine import BacktestResult, run_backtest
from llm_trade_lab.data.ticker_health import THEME_SECTOR_ETF, compute_ticker_health
from llm_trade_lab.data.yfinance_ingest import fetch_ohlcv
from llm_trade_lab.events.event_stream import Event, collect_events
from llm_trade_lab.llm.client import LLMClient, default_client
from llm_trade_lab.llm.generator import (
    HypothesisGenerationError,
    generate_event_driven_hypothesis,
    generate_statistical_hypothesis,
)
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.memory.retriever import HypothesisRetriever
from llm_trade_lab.schema.hypothesis import (
    EventDrivenHypothesis,
    Hypothesis,
    StatisticalHypothesis,
)

logger = logging.getLogger(__name__)


@dataclass
class LoopRun:
    """One iteration through the loop. `hypothesis` is None on generation failure."""

    hypothesis_id: str
    hypothesis: Hypothesis | None
    retrieved_ids: list[str] = field(default_factory=list)
    backtest_results: dict[str, BacktestResult] = field(default_factory=dict)
    error: str | None = None


# ----------------------------------------------------------- helpers

def _format_outcome(ledger: Ledger, hypothesis_id: str) -> str:
    """Compact one-line summary of a past hypothesis's realized outcomes."""
    rows = ledger.query_results(hypothesis_id)
    if not rows:
        return "no realized outcome yet"
    rets = [r["return_pct"] for r in rows if r["return_pct"] is not None]
    sharpes = [r["sharpe"] for r in rows if r["sharpe"] is not None]
    if not rets:
        return f"{len(rows)} backtests recorded but no return data"
    avg_ret = sum(rets) / len(rets)
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
    wins = sum(1 for r in rets if r > 0)
    return (
        f"avg_return={avg_ret:+.2%}, avg_sharpe={avg_sharpe:+.2f} "
        f"across {len(rets)} ticker(s), {wins} winners"
    )


def _retrieve_memory(
    retriever: HypothesisRetriever,
    ledger: Ledger,
    *,
    query: str,
    k: int,
    filter_type: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Top-k retrieved past hypotheses formatted as memory entries + their IDs.

    Diversified by primary ticker: oversample candidates from FAISS, then keep
    at most one per primary ticker. This combats mode collapse where retrieval
    keeps surfacing many variants of the same idea (e.g., 3 NVDA SMA-cross
    refinements) and the LLM keeps refining locally instead of exploring.
    """
    if k <= 0:
        return [], []
    # Oversample 5x then dedupe by primary ticker.
    raw = retriever.search(query, k=k * 5, filter_type=filter_type)
    seen_tickers: set[str] = set()
    memory: list[dict[str, Any]] = []
    ids: list[str] = []
    for hid, score, h in raw:
        primary = h.universe[0].upper() if h.universe else ""
        if primary and primary in seen_tickers:
            continue
        if primary:
            seen_tickers.add(primary)
        memory.append(
            {
                "name": h.name,
                "thesis": h.thesis_text,
                "outcome": _format_outcome(ledger, hid),
                "similarity": score,
            }
        )
        ids.append(hid)
        if len(memory) >= k:
            break
    return memory, ids


# Walk-forward windows for statistical-track backtesting.
# Four non-overlapping 5-month windows covering 2024-07 -> 2026-04. Each
# hypothesis is backtested in every window, so we test the *strategy* across
# different market regimes rather than testing whether the universe rallied
# over a single 22-month bull run.
DEFAULT_WALK_FORWARD_WINDOWS: list[tuple[str, str]] = [
    ("2024-07-01", "2024-12-01"),
    ("2024-12-01", "2025-05-01"),
    ("2025-05-01", "2025-10-01"),
    ("2025-10-01", "2026-04-01"),
]


def _backtest_universe(
    hypothesis: Hypothesis,
    hypothesis_id: str,
    ledger: Ledger,
    *,
    window_start: str,
    window_end: str,
) -> dict[str, BacktestResult]:
    out: dict[str, BacktestResult] = {}
    for ticker in hypothesis.universe:
        try:
            data = fetch_ohlcv(ticker, start=window_start, end=window_end)
        except Exception as e:
            logger.warning("%s: data fetch failed: %s", ticker, e)
            continue
        if len(data) < 3:
            logger.warning("%s: insufficient data (%d bars)", ticker, len(data))
            continue
        try:
            result = run_backtest(hypothesis, data)
        except Exception as e:
            logger.warning("%s: backtest failed: %s", ticker, e)
            continue
        ledger.insert_backtest_result(
            hypothesis_id=hypothesis_id,
            universe_ticker=ticker,
            window_start=window_start,
            window_end=window_end,
            result=result,
        )
        out[ticker] = result
    return out


def _backtest_universe_walk_forward(
    hypothesis: Hypothesis,
    hypothesis_id: str,
    ledger: Ledger,
    *,
    windows: list[tuple[str, str]],
    min_bars: int = 30,
) -> dict[str, BacktestResult]:
    """Run the strategy on each (ticker, window) pair, log every result.

    Returns a dict keyed by "{ticker}__{window_start}" so callers can summarize.
    Windows producing fewer than `min_bars` of data are skipped.
    """
    out: dict[str, BacktestResult] = {}
    for ticker in hypothesis.universe:
        for window_start, window_end in windows:
            try:
                data = fetch_ohlcv(ticker, start=window_start, end=window_end)
            except Exception as e:
                logger.warning(
                    "%s [%s..%s]: data fetch failed: %s", ticker, window_start, window_end, e
                )
                continue
            if len(data) < min_bars:
                logger.debug(
                    "%s [%s..%s]: only %d bars, skipping (need %d)",
                    ticker, window_start, window_end, len(data), min_bars,
                )
                continue
            try:
                result = run_backtest(hypothesis, data)
            except Exception as e:
                logger.warning(
                    "%s [%s..%s]: backtest failed: %s", ticker, window_start, window_end, e
                )
                continue
            ledger.insert_backtest_result(
                hypothesis_id=hypothesis_id,
                universe_ticker=ticker,
                window_start=window_start,
                window_end=window_end,
                result=result,
            )
            out[f"{ticker}__{window_start}"] = result
    return out


def _enrich_event_candidates(suggested_beneficiaries: list[Any]) -> list[dict[str, Any]]:
    """Compute ticker health for each candidate; package with theme context.

    Best-effort: if ticker health computation fails (delisted, no data), the
    candidate still flows through with an empty `flags` list.
    """
    enriched: list[dict[str, Any]] = []
    for b in suggested_beneficiaries:
        sector_etf = THEME_SECTOR_ETF.get(getattr(b, "matched_theme", "") or "")
        try:
            health = compute_ticker_health(b.ticker, sector_etf=sector_etf)
            flags = health.flags()
        except Exception as e:
            logger.warning("ticker_health failed for %s: %s", b.ticker, e)
            flags = []
        enriched.append(
            {
                "ticker": b.ticker,
                "name": b.name,
                "mechanism": b.mechanism,
                "confidence": b.confidence,
                "matched_theme": b.matched_theme,
                "current_context": getattr(b, "current_context", ""),
                "flags": flags,
            }
        )
    return enriched


def _ticker_flags_for(tickers: list[str]) -> dict[str, list[str]]:
    """Compute auto-flags for a list of tickers (no theme context — statistical
    track has no theme anchor). Returns empty list per ticker on fetch failure."""
    out: dict[str, list[str]] = {}
    for t in tickers:
        try:
            out[t.upper()] = compute_ticker_health(t).flags()
        except Exception as e:
            logger.warning("ticker_health failed for %s: %s", t, e)
            out[t.upper()] = []
    return out


_DEFAULT_RESOLVER = None  # cached after first lazy load


def _get_default_resolver():
    """Lazy-load entity resolver for sector-context lookups in the statistical track."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        from llm_trade_lab.data.entity_resolver import load_default

        _DEFAULT_RESOLVER = load_default()
    return _DEFAULT_RESOLVER


def _sector_contexts_for(tickers: list[str]) -> dict[str, list[dict[str, str]]]:
    """For each ticker, return any (theme, context) entries from entities.yaml
    where the ticker appears AND the theme has a non-empty current_context.

    This is what bridges the statistical track to the curated-narrative layer:
    when a universe_hint ticker (say ELV) appears in the healthcare_insurance
    theme and that theme has a current_context note, the LLM sees the narrative
    just like the event-driven track does.
    """
    try:
        resolver = _get_default_resolver()
    except Exception as e:
        logger.warning("entity resolver load failed: %s", e)
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    for t in tickers:
        matches = resolver.lookup_ticker(t)
        if matches:
            out[t.upper()] = [{"theme": theme, "context": ctx} for theme, ctx in matches]
    return out


def _seen_event_doc_ids(ledger: Ledger) -> set[str]:
    """Set of trigger_event.doc_ids already represented by an event_driven hypothesis."""
    seen: set[str] = set()
    for _, h in ledger.iter_hypotheses(hypothesis_type="event_driven"):
        if isinstance(h, EventDrivenHypothesis):
            seen.add(h.trigger_event.doc_id)
    return seen


# ----------------------------------------------------------- single-run

def generate_statistical_with_memory(
    *,
    universe_hint: list[str],
    ledger: Ledger,
    retriever: HypothesisRetriever,
    client: LLMClient | None = None,
    today: str | None = None,
    k: int = 3,                 # smaller default: memory should be suggestive, not dominant
    walk_forward_windows: list[tuple[str, str]] | None = None,
    temperature: float = 0.9,   # higher default: encourages exploration vs. mode collapse
) -> LoopRun:
    """One statistical-track iteration: retrieve memory -> generate -> backtest
    across walk-forward windows -> log.

    Each (ticker, window) pair produces a separate backtest_result row so the
    eval harness sees the strategy tested across multiple regimes, not a single
    multi-year buy-and-hold proxy.
    """
    client = client or default_client()
    today_str = today or date.today().isoformat()
    windows = walk_forward_windows if walk_forward_windows is not None else DEFAULT_WALK_FORWARD_WINDOWS

    query = (
        f"statistical trading hypothesis for {', '.join(universe_hint)} "
        f"as of {today_str}"
    )
    memory, retrieved_ids = _retrieve_memory(
        retriever, ledger, query=query, k=k, filter_type="statistical"
    )

    ticker_flags = _ticker_flags_for(universe_hint)
    sector_contexts = _sector_contexts_for(universe_hint)

    try:
        h = generate_statistical_hypothesis(
            universe_hint=universe_hint,
            ticker_flags=ticker_flags,
            sector_contexts=sector_contexts,
            today=today_str,
            retrieved_memory=memory,
            client=client,
            temperature=temperature,
        )
    except HypothesisGenerationError as e:
        return LoopRun(
            hypothesis_id="", hypothesis=None, retrieved_ids=retrieved_ids, error=str(e)
        )

    h.generation_params["retrieved_hypothesis_ids"] = retrieved_ids
    h.generation_params["retrieved_k"] = len(retrieved_ids)
    h.generation_params["walk_forward_windows"] = [list(w) for w in windows]
    h.generation_params["ticker_flags"] = ticker_flags
    h.generation_params["sector_contexts_keys"] = sorted(sector_contexts.keys())

    hid = ledger.insert_hypothesis(h)
    retriever.add_hypothesis(hid, h)
    bt = _backtest_universe_walk_forward(h, hid, ledger, windows=windows)
    return LoopRun(
        hypothesis_id=hid,
        hypothesis=h,
        retrieved_ids=retrieved_ids,
        backtest_results=bt,
    )


def generate_event_driven_with_memory(
    *,
    event: Event,
    ledger: Ledger,
    retriever: HypothesisRetriever,
    client: LLMClient | None = None,
    today: date | None = None,
    k: int = 5,
    horizon_buffer_days: int = 7,
    temperature: float = 0.7,
) -> LoopRun:
    """One event-driven iteration: retrieve memory -> generate -> backtest -> log."""
    client = client or default_client()
    today = today or date.today()

    query = (
        f"{event.title}\n"
        f"themes: {', '.join(event.matched_themes)}\n"
        f"{event.body[:400]}"
    )
    memory, retrieved_ids = _retrieve_memory(
        retriever, ledger, query=query, k=k, filter_type="event_driven"
    )

    enriched_candidates = _enrich_event_candidates(event.suggested_beneficiaries)

    try:
        h = generate_event_driven_hypothesis(
            event=event,
            candidate_beneficiaries=enriched_candidates,
            retrieved_memory=memory,
            client=client,
            temperature=temperature,
        )
    except HypothesisGenerationError as e:
        return LoopRun(
            hypothesis_id="", hypothesis=None, retrieved_ids=retrieved_ids, error=str(e)
        )

    h.generation_params["retrieved_hypothesis_ids"] = retrieved_ids
    h.generation_params["retrieved_k"] = len(retrieved_ids)
    h.generation_params["candidate_flags"] = {
        c["ticker"]: c.get("flags", []) for c in enriched_candidates
    }

    hid = ledger.insert_hypothesis(h)
    retriever.add_hypothesis(hid, h)

    event_dt = h.trigger_event.event_date
    window_start = event_dt.isoformat()
    window_end = min(
        event_dt + timedelta(days=h.expected_horizon_days + horizon_buffer_days),
        today,
    ).isoformat()
    bt = _backtest_universe(
        h, hid, ledger, window_start=window_start, window_end=window_end
    )
    return LoopRun(
        hypothesis_id=hid,
        hypothesis=h,
        retrieved_ids=retrieved_ids,
        backtest_results=bt,
    )


# ----------------------------------------------------------- batch

def run_event_driven_batch(
    *,
    n: int,
    ledger: Ledger,
    retriever: HypothesisRetriever,
    client: LLMClient | None = None,
    since_days: int = 180,
    fr_agencies: list[str] | None = None,
    bill_congress: int | None = 119,
    skip_seen: bool = True,
    save_every: int = 5,
    temperature: float = 0.7,
    k: int = 5,
    min_realized_days: int = 7,
    limit_per_source: int = 100,
) -> list[LoopRun]:
    """Pull recent events from the unified stream and run up to `n` event-driven
    iterations (oldest-first so realized outcomes accumulate).

    Args:
        min_realized_days: excludes events too recent to backtest meaningfully —
            we need at least this many calendar days between event_date and today
            so the backtest has ~5+ trading bars of realized data.
        limit_per_source: per-source cap on events fetched. Increase if the
            candidate pool is exhausted by `skip_seen` after prior batches.
    """
    client = client or default_client()
    today = date.today()
    since = (today - timedelta(days=since_days)).isoformat()
    until = today.isoformat()

    events = collect_events(
        since=since,
        until=until,
        bill_congress=bill_congress,
        fr_agencies=fr_agencies or [
            "food-and-drug-administration",
            "environmental-protection-agency",
            "energy-department",
            "federal-energy-regulatory-commission",
            "securities-and-exchange-commission",
        ],
        limit_per_source=limit_per_source,
    )
    candidates = [e for e in events if e.suggested_beneficiaries]

    # Drop events too recent to have a meaningful realized window.
    realized_cutoff = today - timedelta(days=min_realized_days)
    candidates = [e for e in candidates if e.event_date.date() <= realized_cutoff]

    if skip_seen:
        seen = _seen_event_doc_ids(ledger)
        candidates = [e for e in candidates if e.raw_id not in seen]

    candidates.sort(key=lambda e: e.event_date)  # oldest first

    runs: list[LoopRun] = []
    for i, event in enumerate(candidates[:n]):
        run = generate_event_driven_with_memory(
            event=event,
            ledger=ledger,
            retriever=retriever,
            client=client,
            today=today,
            k=k,
            temperature=temperature,
        )
        runs.append(run)
        if (i + 1) % save_every == 0:
            retriever.save()
    retriever.save()
    return runs


def run_statistical_batch(
    *,
    n: int,
    universes: list[list[str]],
    ledger: Ledger,
    retriever: HypothesisRetriever,
    client: LLMClient | None = None,
    save_every: int = 5,
    temperature: float = 0.9,   # higher default for statistical track to combat mode collapse
    k: int = 3,                 # smaller default — memory should hint, not dominate
    walk_forward_windows: list[tuple[str, str]] | None = None,
) -> list[LoopRun]:
    """Run `n` statistical-track iterations, rotating through `universes`."""
    client = client or default_client()
    runs: list[LoopRun] = []
    for i in range(n):
        universe = universes[i % len(universes)]
        run = generate_statistical_with_memory(
            universe_hint=universe,
            ledger=ledger,
            retriever=retriever,
            client=client,
            k=k,
            temperature=temperature,
            walk_forward_windows=walk_forward_windows,
        )
        runs.append(run)
        if (i + 1) % save_every == 0:
            retriever.save()
    retriever.save()
    return runs
