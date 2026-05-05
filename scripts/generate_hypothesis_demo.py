"""Phase 1 milestone demo: LLM-driven generation of one statistical and one
event-driven hypothesis, end-to-end through the full pipeline.

For each hypothesis: generate via LLM -> validate against schema -> insert
to ledger -> per-ticker backtest -> log results -> index in FAISS ->
sanity-check retrieval.

Requires either:
  - ANTHROPIC_API_KEY in .env  (recommended; free $5 signup credit), OR
  - `ollama serve` running locally with a model pulled (e.g. qwen2.5:7b-instruct)

No retrieved memory passed yet — the loop driver (next milestone) will wire
that in. This script proves the LLM -> Hypothesis -> ledger plumbing works.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from llm_trade_lab.backtest.engine import run_backtest
from llm_trade_lab.data.yfinance_ingest import fetch_ohlcv
from llm_trade_lab.events.event_stream import collect_events
from llm_trade_lab.llm.client import default_client
from llm_trade_lab.llm.generator import (
    HypothesisGenerationError,
    generate_event_driven_hypothesis,
    generate_statistical_hypothesis,
)
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.memory.retriever import HypothesisRetriever

LEDGER_PATH = Path("data/ledger.db")
FAISS_DIR = Path("data/faiss")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _backtest_universe(
    hypothesis, hypothesis_id: str, ledger: Ledger, *, window_start: str, window_end: str
) -> int:
    print(f"  {'ticker':<6s}  {'bars':>4s}  {'return':>9s}  {'sharpe':>7s}  {'maxDD':>8s}")
    n_ok = 0
    for ticker in hypothesis.universe:
        try:
            data = fetch_ohlcv(ticker, start=window_start, end=window_end)
        except Exception as e:
            print(f"  {ticker:<6s}  fetch failed: {type(e).__name__}")
            continue
        if len(data) < 3:
            print(f"  {ticker:<6s}  insufficient data ({len(data)} bars)")
            continue
        try:
            result = run_backtest(hypothesis, data)
        except Exception as e:
            print(f"  {ticker:<6s}  backtest failed: {type(e).__name__}")
            continue
        ledger.insert_backtest_result(
            hypothesis_id=hypothesis_id,
            universe_ticker=ticker,
            window_start=window_start,
            window_end=window_end,
            result=result,
        )
        print(
            f"  {ticker:<6s}  {len(data):>4d}  {result.return_pct:>+8.2%}  "
            f"{result.sharpe:>+7.2f}  {result.max_drawdown_pct:>+7.2%}"
        )
        n_ok += 1
    return n_ok


def main() -> None:
    load_dotenv()
    today = date.today()

    _section("0. LLM client + FAISS retriever")
    try:
        client = default_client()
    except RuntimeError as e:
        print(f"  {e}")
        return
    print(f"  provider: {client.provider}")
    print(f"  model:    {client.model}")

    ledger = Ledger(LEDGER_PATH)

    # Load BGE-small now (before torch/backtesting/Anthropic claim more memory).
    # On Windows this avoids "paging file too small" OSError 1455 later.
    print("  Loading BGE-small embedder for FAISS ...")
    retriever = HypothesisRetriever(ledger, index_dir=FAISS_DIR)
    print(f"  FAISS index size at start: {retriever.size}")

    # ---------- statistical ----------
    _section("1. Generate statistical hypothesis via LLM (no memory yet)")
    try:
        stat = generate_statistical_hypothesis(
            universe_hint=["AAPL", "MSFT", "NVDA", "SPY"],
            today=today.isoformat(),
            client=client,
            temperature=0.7,
        )
    except HypothesisGenerationError as e:
        print(f"  generation failed: {e}")
        if e.raw_text:
            print(f"  raw output: {e.raw_text[:300]}")
        return

    print(f"  name:     {stat.name}")
    print(f"  thesis:   {stat.thesis_text[:200]}")
    print(f"  universe: {stat.universe}")
    print(f"  entry:    {stat.entry_rule.strategy_id} {stat.entry_rule.params}")
    print(f"  sizing:   {stat.sizing}")

    stat_hid = ledger.insert_hypothesis(stat)
    retriever.add_hypothesis(stat_hid, stat)
    print(f"  ledger_id: {stat_hid}  (added to FAISS)")

    print("\n  Backtesting on post-Qwen3-cutoff window 2024-07-01 -> today:")
    _backtest_universe(
        stat, stat_hid, ledger, window_start="2024-07-01", window_end=today.isoformat()
    )

    # ---------- event-driven ----------
    _section("2. Generate event-driven hypothesis via LLM (from real event)")
    print("  Collecting events from unified stream ...")
    events = collect_events(
        since=(today - timedelta(days=180)).isoformat(),
        until=today.isoformat(),
        bill_congress=119,
        fr_agencies=[
            "food-and-drug-administration",
            "environmental-protection-agency",
            "energy-department",
            "federal-energy-regulatory-commission",
        ],
        limit_per_source=30,
    )

    horizon_cutoff = today - timedelta(days=37)
    fully_realized = sorted(
        [e for e in events if e.suggested_beneficiaries and e.event_date.date() <= horizon_cutoff],
        key=lambda e: e.event_date,
        reverse=True,
    )
    partial = sorted(
        [
            e for e in events
            if e.suggested_beneficiaries
            and horizon_cutoff < e.event_date.date() <= today - timedelta(days=7)
        ],
        key=lambda e: e.event_date,
    )
    candidates = fully_realized or partial
    if not candidates:
        print("  No events with theme-matched beneficiaries found in window.")
        return
    chosen = candidates[0]
    is_realized = chosen in fully_realized
    print(f"  Chosen ({'realized' if is_realized else 'partial'}): {chosen.title[:130]}")
    print(f"  source={chosen.source}, date={chosen.event_date.date()}, themes={chosen.matched_themes}")

    try:
        evt = generate_event_driven_hypothesis(
            event=chosen, client=client, temperature=0.7
        )
    except HypothesisGenerationError as e:
        print(f"  generation failed: {e}")
        if e.raw_text:
            print(f"  raw output: {e.raw_text[:300]}")
        return

    print(f"\n  name:          {evt.name}")
    print(f"  thesis:        {evt.thesis_text[:200]}")
    print(f"  universe:      {evt.universe}")
    print(f"  event_prob:    {evt.event_probability:.2f}")
    print(f"  horizon_days:  {evt.expected_horizon_days}")
    print(f"  confounders:   {evt.confounders}")
    print(f"  beneficiaries:")
    for b in evt.beneficiaries:
        print(f"    {b.ticker:<6s} conf={b.confidence:.2f}  {b.mechanism[:90]}")

    evt_hid = ledger.insert_hypothesis(evt)
    retriever.add_hypothesis(evt_hid, evt)
    print(f"  ledger_id: {evt_hid}  (added to FAISS)")

    event_dt = evt.trigger_event.event_date
    window_start = event_dt.isoformat()
    window_end = min(event_dt + timedelta(days=evt.expected_horizon_days + 7), today).isoformat()
    print(f"\n  Backtesting window {window_start} -> {window_end}:")
    _backtest_universe(
        evt, evt_hid, ledger, window_start=window_start, window_end=window_end
    )

    # ---------- FAISS retrieval ----------
    _section("3. Save FAISS index + retrieval sanity check")
    retriever.save()
    print(f"  index size: {retriever.size}")

    print(f"\n  Statistical query (filter=statistical, k=2):")
    for hid, score, h in retriever.search(stat.thesis_text, k=2, filter_type="statistical"):
        print(f"    sim={score:+.3f}  {h.name}")

    print(f"\n  Event-driven query (filter=event_driven, k=2):")
    for hid, score, h in retriever.search(evt.thesis_text, k=2, filter_type="event_driven"):
        print(f"    sim={score:+.3f}  {h.name}")

    _section("Phase 1 milestone: LLM-driven generation working")
    print("  Both hypothesis types generated by the LLM, validated against the")
    print("  schema, backtested on real OHLCV, persisted, and retrievable.")
    print("  Next: RAG-with-memory loop driver (retrieval feeds back into prompts).")


if __name__ == "__main__":
    main()
