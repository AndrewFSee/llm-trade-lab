"""Phase 0 verification target: event-driven end-to-end round trip.

Pulls live events from the unified stream -> picks one with theme-matched
beneficiaries -> hand-builds an EventDrivenHypothesis -> backtests each
beneficiary on a [event_date, event_date+horizon] window -> logs results
to the ledger -> indexes the hypothesis in FAISS -> runs a similarity query.

No LLM yet. This proves both hypothesis types round-trip end-to-end through
every Phase 0 component, which is the gate the plan calls for before
moving to Phase 1's LLM-driven generation.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from llm_trade_lab.backtest.engine import run_backtest
from llm_trade_lab.data.yfinance_ingest import fetch_ohlcv
from llm_trade_lab.events.event_stream import collect_events
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.memory.retriever import HypothesisRetriever
from llm_trade_lab.schema.hypothesis import (
    EntryExitRule,
    EventDrivenHypothesis,
    TriggerEvent,
)

LEDGER_PATH = Path("data/ledger.db")
FAISS_DIR = Path("data/faiss")
HORIZON_DAYS = 30


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    load_dotenv()
    today = date.today()

    # ---------- 1. collect events ----------
    _section("1. Collect events from unified stream")
    since = (today - timedelta(days=180)).isoformat()
    until = today.isoformat()
    print(f"  window {since} -> {until}")
    events = collect_events(
        since=since,
        until=until,
        bill_congress=119,
        fr_agencies=[
            "food-and-drug-administration",
            "environmental-protection-agency",
            "federal-energy-regulatory-commission",
            "energy-department",
            "securities-and-exchange-commission",
        ],
        limit_per_source=50,
    )
    print(f"  -> {len(events)} events collected")

    # ---------- 2. pick a candidate ----------
    _section("2. Pick an event with theme-matched beneficiaries")
    horizon_cutoff = today - timedelta(days=HORIZON_DAYS + 5)
    min_data_cutoff = today - timedelta(days=7)  # need >= ~5 trading days
    # Oldest first so we get the most-realized window for any chosen candidate.
    by_date_asc = sorted(events, key=lambda e: e.event_date)
    fully_realized = [
        e for e in by_date_asc
        if e.suggested_beneficiaries and e.event_date.date() <= horizon_cutoff
    ]
    partial = [
        e for e in by_date_asc
        if e.suggested_beneficiaries
        and horizon_cutoff < e.event_date.date() <= min_data_cutoff
    ]
    candidates = fully_realized or partial

    if not candidates:
        print(
            "  No events with theme-matched beneficiaries old enough to backtest.\n"
            "  Suggestions: expand configs/entities.yaml with more keywords, or\n"
            "  broaden the FR agency list to include more sectors."
        )
        return

    # Prefer the most-recent fully-realized candidate (most relevant context)
    # but fall back to the oldest partial (most data) if nothing is realized.
    chosen = fully_realized[-1] if fully_realized else partial[0]
    is_realized = chosen in fully_realized
    print(f"  Status: {'fully realized window' if is_realized else 'PARTIAL window (event too recent)'}")
    print(f"  Title:    {chosen.title[:140]}")
    print(f"  Source:   {chosen.source}")
    print(f"  Date:     {chosen.event_date.date()}")
    print(f"  Themes:   {', '.join(chosen.matched_themes) or '-'}")
    print(
        "  Beneficiaries: "
        + ", ".join(f"{b.ticker}({b.confidence:.2f})" for b in chosen.suggested_beneficiaries)
    )

    # ---------- 3. build hypothesis ----------
    _section("3. Build EventDrivenHypothesis")
    hypothesis = EventDrivenHypothesis(
        name=f"e2e_{chosen.source}_{chosen.raw_id[:24]}",
        thesis_text=(
            f"Event ({chosen.source}, {chosen.event_date.date()}): {chosen.title[:200]}. "
            f"Theme(s): {', '.join(chosen.matched_themes) or 'unspecified'}. "
            f"Hypothesis: long named beneficiaries for {HORIZON_DAYS} days starting at event publication."
        ),
        universe=[b.ticker for b in chosen.suggested_beneficiaries],
        entry_rule=EntryExitRule(strategy_id="event_hold", params={}),
        exit_rule=EntryExitRule(strategy_id="event_hold", params={}),
        holding_period_days=HORIZON_DAYS,
        sizing=1.0,
        generated_at=datetime.now(timezone.utc),
        model_version_hash="hand_written_v0",
        trigger_event=TriggerEvent(
            source=chosen.source,
            doc_id=chosen.raw_id,
            event_type=chosen.event_type[:80] or "event",
            event_date=chosen.event_date.date(),
        ),
        beneficiaries=[b.to_schema() for b in chosen.suggested_beneficiaries],
        event_probability=0.5,
        expected_horizon_days=HORIZON_DAYS,
        confounders=["broad market drawdown", "beneficiary-specific earnings reactions"],
    )
    print(f"  name:    {hypothesis.name}")
    print(f"  universe: {hypothesis.universe}")

    # ---------- 4. log + per-ticker backtest ----------
    _section("4. Per-beneficiary backtest + ledger insert")
    ledger = Ledger(LEDGER_PATH)
    hid = ledger.insert_hypothesis(hypothesis)
    print(f"  hypothesis_id: {hid}")

    event_dt = hypothesis.trigger_event.event_date
    window_start = event_dt.isoformat()
    window_end = min(event_dt + timedelta(days=HORIZON_DAYS + 7), today).isoformat()
    print(f"  data window: {window_start} -> {window_end}")
    print()
    print(f"  {'ticker':<6s}  {'bars':>4s}  {'return':>10s}  {'sharpe':>7s}  {'maxDD':>8s}")
    n_ok = 0
    for ticker in hypothesis.universe:
        try:
            data = fetch_ohlcv(ticker, start=window_start, end=window_end)
        except Exception as e:
            print(f"  {ticker:<6s}  fetch failed: {type(e).__name__}: {e}")
            continue
        if len(data) < 3:
            print(f"  {ticker:<6s}  insufficient data ({len(data)} bars)")
            continue
        try:
            result = run_backtest(hypothesis, data)
        except Exception as e:
            print(f"  {ticker:<6s}  backtest failed: {type(e).__name__}: {e}")
            continue
        ledger.insert_backtest_result(
            hypothesis_id=hid,
            universe_ticker=ticker,
            window_start=window_start,
            window_end=window_end,
            result=result,
        )
        print(
            f"  {ticker:<6s}  {len(data):>4d}  {result.return_pct:>+9.2%}  "
            f"{result.sharpe:>+7.2f}  {result.max_drawdown_pct:>+7.2%}"
        )
        n_ok += 1

    if n_ok == 0:
        print("\n  No backtests succeeded; skipping FAISS step.")
        return
    print(f"\n  -> {n_ok}/{len(hypothesis.universe)} backtests logged")

    # ---------- 5. FAISS index + retrieval sanity check ----------
    _section("5. Index in FAISS + similarity query")
    print("  Loading sentence-transformers (BGE-small) and indexing ...")
    retriever = HypothesisRetriever(ledger, index_dir=FAISS_DIR)
    retriever.add_hypothesis(hid, hypothesis)
    retriever.save()
    print(f"  index size: {retriever.size}  (this hypothesis only; reindex_from_ledger() picks up earlier ones)")

    print("\n  Query with the hypothesis's own thesis text (expect self as top hit):")
    results = retriever.search(hypothesis.thesis_text, k=3, filter_type="event_driven")
    for rid, score, h in results:
        print(f"    sim={score:+.3f}  [{h.type:<14s}]  {h.name[:60]}")

    _section("Phase 0 verification: COMPLETE")
    print("  Both hypothesis types now round-trip:")
    print("    - statistical:    scripts/end_to_end_demo.py")
    print("    - event_driven:   scripts/event_driven_e2e_demo.py  (this script)")
    print("  Ready for Phase 1: LLM client + prompts + RAG-with-memory loop.")


if __name__ == "__main__":
    main()
