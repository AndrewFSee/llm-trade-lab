"""RAG-with-memory loop demo / bootstrap runner.

Runs N_STATISTICAL + N_EVENT_DRIVEN generations through the full loop:
  retrieve top-k similar past hypotheses -> generate with memory injected ->
  validate -> backtest universe -> log to ledger -> add to FAISS.

Defaults are tiny (2 + 3) for a quick demo. For a bootstrap batch, run:
  uv run python scripts/rag_loop_demo.py --n-statistical 25 --n-event-driven 25

Cost on claude-haiku-4-5 with prompt caching: ~$0.01-0.02 per generation.
50 generations is roughly $0.50-$1.00.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from llm_trade_lab.eval.quality import compare_with_without_memory, compute_quality_report
from llm_trade_lab.llm.client import default_client
from llm_trade_lab.llm.loop import (
    run_event_driven_batch,
    run_statistical_batch,
)
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.memory.retriever import HypothesisRetriever

LEDGER_PATH = Path("data/ledger.db")
FAISS_DIR = Path("data/faiss")
UNIVERSES = [
    # mega-cap tech
    ["AAPL", "MSFT", "NVDA"],
    ["GOOGL", "META", "AMZN"],
    # broad-index ETFs
    ["SPY", "QQQ", "IWM"],
    # sector ETFs (cycle through three groups for breadth)
    ["XLF", "XLE", "XLK"],     # financials, energy, tech
    ["XLV", "XLP", "XLU"],     # healthcare, staples, utilities
    ["XLY", "XLI", "XLB"],     # discretionary, industrials, materials
    # specific industry buckets (drive the model into less-explored territory)
    ["UNH", "ELV", "HUM"],     # managed care
    ["XOM", "CVX", "COP"],     # oil majors
    ["JPM", "BAC", "WFC"],     # money-center banks
    ["LMT", "RTX", "NOC"],     # defense primes
    ["LLY", "MRK", "PFE"],     # large pharma
    ["TSLA", "F", "GM"],       # autos / EVs
    ["DHI", "LEN", "NVR"],     # homebuilders
    ["DE", "CAT", "AGCO"],     # agriculture / heavy equipment
    ["MAR", "HLT", "RCL"],     # travel & leisure
    ["AMD", "TSM", "AVGO"],    # semiconductors
    ["CRWD", "PANW", "FTNT"],  # cybersecurity
    ["NEE", "DUK", "SO"],      # regulated utilities
    ["FCX", "NEM", "GOLD"],    # metals / miners
    ["V", "MA", "AXP"],        # payments
]


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-statistical", type=int, default=2,
                   help="Number of statistical hypotheses to generate (default 2)")
    p.add_argument("--n-event-driven", type=int, default=3,
                   help="Number of event-driven hypotheses to generate (default 3)")
    p.add_argument("--k", type=int, default=3,
                   help="Top-k retrieval depth from FAISS (default 3 — small enough memory hints, not dominates)")
    p.add_argument("--temperature", type=float, default=0.85,
                   help="LLM sampling temperature (default 0.85 — higher than chat default to combat mode collapse)")
    p.add_argument("--since-days", type=int, default=180,
                   help="Look back this many days for events (default 180)")
    p.add_argument("--min-realized-days", type=int, default=7,
                   help="Skip events more recent than this many days (need ~5+ trading bars to backtest)")
    p.add_argument("--limit-per-source", type=int, default=100,
                   help="Per-source event fetch cap. Increase when skip_seen exhausts the pool.")
    p.add_argument("--no-skip-seen", action="store_true",
                   help="Allow re-generating hypotheses for events already in the ledger")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    _section("0. Setup")
    try:
        client = default_client()
    except RuntimeError as e:
        print(f"  {e}")
        return
    print(f"  provider={client.provider}, model={client.model}")
    ledger = Ledger(LEDGER_PATH)
    print("  Loading FAISS retriever (BGE-small) ...")
    retriever = HypothesisRetriever(ledger, index_dir=FAISS_DIR)
    # Backfill index with any existing ledger rows we haven't indexed yet.
    if retriever.size < len(ledger.iter_hypotheses()):
        n = retriever.reindex_from_ledger()
        retriever.save()
        print(f"  Reindexed {n} existing hypotheses")
    print(f"  ledger hypotheses: {len(ledger.iter_hypotheses())}")
    print(f"  FAISS index size:  {retriever.size}")

    _section("1. Quality report (BEFORE batch)")
    print(compute_quality_report(ledger).render())

    _section(f"2. Run statistical batch (n={args.n_statistical})")
    stat_runs = run_statistical_batch(
        n=args.n_statistical,
        universes=UNIVERSES,
        ledger=ledger,
        retriever=retriever,
        client=client,
        k=args.k,
        temperature=args.temperature,
    )
    for i, run in enumerate(stat_runs, 1):
        if run.error:
            print(f"  [{i}] FAILED: {run.error}")
            continue
        n_bt = len(run.backtest_results)
        rets = [r.return_pct for r in run.backtest_results.values()]
        avg = sum(rets) / len(rets) if rets else 0.0
        print(
            f"  [{i}] {run.hypothesis.name:<32s}  "
            f"retrieved={len(run.retrieved_ids):>1d}  "
            f"backtests={n_bt:>1d}  avg_return={avg:+.2%}"
        )

    _section(f"3. Run event-driven batch (n={args.n_event_driven})")
    evt_runs = run_event_driven_batch(
        n=args.n_event_driven,
        ledger=ledger,
        retriever=retriever,
        client=client,
        since_days=args.since_days,
        skip_seen=not args.no_skip_seen,
        k=args.k,
        temperature=args.temperature,
        min_realized_days=args.min_realized_days,
        limit_per_source=args.limit_per_source,
    )
    for i, run in enumerate(evt_runs, 1):
        if run.error:
            print(f"  [{i}] FAILED: {run.error}")
            continue
        n_bt = len(run.backtest_results)
        rets = [r.return_pct for r in run.backtest_results.values()]
        avg = sum(rets) / len(rets) if rets else 0.0
        evt_prob = run.hypothesis.event_probability  # type: ignore[union-attr]
        print(
            f"  [{i}] {run.hypothesis.name[:36]:<36s}  "
            f"retrieved={len(run.retrieved_ids):>1d}  "
            f"backtests={n_bt:>1d}  avg_return={avg:+.2%}  P(event)={evt_prob:.2f}"
        )

    _section("4. Quality report (AFTER batch)")
    print(compute_quality_report(ledger).render())

    _section("5. Memory vs cold-start comparison")
    cmp = compare_with_without_memory(ledger)
    for label, key in (("With memory   ", "with_memory"), ("Cold start    ", "cold_start")):
        b = cmp[key]
        mr = b["mean_return"]
        wr = b["win_rate"]
        print(
            f"  {label}  n_hyp={b['n_hypotheses']:>2d}  n_results={b['n_results']:>2d}  "
            f"mean_return={mr:+.2%}" if mr is not None else f"  {label}  n_hyp={b['n_hypotheses']:>2d}  (no results)"
        )
        if wr is not None:
            print(f"    win_rate={wr:.1%}")

    _section("Done")
    print("  Bootstrap larger batches with --n-statistical 25 --n-event-driven 25")
    print("  At ~50 hypotheses with realized outcomes, retrieval starts adding signal.")
    print("  Standalone status: uv run python scripts/quality_report.py")


if __name__ == "__main__":
    main()
