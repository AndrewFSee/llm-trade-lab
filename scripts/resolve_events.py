"""Scan event-driven hypotheses, resolve their trigger events, log the result.

For each event_driven hypothesis whose trigger_event.source == 'congress_bill',
look up the current status of the bill via Congress.gov and upsert a
resolution row. Skips already-resolved (passed/failed) by default since those
don't change.

Usage:
  uv run python scripts/resolve_events.py
  uv run python scripts/resolve_events.py --recheck       # re-resolve everything
  uv run python scripts/resolve_events.py --limit 20      # cap how many to check

Cost: free (Congress.gov API; ~1 call per hypothesis).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm_trade_lab.events.resolution import resolve_congress_bill
from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.schema.hypothesis import EventDrivenHypothesis

LEDGER_PATH = Path("data/ledger.db")
TERMINAL_STATUSES = {"passed", "failed"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recheck", action="store_true",
                   help="Re-resolve hypotheses already marked passed/failed (terminal)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap how many hypotheses to resolve in this run")
    p.add_argument("--sleep-ms", type=int, default=200,
                   help="Sleep between Congress.gov calls (be a polite client)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    ledger = Ledger(LEDGER_PATH)
    event_hyps = [
        (hid, h)
        for hid, h in ledger.iter_hypotheses(hypothesis_type="event_driven")
        if isinstance(h, EventDrivenHypothesis)
        and h.trigger_event.source == "congress_bill"
    ]
    print(f"Found {len(event_hyps)} event-driven hypotheses with congress_bill triggers.")

    to_resolve = []
    for hid, h in event_hyps:
        existing = ledger.get_event_resolution(hid)
        if existing and existing["resolution_status"] in TERMINAL_STATUSES and not args.recheck:
            continue
        to_resolve.append((hid, h))

    if args.limit is not None:
        to_resolve = to_resolve[: args.limit]
    print(f"Resolving {len(to_resolve)} (skipping terminal unless --recheck).")

    counts: Counter[str] = Counter()
    for i, (hid, h) in enumerate(to_resolve, 1):
        bill_id = h.trigger_event.doc_id
        res = resolve_congress_bill(bill_id)
        ledger.upsert_event_resolution(
            hypothesis_id=hid,
            trigger_event_doc_id=bill_id,
            trigger_event_source=h.trigger_event.source,
            resolution_status=res.status,
            resolution_evidence=res.evidence,
            resolution_date=res.resolution_date.isoformat() if res.resolution_date else None,
            realized_outcome=res.realized_outcome,
            p_event_predicted=h.event_probability,
        )
        counts[res.status] += 1
        print(
            f"  [{i:>3}/{len(to_resolve)}] {bill_id:<20s}  "
            f"P={h.event_probability:.2f}  -> {res.status:<9s}  "
            f"({res.evidence[:60]})"
        )
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print("\nResolution summary:")
    for status, n in counts.most_common():
        print(f"  {status:<10s}  {n}")
    print(f"\nNext: uv run python scripts/calibration_report.py")


if __name__ == "__main__":
    main()
