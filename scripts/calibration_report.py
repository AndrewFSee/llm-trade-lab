"""Score the LLM's event_probability calibration against realized resolutions.

Reads the event_resolution table populated by resolve_events.py. Computes:
  - Brier score   (lower is better; perfect = 0, naive 0.5 = 0.25)
  - ECE           (lower is better; perfect calibration = 0)
  - Reliability bins  (mean predicted vs realized rate, per bin)
  - Per-bucket counts (passed/advanced/stalled/failed/pending)

Only resolutions with realized_outcome != NULL contribute to scoring.
Pending/unknown are tallied but excluded from Brier/ECE.

Usage:
  uv run python scripts/calibration_report.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm_trade_lab.eval.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from llm_trade_lab.memory.ledger import Ledger

LEDGER_PATH = Path("data/ledger.db")


def main() -> None:
    ledger = Ledger(LEDGER_PATH)
    rows = ledger.iter_event_resolutions()
    if not rows:
        print("No event resolutions in the ledger. Run scripts/resolve_events.py first.")
        return

    print(f"Total event resolutions: {len(rows)}")
    print()

    # Status distribution
    print("By resolution status:")
    counts = Counter(r["resolution_status"] for r in rows)
    for status, n in counts.most_common():
        pct = 100 * n / len(rows)
        print(f"  {status:<10s}  {n:>4d}  ({pct:>5.1f}%)")
    print()

    # Filter to resolved (realized_outcome not None) for scoring
    scored = [r for r in rows if r["realized_outcome"] is not None]
    if not scored:
        print("No resolved hypotheses yet (all pending/unknown). Wait or use --recheck.")
        return

    predicted = [r["p_event_predicted"] for r in scored]
    outcomes = [r["realized_outcome"] for r in scored]

    print(f"Resolved (Brier-eligible): {len(scored)} / {len(rows)}")
    print()
    print("Calibration scores:")
    print(f"  Brier score:                 {brier_score(predicted, outcomes):.4f}")
    print(f"  Expected calibration error:  {expected_calibration_error(predicted, outcomes, n_bins=5):.4f}")
    print()

    # Naive baselines for sanity
    base_rate = sum(outcomes) / len(outcomes)
    print(f"Realized base rate:            {base_rate:.2%}")
    naive_brier = brier_score([base_rate] * len(outcomes), outcomes)
    print(f"  Brier of constant base rate: {naive_brier:.4f}  (lower than ours? we're worse than constant)")
    print()

    # Reliability bins
    bins = reliability_curve(predicted, outcomes, n_bins=5)
    if bins:
        print("Reliability curve (5 bins):")
        print(f"  {'predicted':>12s}  {'realized':>12s}  {'count':>6s}")
        for b in bins:
            print(
                f"  {b.mean_predicted:>12.2%}  {b.fraction_positive:>12.2%}  {b.count:>6d}"
            )

    print()
    print("Notes:")
    print("- 'advanced' is treated as outcome=1 (event happened in the predicted direction).")
    print("- 'stalled'/'failed' are outcome=0.")
    print("- 'pending' bills haven't resolved yet — re-run scripts/resolve_events.py later.")


if __name__ == "__main__":
    main()
