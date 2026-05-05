"""Print the current ledger quality report + memory-vs-cold comparison.

No LLM calls, no API quota burned — just reads the SQLite ledger.
Use after a bootstrap batch (or at any point) to see where the system stands.
"""
from __future__ import annotations

from pathlib import Path

from llm_trade_lab.eval.quality import compare_with_without_memory, compute_quality_report
from llm_trade_lab.memory.ledger import Ledger

LEDGER_PATH = Path("data/ledger.db")


def main() -> None:
    if not LEDGER_PATH.exists():
        print(f"No ledger at {LEDGER_PATH}. Run a generation script first.")
        return
    ledger = Ledger(LEDGER_PATH)

    print("=" * 60)
    print(compute_quality_report(ledger).render())

    print("\n" + "=" * 60)
    print("Memory vs cold-start comparison")
    cmp = compare_with_without_memory(ledger)
    for label, key in (("With memory  ", "with_memory"), ("Cold start   ", "cold_start")):
        b = cmp[key]
        n_h = b["n_hypotheses"]
        n_r = b["n_results"]
        if not n_r:
            print(f"  {label}  n_hyp={n_h:>3d}  (no backtest results)")
            continue
        print(
            f"  {label}  n_hyp={n_h:>3d}  n_results={n_r:>3d}  "
            f"mean_return={b['mean_return']:+.2%}  median={b['median_return']:+.2%}  "
            f"win_rate={b['win_rate']:.1%}"
        )

    if cmp["with_memory"]["n_hypotheses"] < 20 or cmp["cold_start"]["n_hypotheses"] < 5:
        print("\n  Note: comparison sample is small; treat directional reads as noisy")
        print("  until both buckets have 20+ hypotheses with realized outcomes.")


if __name__ == "__main__":
    main()
