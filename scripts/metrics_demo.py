"""Show all the eval metrics on synthetic data:
  - Two strategies (modest + lucky-best-of-100), with PSR and DSR
  - Three calibration scenarios (well-calibrated, overconfident, underconfident)
"""
from __future__ import annotations

import numpy as np

from llm_trade_lab.eval.metrics import (
    brier_score,
    deflated_sharpe_ratio,
    expected_calibration_error,
    probabilistic_sharpe_ratio,
    reliability_curve,
    sharpe_ratio,
)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    rng = np.random.default_rng(7)

    # ---------- Sharpe / PSR / DSR ----------
    section("Sharpe + PSR + DSR")
    n = 252  # one trading year
    real_edge = rng.normal(0.0008, 0.012, size=n)
    sr_real = sharpe_ratio(real_edge)
    psr_real = probabilistic_sharpe_ratio(sr_real, n, returns=real_edge)
    print(f"  Strategy A (real modest edge):")
    print(f"    n={n}  Sharpe={sr_real:.2f}  PSR(SR>0)={psr_real:.1%}")

    # Now imagine we tested 100 random strategies and picked the best.
    n_trials = 100
    sr_trials = rng.normal(0, 1.0, size=n_trials)  # noise Sharpes around 0
    best_sr = float(sr_trials.max())
    best_returns = rng.normal(best_sr / np.sqrt(252), 0.012, size=n)  # series with that SR
    psr_naive = probabilistic_sharpe_ratio(best_sr, n)
    dsr_corrected = deflated_sharpe_ratio(
        best_sr, n_observations=n, n_trials=n_trials, var_sr_trials=float(sr_trials.var(ddof=1)),
    )
    print(f"  Strategy B (best of {n_trials} pure-noise trials):")
    print(f"    n={n}  best_Sharpe={best_sr:.2f}")
    print(f"    PSR(SR>0)        = {psr_naive:.1%}    <- naive: 'looks great'")
    print(f"    DSR(corrected)   = {dsr_corrected:.1%}    <- after multiple-trials adjustment")
    print(f"    -> DSR drops because best-of-{n_trials}-noise looks like a real edge by chance.")

    # ---------- Calibration ----------
    n_evt = 2000
    base_probs = rng.uniform(0, 1, size=n_evt)

    # well-calibrated: outcome ~ Bernoulli(p)
    y_calib = (rng.uniform(0, 1, size=n_evt) < base_probs).astype(int)
    # overconfident: predict 0.9 when truth is 50/50
    y_overconf = rng.integers(0, 2, size=n_evt)
    p_overconf = np.full(n_evt, 0.9)
    # underconfident: predict 0.55 when truth has clear signal (mostly 1)
    p_underconf = np.full(n_evt, 0.55)
    y_underconf = (rng.uniform(0, 1, size=n_evt) < 0.85).astype(int)

    for label, p, y in (
        ("Well-calibrated     ", base_probs, y_calib),
        ("Overconfident (0.9) ", p_overconf, y_overconf),
        ("Underconfident (0.55)", p_underconf, y_underconf),
    ):
        bs = brier_score(p, y)
        ece = expected_calibration_error(p, y, n_bins=10)
        print(f"  {label}  Brier={bs:.3f}  ECE={ece:.3f}")

    section("Reliability curve (well-calibrated)")
    bins = reliability_curve(base_probs, y_calib, n_bins=10)
    print(f"  {'bin':<14s}  {'mean_pred':>10s}  {'frac_pos':>10s}  {'count':>6s}")
    for b in bins:
        bar = "#" * int(round(b.fraction_positive * 30))
        print(
            f"  [{b.bin_lower:.2f}, {b.bin_upper:.2f})  "
            f"{b.mean_predicted:>10.3f}  {b.fraction_positive:>10.3f}  "
            f"{b.count:>6d}  {bar}"
        )


if __name__ == "__main__":
    main()
