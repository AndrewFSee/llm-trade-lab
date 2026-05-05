# llm-trade-lab — STATUS

State of the project as of late Phase 1. This is a hobby experiment in fine-tuning
an LLM to generate trading hypotheses, currently at the **RAG-with-memory loop**
milestone, before any actual fine-tuning. The honest verdict is up-front: the
system isn't producing alpha, it's producing well-reasoned, calibrated, traceable
hypotheses that underperform SPY by a structural ~2-3% on average in a bull
regime. That's the expected baseline for "long-only individual stock picks vs.
broad index." But the *instrument* is working, and the corpus we've built is a
real basis for either continued iteration or a Phase 2 fine-tuning attempt.

## What was built

```
                   ┌───────────────────────────┐
                   │   Unified Event Stream    │
                   │  (6 ingest sources →      │
                   │   normalized Event objs)  │
                   └─────────────┬─────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │  Entity Resolver (42 themes,│
                  │  10 with curated context)   │
                  └──────────────┬──────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
 Statistical track     Event-driven track          FAISS retriever
 (universe rotation,   (event candidates,           (BGE-small,
  ticker_health flags, current_context narrative,    type-aware,
  sector_context)      auto-flags, anti-shoehorn)    ticker-deduped)
        │                        │                        │
        └────────────┬───────────┴────────────────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  Claude Haiku 4.5 (LLM)   │
         │  Prompt-cached system     │
         │  Hypothesis generator     │
         └─────────────┬─────────────┘
                       │
                       ▼
         ┌───────────────────────────┐
         │  Hypothesis (validated)   │
         │  → SQLite ledger          │
         │  → FAISS index            │
         │  → walk-forward backtest  │
         │     (4 windows × tickers) │
         └───────────────────────────┘
```

206 hypotheses logged, 640 backtest_result rows, 95% testability rate, 10 prompt
versions (`p1` → `p7` plus `hand_written_v0`).

## Data sources

| Source | Purpose | Key module |
|---|---|---|
| yfinance | Daily OHLCV for backtesting + ticker health flags | `data/yfinance_ingest.py` |
| SEC EDGAR (8-K, 10-K, 10-Q) | Material-event filings as catalysts | `data/edgar_ingest.py` |
| SEC Form 4 | Insider buy/sell transactions | `data/form4_ingest.py` |
| Congress.gov | Bill text + sponsor/cosponsor metadata | `data/congress_ingest.py` |
| Federal Register | Agency rules / FDA approvals / EPA actions | `data/federal_register_ingest.py` |
| FRED | Macro time series + release calendar | `data/fred_ingest.py` |
| Alpha Vantage | News with pre-tagged per-ticker sentiment | `data/alphavantage_news_ingest.py` |

Reddit ingest exists but is dormant — Reddit pushed devs to Devvit in 2026 and
StockTwits is gated behind Cloudflare. Alpha Vantage is the live sentiment layer.

## Key components

### Entity resolver (42 themes, 10 with curated context)
`configs/entities.yaml` is hand-curated. Each theme has a keyword list,
beneficiary tickers with mechanism + prior confidence, and optionally a
`current_context` narrative — a hand-written paragraph about ongoing sector
dynamics the LLM may not know post-cutoff.

The 10 themes with curated context cover the highest-value blind spots:
healthcare_insurance, homebuilders, ev, cannabis, semiconductors, ai_infra,
banks_large, regional_banks, crypto_exposure, nuclear.

### Walk-forward backtesting
Statistical hypotheses are tested across **4 non-overlapping 5-month windows**
covering 2024-07 → 2026-04. This unbiased the test from "did this ticker rip
over 22 months" to "does this strategy add edge across regimes." The single
biggest signal-quality improvement — closed the mean-excess-vs-SPY gap from
~-8% to ~-3% by itself.

### Hybrid ticker context
For every ticker in scope (statistical universe or event-driven beneficiary),
two layers of information are injected into the prompt:

1. **Auto-flags** (computed from cached price data, no judgment thresholds):
   `1y return`, `drawdown from 1y peak`, `1y excess vs sector ETF`, `% vs 200d SMA`.
2. **Curated `current_context`** (hand-written narrative for themes where the LLM
   has stale info): paragraph-form sector dynamics, headwinds, valuation regime.

The model gets both as facts and is required to engage with them in `thesis_text`
rather than ignore them.

### FAISS retriever (memory)
sentence-transformers (BGE-small, 384-dim) over the hypothesis ledger.
Type-aware (filter by `statistical` vs `event_driven`) and ticker-deduped on
retrieval (top-k oversampled then deduped to surface diverse memory rather
than 5 NVDA refinements). Lineage tracked in `generation_params.retrieved_hypothesis_ids`.

### Calibrated event probabilities
Event-driven hypotheses must commit to an `event_probability` (P that the bill
passes / rule finalizes / approval granted). The model anchors low for
introduced bills (~0.05-0.15), high for already-published rules (~0.85-0.95),
and produces explicit confounders. Calibration discipline is enforced via the
prompt's anti-shoehorn block: peripheral theme matches must use sizing < 0.15
and P(event) < 0.10 instead of fabricating mechanisms.

## Prompt evolution

| Version | Key change | Effect |
|---|---|---|
| p1 | Initial baseline | Mode collapse on NVDA buy-and-hold |
| p2 | Added DIVERSITY REQUIREMENT block | Some diversity but still bollinger-monoculture |
| p3 | Added `event_hold` strategy ban from statistical track | Forced real signal-based strategies |
| p4 | Walk-forward windows (engine change, not prompt) | **-8% → -3% mean excess** (largest single win) |
| p5 | Hybrid context layer 1: event-driven gets auto-flags + sector context | Model cites "MP +170%, USAR +147% YTD" verbatim |
| p6 | Hybrid context extended to statistical track via `lookup_ticker` reverse mapping | Auto-flags fully integrated; narrative underused |
| p7 | SECTOR CONTEXT ENGAGEMENT block: thesis_text MUST cite specific phrases from context | "NIM compression headwinds", "CMS rate-cut pressure" appear verbatim in theses |

## Empirical results

### Aggregate journey

| Cohort | n | mean RAW | mean EXCESS vs SPY | win rate vs SPY |
|---|---|---|---|---|
| pre-walk-forward (p1-p3) | 110 | +14.4% | -8.04% | 33.9% |
| p4 (walk-forward) | 25 | +1.04% | -3.04% | 41.1% |
| p5 (hybrid event-driven) | 13 | n/a | -2.29% | 45.0% |
| p6 (hybrid stat track) | 8 | n/a | -3.01% | 44.4% |
| p7 (engagement-tightened) | 50 | +1.96% | **-2.69%** | 39.9% |

The journey from -8% to -3% is real; the journey from -3% to -2.7% is noise.
**Walk-forward did the structural work; the hybrid context layers improved
hypothesis quality without meaningfully moving the aggregate signal.** That's
consistent with the interpretation that the remaining gap is the long-only
individual-pick baseline in a bull regime — irreducible without short side,
hedged pairs, or conviction-weighted sizing.

### What "well-reasoned" looks like (p7 example)

```
hum_bollinger_reversion_ma_stress
  thesis: "HUM trades -2% vs 200d SMA despite sector-wide CMS rate-cut
           pressure and multi-quarter EPS misses, suggesting oversold
           positioning relative to peers. We hypothesize a mean-reversion
           entry when HUM's close drops below (20-day SMA - 2.0 * 20-day std),
           targeting relief rallies as technical extremes unwind, with
           exit on recovery back to the SMA. This isolates HUM's idiosyncratic
           weakness from UNH/ELV's relative resilience and avoids the
           failed long-momentum trap."
```

In one paragraph: cites two specific phrases from the curated `healthcare_insurance`
context ("CMS rate-cut pressure", "multi-quarter EPS misses"), uses peer-relative
reasoning (HUM vs UNH/ELV), engages with retrieved memory (the prior failed
SMA-cross hypothesis), and builds an explicit contrarian thesis. The trade still
lost (-21.95% in window 1), but the *reasoning* is correct.

## Standout examples

### Wins (event-driven track)

**Mobility Tax Credit Act → US rare earth producers** (Mar-Jun 2025)
- USAR: +97.74% raw, +90.57% excess over 90 days
- MP: +33.28% raw, +26.11% excess
- Mechanism: vehicle tax credit favoring domestic supply chain. Direct mechanism, calibrated low P(event)=0.12, beneficiaries from entity resolver, both stocks rallied hard after the bill advanced.

**Pay Our Military Act → defense primes** (Oct 2025-Feb 2026)
- LMT: +30.77% excess. HII: +49.95% excess. NOC: +15.86% excess.
- 4 of 5 named beneficiaries beat SPY substantially. Sharpe 1.2-2.2.
- Cleanest example of "model picks beneficiaries → catalyst plays out → trade works."

**Sabine Pass LNG re-export approval → natgas producers** (Apr 2026)
- All 5 named tickers (LNG, EQT, AR, RRC, CTRA) beat SPY in the 11-day post-event window. Sharpes 2.7-4.5. Caveat: short window, could be noise.

### Losses (informative)

**Healthcare insurer hypotheses (UNH/HUM/ELV)** — pre-curated-context era
- Multiple hypotheses across p1-p3 generated naive momentum/breakout strategies on UNH/HUM. Returns: -42% to -60% over 22-month windows vs SPY +33%. Excess: -75% to -94%.
- The LLM had no information about the Medicare Advantage rate cuts or the DOJ probe. It saw "healthcare insurance, defensive sector" without the 2024-2025 collapse.
- Fixed structurally by adding `current_context` to `healthcare_insurance` in p5+. Post-fix, model picks ELV/HUM only as deliberate contrarian oversold plays with explicit acknowledgment of the headwinds.

### Anti-shoehorn working

**`second_amendment_resolution_no_mechanism`** (p4)
- The model was offered HRES-339 (gun rights resolution) with rare-earth tickers as candidates (because the bill happened to mention "minerals" in passing). Instead of fabricating a stretched mechanism, the model named the hypothesis "no_mechanism," set MP/USAR confidence to 0.05, P(event)=0.08, and explicitly said "Candidate beneficiaries operate in rare earth mining — sectors with no causal connection to Second Amendment policy. This is a thematic mismatch."
- Trade lost as predicted (-25% to -32% excess). Calibration was correct: it said low conviction, low conviction was right.

## Honest assessment

**What works:**
- Pipeline is solid. Hypotheses are validated, persisted, retrievable, backtested with proper walk-forward.
- Calibration discipline is real. Event probabilities span 0.05-0.95 appropriately based on event type.
- Anti-shoehorn works. Model self-flags low-conviction mismatches.
- Hybrid context engagement works at the prompt level (auto-flags + curated narrative both surface in theses by p7).
- Retrieval prevents mode collapse. Ticker-dedupe at retrieval ensures memory is diverse rather than 5 NVDA refinements.
- Some real signal candidates exist (rare-earth on Mobility Act, defense on Pay Act, natgas on Sabine Pass). Whether they're repeatable or selection bias is unknowable at n=3.

**What doesn't work (and why):**
- Aggregate excess vs SPY plateaus around -3%. This is the structural ceiling for long-only individual stock picks in a bull market without:
  - Short side (we have none)
  - Hedged pairs (we have none)
  - Conviction-weighted sizing across the portfolio (each hypothesis is sized in isolation)
- The model doesn't know about real-world sector crises post-cutoff. The curated `current_context` is a hand-maintenance tax that doesn't generalize.
- Statistical strategies test poorly across walk-forward — most win in some windows and lose in others. Real edge would require strategy *parameters* that work cross-regime, which the LLM doesn't optimize.

**What we explicitly didn't build:**
- Short side / hedged pairs
- Cross-hypothesis portfolio construction (each is sized independently)
- Event-resolution tracking (we never check whether bills actually pass; P(event) is unscored)
- Walk-forward for event-driven track (uses single event_date+horizon window)
- Fine-tuning anything (the original Phase 2 deliverable; deferred until a larger corpus)

## Next steps

In rough priority order:

1. **Event-resolution tracking** — small async check against Congress.gov / Federal Register to mark bills as `passed/failed/stalled`. Closes the calibration loop on `event_probability`. ~2 hours.
2. **More `current_context` entries** as you spot blind spots. Each is a YAML edit. The system shows the model engages with whatever's there.
3. **Cross-hypothesis portfolio sizing** — use `event_probability` × `confidence` × `sizing` to construct a notional portfolio across N concurrent hypotheses. Would let us test whether calibration discipline yields portfolio-level edge even when individual trades don't.
4. **Fine-tune prep** — the corpus is now 206 hypotheses with realized outcomes. Threshold for first SFT pass per the plan is ≥500. Roughly 2-3 more bootstrap rounds away. Cheaper to keep accumulating than to fine-tune now.
5. **Walk-forward for event-driven** — currently only statistical does this. Less natural for events (each has one real date) but could simulate by re-running on shifted windows.

## Reproducibility

```
# Run a 50-hypothesis batch (statistical + event-driven)
uv run python scripts/rag_loop_demo.py --n-statistical 35 --n-event-driven 15 \
    --no-skip-seen --since-days 365 --limit-per-source 200

# Inspect the latest cohort
uv run python scripts/inspect_hypothesis.py --model-version p7 --excess --top 5

# Aggregate quality report
uv run python scripts/quality_report.py
```

State files (gitignored):
- `data/ledger.db` — SQLite hypothesis ledger
- `data/faiss/` — FAISS index + ID list
- `data/cache/` — yfinance / EDGAR / Congress.gov / etc. caches

108 offline tests pass. 5 integration tests opt-in via `uv run pytest -m integration`.
