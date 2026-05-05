# llm-trade-lab

Phased experiment in using LLMs to generate trading hypotheses for US equities,
with the long-term goal of fine-tuning on accumulated outcomes.

- **Original phased plan:** `~/.claude/plans/i-want-to-experiment-agile-crescent.md`
- **Current state + empirical results:** [STATUS.md](STATUS.md)

## Status

**Phase 1 — RAG-with-memory loop is operational.** Anthropic Claude generates
both statistical and event-driven hypotheses with retrieval-augmented memory,
hybrid context (price-action auto-flags + curated sector narrative), and
walk-forward backtesting. 206 hypotheses logged across 7 prompt versions.

Aggregate signal: **-2.7% mean excess return vs SPY** with 40% win rate vs SPY —
the structural ceiling for long-only individual stock picks in a bull regime.
The system isn't producing alpha; it's producing well-reasoned, calibrated,
traceable hypotheses. See [STATUS.md](STATUS.md) for the full empirical journey
and standout examples.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned (matches Colab runtime).

```bash
uv sync
cp .env.example .env  # then fill in your API keys
```

Required keys for the live LLM pipeline:
- `ANTHROPIC_API_KEY` — Claude (free $5 credit at console.anthropic.com)
- `SEC_IDENTITY` — your name + email for SEC EDGAR rate-limit compliance

Optional for richer event ingest:
- `CONGRESS_GOV_API_KEY`, `FRED_API_KEY`, `ALPHAVANTAGE_API_KEY` — all free tier

## Structure

```
src/llm_trade_lab/
├── data/        ingest: yfinance, EDGAR, Form 4, Congress.gov,
│                       Federal Register, FRED, Alpha Vantage,
│                       entity_resolver (configs/entities.yaml),
│                       ticker_health (auto-flags from price data)
├── events/      unified Event stream normalizing all sources
├── schema/      pydantic Hypothesis schema (statistical | event_driven)
├── backtest/    backtesting.py wrapper, strategy registry
├── memory/      SQLite hypothesis ledger + FAISS retriever (BGE-small)
├── llm/         Anthropic + Ollama clients, prompt templates,
│                generator, RAG-with-memory loop driver
└── eval/        Sharpe / Sortino / Brier / ECE / quality reports
```

## Quick start

```bash
# Cheapest way to see the full loop run end-to-end (~$0.30)
uv run python scripts/rag_loop_demo.py --n-statistical 5 --n-event-driven 3

# Look at what the model produced
uv run python scripts/inspect_hypothesis.py --top 3 --excess

# Quality + memory-vs-cold-start report
uv run python scripts/quality_report.py
```

For a larger bootstrap batch:
```bash
uv run python scripts/rag_loop_demo.py --n-statistical 35 --n-event-driven 15 \
    --no-skip-seen --since-days 365 --limit-per-source 200
```

## Caveats baked into the design

- **yfinance has survivorship bias** — only currently-listed tickers.
- **Realistic costs from day one** — 5 bps slippage + commissions baked in.
- **Strict OOS evaluation** — all backtest windows sit after the base LLM's
  knowledge cutoff (mid-2024) to avoid lookahead bias at the model weights.
- **Statistical track uses walk-forward windows** — strategies are tested
  across 4 non-overlapping 5-month regimes, not one 22-month buy-and-hold.
- **Honest mean-Sharpe is misleading** on short event-windows — eval reports
  median Sharpe as primary; mean is shown but flagged as distorted.
- **The aggregate excess vs SPY plateaus around -3%** — that's the structural
  baseline for long-only individual picks vs. broad index in a bull market.

## Tests

```bash
uv run pytest                  # 108 offline tests, no network
uv run pytest -m integration   # 5 live API tests (needs .env keys)
```
