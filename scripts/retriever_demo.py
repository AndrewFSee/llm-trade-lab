"""Build a small ledger of hand-written hypotheses, index them with the real
sentence-transformers model, and run a few semantic queries.

First run downloads BAAI/bge-small-en-v1.5 (~134MB) to your HF cache.
"""
from __future__ import annotations

import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.memory.retriever import HypothesisRetriever
from llm_trade_lab.schema.hypothesis import (
    Beneficiary,
    EntryExitRule,
    EventDrivenHypothesis,
    StatisticalHypothesis,
    TriggerEvent,
)


def _stat(name: str, thesis: str, universe: list[str]) -> StatisticalHypothesis:
    return StatisticalHypothesis(
        name=name, thesis_text=thesis, universe=universe,
        entry_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        exit_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 10, "slow": 30}),
        sizing=1.0, generated_at=datetime.now(timezone.utc),
        model_version_hash="demo_v0",
    )


def _evt(name: str, thesis: str, universe: list[str], event_type: str, beneficiaries: list[tuple[str, str, float]]) -> EventDrivenHypothesis:
    return EventDrivenHypothesis(
        name=name, thesis_text=thesis, universe=universe,
        entry_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 5, "slow": 20}),
        exit_rule=EntryExitRule(strategy_id="sma_cross", params={"fast": 5, "slow": 20}),
        sizing=0.5, generated_at=datetime.now(timezone.utc),
        model_version_hash="demo_v0",
        trigger_event=TriggerEvent(
            source="congress_gov", doc_id=f"{name}-doc",
            event_type=event_type, event_date=date(2025, 3, 1),
        ),
        beneficiaries=[Beneficiary(ticker=t, mechanism=m, confidence=c) for t, m, c in beneficiaries],
        event_probability=0.45, expected_horizon_days=60,
    )


HYPOTHESES = [
    _stat("aapl_sma_momentum", "Buy AAPL when 20d SMA crosses above 50d SMA, signaling momentum continuation.", ["AAPL"]),
    _stat("spy_mean_reversion", "Buy SPY when RSI(14) drops below 30, signaling oversold mean-reversion setup.", ["SPY"]),
    _stat("xlf_curve_steepening", "Buy XLF when the 2s10s yield curve steepens by more than 25bps over a quarter; banks earn more on widened NIM.", ["XLF"]),
    _stat("nvda_breakout", "Buy NVDA on volume-confirmed breakouts above 50d high; ride momentum until close below 20d EMA.", ["NVDA"]),
    _evt("farm_bill_fertilizer", "Farm bill includes fertilizer subsidies for corn growers; CF and MOS benefit from increased domestic demand.",
         ["CF", "MOS"], "bill_advanced", [("CF", "nitrogen producer benefits from subsidized corn fertilizer demand", 0.8), ("MOS", "potash demand uplift", 0.75)]),
    _evt("ira_solar_extension", "IRA solar tax credit extension passes Senate; FSLR benefits from accelerated US utility-scale demand.",
         ["FSLR"], "bill_passed", [("FSLR", "utility-scale solar pull-forward", 0.85)]),
    _evt("fda_glp1_approval", "FDA approves new GLP-1 indication for cardiovascular outcomes; LLY's tirzepatide gains additional addressable market.",
         ["LLY"], "fda_approval", [("LLY", "tirzepatide expanded label", 0.9), ("NVO", "competitive read-through to semaglutide", 0.6)]),
    _evt("china_chip_tariff", "New tariffs on Chinese semiconductor imports announced; TSM and US-domestic semis benefit from supply rebalancing.",
         ["TSM", "INTC"], "tariff_announced", [("TSM", "leading-edge foundry beneficiary", 0.7), ("INTC", "domestic foundry tailwind", 0.6)]),
]

QUERIES = [
    ("Looking for ideas about apple stock and momentum signals", None),
    ("Need bills or regulatory events helping fertilizer companies", "event_driven"),
    ("FDA drug approval impact on pharma names", "event_driven"),
    ("Mean reversion or RSI-based setups", "statistical"),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ledger = Ledger(tmp_path / "ledger.db")
        for h in HYPOTHESES:
            ledger.insert_hypothesis(h)

        print(f"Loading sentence-transformers model + building FAISS index over {len(HYPOTHESES)} hypotheses ...")
        r = HypothesisRetriever(ledger, index_dir=tmp_path / "faiss")
        n = r.reindex_from_ledger()
        print(f"  -> indexed {n} hypotheses (dim={r.dim})\n")

        for query, filt in QUERIES:
            print(f'> "{query}"  (filter_type={filt})')
            results = r.search(query, k=3, filter_type=filt)
            for hid, score, h in results:
                print(f"    sim={score:+.3f}  [{h.type:<14s}]  {h.name:<24s}  {h.thesis_text[:80]}")
            print()


if __name__ == "__main__":
    main()
