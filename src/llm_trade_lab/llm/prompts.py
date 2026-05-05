"""Prompt templates for hypothesis generation.

Two tracks share schema and tooling but use distinct system prompts that
emphasize the right discipline for each:
  - Statistical: rigor about testability + falsifiability; no fake alpha claims.
  - Event-driven: calibrated event_probability + concrete mechanism per beneficiary.

PROMPT_VERSION is bumped whenever a system or builder template changes; it
flows into the hypothesis's `model_version_hash` so we can attribute outcomes
to specific prompt revisions during fine-tune analysis.
"""
from __future__ import annotations

from typing import Any

PROMPT_VERSION = 7


# ---------------------------------------------------- per-track strategy catalogs

# Statistical track: signal-based strategies that take params and produce
# entry/exit logic from price/volume features. These are appropriate for the
# 22-month default backtest window because they enter/exit multiple times.
STATISTICAL_STRATEGIES = {
    "sma_cross": {
        "params": {"fast": "int (e.g. 20)", "slow": "int (e.g. 50)"},
        "description": "SMA crossover (trend-following). Long when fast SMA crosses above slow SMA; exit on reverse cross.",
    },
    "rsi_reversion": {
        "params": {"period": "int (default 14)", "buy_threshold": "int (default 30)", "exit_threshold": "int (default 70)"},
        "description": "RSI mean-reversion (overbought/oversold). Long when RSI drops below buy_threshold; exit when RSI rises above exit_threshold.",
    },
    "donchian_breakout": {
        "params": {"entry_lookback": "int (default 20)", "exit_lookback": "int (default 10)"},
        "description": "Donchian channel breakout (trend-following). Long when close > prior entry_lookback-day high; exit when close < prior exit_lookback-day low.",
    },
    "bollinger_reversion": {
        "params": {"period": "int (default 20)", "n_std": "float (default 2.0)"},
        "description": "Bollinger Band mean-reversion. Long when close drops below (SMA - n_std * std); exit when close rises back above SMA.",
    },
}

# Event-driven track: strategies that operate over a single event window
# (caller slices OHLCV to [event_date, event_date + horizon_days]).
EVENT_DRIVEN_STRATEGIES = {
    "event_hold": {
        "params": {},
        "description": "Buy on first bar of the event window, hold to end of window. Use ONLY for event-driven hypotheses with a real catalyst event_date.",
    },
}

# Combined catalog used by the engine registry. Both tracks can technically
# call any strategy, but the prompts steer each track to its appropriate set.
STRATEGY_CATALOG = {**STATISTICAL_STRATEGIES, **EVENT_DRIVEN_STRATEGIES}


def _format_catalog(catalog: dict) -> str:
    lines = []
    for sid, info in catalog.items():
        params = ", ".join(f"{k}: {v}" for k, v in info["params"].items()) or "(no params)"
        lines.append(f'  - "{sid}" -- {info["description"]}\n      Params: {params}')
    return "\n".join(lines)


# --------------------------------------------------------- statistical track

STATISTICAL_SYSTEM_PROMPT = f"""You are a disciplined quantitative trading hypothesis generator for US equities.

Your job: given a market regime snapshot and a few retrieved past hypotheses with their realized outcomes, produce ONE NEW falsifiable statistical hypothesis as a JSON object.

Hard rules:
- The hypothesis MUST be machine-executable. `entry_rule.strategy_id` must reference a registered strategy.
- Available strategies for the STATISTICAL track (use the strategy_id literally):
{_format_catalog(STATISTICAL_STRATEGIES)}
- DO NOT use `event_hold` here — that strategy is reserved for the event-driven track. On the statistical track over a multi-month window it degenerates to buy-and-hold and produces meaningless backtests.
- The mechanism must be concrete (a specific market condition or signal), not "the stock will go up because momentum."
- Do NOT claim guaranteed alpha. The thesis_text should describe a measurable, testable conjecture.
- Universe: 1-5 tickers. Avoid over-broad baskets.
- Sizing MUST be in [0.05, 1.0]. Use 0.05-0.15 for low conviction, 0.3-0.7 for typical, 0.8-1.0 for high conviction. NEVER 0.

DIVERSITY REQUIREMENT (critical):
- The hypothesis MUST be SUBSTANTIALLY DIFFERENT from every retrieved memory entry.
- Do NOT just rename, rephrase, or re-parameterize a prior idea (e.g., changing "fast: 20" to "fast: 25").
- If retrieved memory is dense in one pattern (e.g., SMA crossovers on tech), generate something orthogonal:
  different universe, different signal type, different timeframe, or different sector.
- Treat the retrieved memory as a "what's already been tried" list — your job is to extend the search,
  not to consolidate it. Coverage of the hypothesis space is more valuable than another local refinement.

SECTOR CONTEXT ENGAGEMENT (when SECTOR CONTEXT is present in the user prompt):
- When the user prompt provides SECTOR CONTEXT for a ticker, the `thesis_text` MUST explicitly
  cite a SPECIFIC concern, dynamic, or phrase from that context. Examples:
    GOOD: "...despite ongoing CMS Medicare Advantage rate cuts and the active DOJ probe..."
    GOOD: "...we view the AI capex digestion risk as overstated because..."
    BAD:  "...as the sector stabilizes..."          (vague — does not engage)
    BAD:  "...sector rotation favors this name..."  (vague — does not engage)
- If your thesis CONTRADICTS the context (bullish on a documented-headwind sector), explicitly
  say why the consensus framing is wrong or about to reverse.
- If your thesis ALIGNS with the context (contrarian/short/hedge play), name the specific
  concern from the context that motivates the trade.
- The failure mode to avoid is ignoring the context entirely. The context exists precisely
  because your training data doesn't include it; ignoring it is equivalent to ignoring a
  current news headline.

Output schema:
{{
  "name": "snake_case_short_name",
  "thesis_text": "1-2 sentence falsifiable hypothesis with mechanism",
  "universe": ["TICKER1"],
  "entry_rule": {{"strategy_id": "sma_cross", "params": {{"fast": 20, "slow": 50}}}},
  "exit_rule":  {{"strategy_id": "sma_cross", "params": {{"fast": 20, "slow": 50}}}},
  "holding_period_days": null,
  "sizing": 1.0
}}

Return ONLY the JSON object. No prose, no code fences."""


def build_statistical_user_prompt(
    *,
    today: str,
    universe_hint: list[str] | None = None,
    ticker_flags: dict[str, list[str]] | None = None,
    sector_contexts: dict[str, list[dict[str, str]]] | None = None,
    regime_features: dict[str, Any] | None = None,
    retrieved_memory: list[dict[str, Any]] | None = None,
) -> str:
    parts = [f"Today: {today}"]
    if universe_hint:
        has_enrichment = bool(ticker_flags) or bool(sector_contexts)
        if has_enrichment:
            # Per-ticker auto-flags + sector context. Auto-flags come from price
            # data; sector context comes from hand-curated narrative for themes
            # the LLM may not have current information about (post-cutoff).
            # Not blocking — model can override either layer with explicit reasoning.
            lines = [
                "Universe hint (you may pick a subset or related names).",
                "AUTO-FLAGS are facts from recent price data. SECTOR CONTEXT is curated narrative",
                "about ongoing dynamics that may post-date the model training cutoff.",
                "If either contradicts your thesis, address it explicitly or reduce conviction.",
                "",
            ]
            for t in universe_hint:
                tu = t.upper()
                fs = (ticker_flags or {}).get(tu, [])
                if fs:
                    lines.append(f"  {tu:<6s}  {' | '.join(fs)}")
                else:
                    lines.append(f"  {tu:<6s}  (no auto-flags available)")
                ctxs = (sector_contexts or {}).get(tu, [])
                for entry in ctxs:
                    theme = entry.get("theme", "?")
                    ctx = (entry.get("context") or "").strip()
                    if ctx:
                        lines.append(f"    [{theme}]")
                        for line in ctx.splitlines():
                            line = line.strip()
                            if line:
                                lines.append(f"      {line}")
            parts.append("\n".join(lines))
        else:
            parts.append(f"Universe hint (you may pick a subset or related names): {', '.join(universe_hint)}")
    if regime_features:
        feat_lines = "\n".join(f"  {k}: {v}" for k, v in regime_features.items())
        parts.append(f"Current market regime features:\n{feat_lines}")
    if retrieved_memory:
        parts.append("Retrieved past hypotheses with outcomes (most-similar first):")
        for i, m in enumerate(retrieved_memory[:5], 1):
            outcome = m.get("outcome", "no realized outcome yet")
            parts.append(f"  [{i}] {m.get('name', '?')}: {m.get('thesis', '?')[:160]}\n      outcome: {outcome}")
    parts.append("\nGenerate ONE new statistical hypothesis matching the schema.")
    return "\n\n".join(parts)


# ------------------------------------------------------- event-driven track

EVENT_DRIVEN_SYSTEM_PROMPT = f"""You are a disciplined event-driven trading analyst for US equities.

Your job: given a real event (bill, regulatory filing, FDA notice, FOMC, news) plus theme-matched candidate beneficiaries plus a few retrieved past similar events with outcomes, produce ONE NEW event-driven hypothesis as a JSON object.

Hard rules:
- Reference the supplied event verbatim in trigger_event.
- Beneficiaries: pick from the provided candidate list (you MAY narrow it, but do not invent tickers). For each, give a concrete mechanism — *how* this event flows to that ticker's revenue/cost/sentiment.
- ANTI-SHOEHORN: pick a beneficiary ONLY if its theme is genuinely central to the event. If the entity resolver returned candidates whose theme is only peripherally mentioned (e.g., the bill mentions "rare earth" once in passing but is primarily about something else), DO NOT force the trade. Instead express low conviction: sizing in [0.05, 0.15] AND event_probability < 0.10. It is far better to flag a stretched hypothesis honestly than to fabricate a mechanism.
- Sizing MUST be in [0.05, 1.0]. NEVER 0. Use 0.05-0.15 for low conviction (peripheral theme, weak mechanism), 0.3-0.7 for typical, 0.8-1.0 only for direct primary mechanisms.
- event_probability: calibrated probability that the event RESOLVES IN THE EXPECTED DIRECTION (bill passes / rule finalized as proposed / approval granted). Don't say 0.9 unless you'd bet at those odds. For introduced bills with no committee action, default ~0.05-0.15. For final rules already published, ~0.95.
- expected_horizon_days: realistic window for the catalyst to play out. MUST be >= 5. Bills: 60-180. Regs/FDA: 5-30. FOMC: 1-5 (use 5).
- confounders: list 2-4 specific things that could break the thesis (e.g., "bill stalls in committee", "broad market drawdown", "Fed cuts rates first").
- Available strategies for the EVENT-DRIVEN track (use literally):
{_format_catalog(EVENT_DRIVEN_STRATEGIES)}

SECTOR CONTEXT ENGAGEMENT (when SECTOR CONTEXT is present in the user prompt):
- When the user prompt provides SECTOR CONTEXT for any matched theme, the `thesis_text` MUST
  explicitly cite a SPECIFIC concern, dynamic, or phrase from that context. Examples:
    GOOD: "...despite documented CMS Medicare Advantage rate-cut pressure on UNH/HUM..."
    GOOD: "...the bill's rare-earth carve-out is meaningful BECAUSE MP/USAR are already trading
           at 33% drawdown from peak, suggesting prior weakness is partially digested..."
    BAD:  "...sector positioning favorable..."   (vague — does not engage)
- If the SECTOR CONTEXT contradicts the trade direction (e.g., bullish event on a sector with
  documented headwinds), either reduce sizing/confidence or explicitly justify why this catalyst
  outweighs the headwinds.
- The failure mode to avoid is ignoring the context entirely. The context exists precisely
  because your training data doesn't include it.

Output schema:
{{
  "name": "snake_case_short_name",
  "thesis_text": "1-2 sentence event mechanism + expected directional impact",
  "universe": ["TICKER1", "TICKER2"],
  "entry_rule": {{"strategy_id": "event_hold", "params": {{}}}},
  "exit_rule":  {{"strategy_id": "event_hold", "params": {{}}}},
  "holding_period_days": 30,
  "sizing": 0.5,
  "trigger_event": {{
    "source": "<one of: congress_bill, edgar_8k, form4, federal_register, fomc, fred_release, alphavantage_news>",
    "doc_id": "<source-specific id from the event>",
    "event_type": "<short type label>",
    "event_date": "YYYY-MM-DD"
  }},
  "beneficiaries": [
    {{"ticker": "T1", "mechanism": "concrete causal chain", "confidence": 0.7}}
  ],
  "event_probability": 0.40,
  "expected_horizon_days": 30,
  "confounders": ["risk 1", "risk 2"]
}}

Return ONLY the JSON object. No prose, no code fences."""


def build_event_driven_user_prompt(
    *,
    event: dict[str, Any],
    candidate_beneficiaries: list[dict[str, Any]],
    retrieved_memory: list[dict[str, Any]] | None = None,
) -> str:
    parts = ["EVENT to generate a hypothesis for:"]
    parts.append(f"  source:     {event.get('source')}")
    parts.append(f"  doc_id:     {event.get('doc_id') or event.get('raw_id')}")
    parts.append(f"  event_type: {event.get('event_type')}")
    parts.append(f"  event_date: {event.get('event_date')}")
    parts.append(f"  title:      {event.get('title', '')[:200]}")
    body = event.get("body", "") or ""
    if body:
        parts.append(f"  body excerpt: {body[:1500]}")

    # Sector context: per-theme narrative the LLM may not know post-cutoff.
    # Dedupe across multiple beneficiaries from the same theme.
    contexts_seen: dict[str, str] = {}
    for b in candidate_beneficiaries:
        theme = b.get("matched_theme") or ""
        ctx = (b.get("current_context") or "").strip()
        if theme and ctx and theme not in contexts_seen:
            contexts_seen[theme] = ctx
    if contexts_seen:
        parts.append("")
        parts.append("SECTOR CONTEXT (information beyond the model training cutoff — weigh accordingly):")
        for theme, ctx in contexts_seen.items():
            parts.append(f"  [{theme}]")
            for line in ctx.splitlines():
                line = line.strip()
                if line:
                    parts.append(f"    {line}")

    parts.append("")
    parts.append("CANDIDATE BENEFICIARIES (from entity resolver; pick a subset, don't invent).")
    parts.append("AUTO-FLAGS are facts derived from recent price data — not blocking, but you should reason about them.")
    for b in candidate_beneficiaries:
        parts.append(
            f"  {b.get('ticker')}  ({b.get('matched_theme', '?')})  "
            f"prior_conf={b.get('confidence', 0):.2f}"
        )
        parts.append(f"    mechanism: {b.get('mechanism', '')[:140]}")
        flags = b.get("flags") or []
        if flags:
            parts.append(f"    AUTO-FLAGS: {' | '.join(flags)}")

    if retrieved_memory:
        parts.append("")
        parts.append("RETRIEVED PAST SIMILAR EVENTS + OUTCOMES (most-similar first):")
        for i, m in enumerate(retrieved_memory[:5], 1):
            outcome = m.get("outcome", "no realized outcome yet")
            parts.append(f"  [{i}] {m.get('name', '?')}: {m.get('thesis', '?')[:160]}\n      outcome: {outcome}")

    parts.append("")
    parts.append("Generate ONE event-driven hypothesis matching the schema.")
    parts.append("If the AUTO-FLAGS or SECTOR CONTEXT contradict your thesis, either skip the affected beneficiaries or address the headwinds explicitly in thesis_text and lower confidence/sizing accordingly.")
    return "\n".join(parts)
