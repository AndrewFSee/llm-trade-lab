"""Entity resolver tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_trade_lab.data.entity_resolver import (
    EntityResolver,
    ResolvedBeneficiary,
    load_default,
)


def _toy_resolver() -> EntityResolver:
    themes = {
        "fertilizer": {
            "keywords": ["fertilizer", "potash"],
            "beneficiaries": [
                {"ticker": "CF", "name": "CF", "mechanism": "n", "confidence": 0.9},
                {"ticker": "MOS", "name": "Mosaic", "mechanism": "p", "confidence": 0.9},
            ],
        },
        "agriculture": {
            "keywords": ["farm", "agriculture"],
            "beneficiaries": [
                {"ticker": "DE", "name": "Deere", "mechanism": "tractors", "confidence": 0.9},
                {"ticker": "CF", "name": "CF", "mechanism": "n", "confidence": 0.6},
            ],
        },
    }
    return EntityResolver(themes)


def test_resolve_single_theme() -> None:
    r = _toy_resolver()
    out = r.resolve("Bill provides fertilizer subsidies for small farms.")
    tickers = {b.ticker for b in out}
    assert tickers == {"CF", "MOS", "DE"}  # both fertilizer and agriculture themes match


def test_resolve_dedupes_by_max_confidence() -> None:
    r = _toy_resolver()
    out = r.resolve("Bill provides fertilizer subsidies for small farms.")
    by_ticker = {b.ticker: b for b in out}
    # CF appears in both themes (0.9 and 0.6); the 0.9 entry should win
    assert by_ticker["CF"].confidence == pytest.approx(0.9)


def test_resolve_no_match_returns_empty() -> None:
    r = _toy_resolver()
    assert r.resolve("Nothing relevant here.") == []
    assert r.resolve("") == []


def test_resolve_case_insensitive() -> None:
    r = _toy_resolver()
    upper = r.resolve("FERTILIZER")
    lower = r.resolve("fertilizer")
    assert {b.ticker for b in upper} == {b.ticker for b in lower}


def test_known_themes_sorted() -> None:
    r = _toy_resolver()
    assert r.known_themes() == ["agriculture", "fertilizer"]


def test_matched_themes() -> None:
    r = _toy_resolver()
    assert set(r.matched_themes("potash")) == {"fertilizer"}
    assert set(r.matched_themes("farm potash")) == {"fertilizer", "agriculture"}


def test_to_schema_conversion() -> None:
    rb = ResolvedBeneficiary(
        ticker="CF", name="CF", mechanism="n", confidence=0.9, matched_theme="fertilizer"
    )
    s = rb.to_schema()
    assert s.ticker == "CF"
    assert s.mechanism == "n"
    assert s.confidence == 0.9


def test_from_yaml_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        EntityResolver.from_yaml(Path("/nonexistent/path/entities.yaml"))


def test_load_default_works() -> None:
    """Verify the bundled configs/entities.yaml loads and has expected themes."""
    r = load_default()
    themes = r.known_themes()
    # Sanity-check a few we know are in the starter set.
    for expected in ["fertilizer", "solar", "ev", "semiconductors", "glp1", "oil_majors"]:
        assert expected in themes, f"theme {expected!r} missing from default config"


def test_default_resolves_known_examples() -> None:
    r = load_default()
    fertilizer = r.resolve("farm bill subsidizes fertilizer for corn growers")
    assert "CF" in {b.ticker for b in fertilizer}
    assert "MOS" in {b.ticker for b in fertilizer}

    glp1 = r.resolve("FDA approves new GLP-1 indication")
    assert "LLY" in {b.ticker for b in glp1}
    assert "NVO" in {b.ticker for b in glp1}


def test_lookup_ticker_returns_themes_with_context() -> None:
    r = load_default()
    # UNH is in healthcare_insurance which has current_context
    matches = r.lookup_ticker("UNH")
    assert matches, "UNH should match at least one theme with current_context"
    theme_names = {t for t, _ in matches}
    assert "healthcare_insurance" in theme_names
    # Each match returns non-empty context
    for theme, ctx in matches:
        assert ctx, f"theme {theme!r} returned empty context"


def test_lookup_ticker_skips_themes_without_context() -> None:
    # A toy resolver with one theme that has no current_context
    themes = {
        "no_ctx_theme": {
            "keywords": ["foo"],
            "beneficiaries": [{"ticker": "AAPL", "name": "Apple", "mechanism": "x", "confidence": 0.9}],
        }
    }
    r = EntityResolver(themes)
    assert r.lookup_ticker("AAPL") == []


def test_lookup_ticker_unknown_ticker() -> None:
    r = load_default()
    assert r.lookup_ticker("ZZZZZ_NOT_A_TICKER") == []
