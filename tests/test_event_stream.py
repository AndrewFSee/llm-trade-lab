"""Unified event stream tests. Adapter unit tests use synthetic source objects."""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from llm_trade_lab.data.entity_resolver import EntityResolver
from llm_trade_lab.events.event_stream import (
    Event,
    event_from_8k,
    event_from_bill,
    event_from_form4,
    event_from_fr_doc,
    event_from_news,
)


@pytest.fixture
def resolver() -> EntityResolver:
    return EntityResolver(
        {
            "fertilizer": {
                "keywords": ["fertilizer", "potash"],
                "beneficiaries": [
                    {"ticker": "CF", "name": "CF", "mechanism": "n", "confidence": 0.9},
                    {"ticker": "MOS", "name": "Mosaic", "mechanism": "p", "confidence": 0.9},
                ],
            },
            "solar": {
                "keywords": ["solar"],
                "beneficiaries": [
                    {"ticker": "FSLR", "name": "First Solar", "mechanism": "modules", "confidence": 0.9},
                ],
            },
        }
    )


def test_event_to_dict_round_trip_shape(resolver: EntityResolver) -> None:
    filing = SimpleNamespace(
        ticker="AAPL", cik="320193", accession_number="0001-25-001",
        form="8-K", filing_date=date(2025, 6, 1),
        items=["Item 5.02"], text="Apple announces CEO transition.",
        primary_url="https://www.sec.gov/x",
    )
    e = event_from_8k(filing, resolver)
    d = e.to_dict()
    assert d["source"] == "edgar_8k"
    assert d["primary_tickers"] == ["AAPL"]
    assert d["structured"]["items"] == ["Item 5.02"]
    assert d["url"] == "https://www.sec.gov/x"


def test_event_id_property(resolver: EntityResolver) -> None:
    filing = SimpleNamespace(
        ticker="AAPL", cik="320193", accession_number="0001-25-001",
        form="8-K", filing_date=date(2025, 6, 1),
        items=[], text="", primary_url="",
    )
    e = event_from_8k(filing, resolver)
    assert e.event_id == "edgar_8k:0001-25-001"


def test_8k_enriches_with_themes(resolver: EntityResolver) -> None:
    filing = SimpleNamespace(
        ticker="CF", cik="111", accession_number="acc-1",
        form="8-K", filing_date=date(2025, 6, 1),
        items=["Item 8.01"],
        text="The company reports record fertilizer demand for the quarter.",
        primary_url="",
    )
    e = event_from_8k(filing, resolver)
    assert "fertilizer" in e.matched_themes
    tickers = {b.ticker for b in e.suggested_beneficiaries}
    assert tickers == {"CF", "MOS"}


def test_form4_buy_event(resolver: EntityResolver) -> None:
    txn = SimpleNamespace(
        ticker="MSFT", cik="789019", accession_number="acc-2",
        filing_date=date(2025, 6, 1), transaction_date=date(2025, 5, 30),
        insider_name="S. Nadella", insider_title="CEO",
        is_director=True, is_officer=True, is_ten_percent_owner=False,
        transaction_code="P", transaction_type="Purchase",
        shares=10000.0, price_per_share=420.0, total_value=4_200_000.0,
        shares_owned_after=900_000.0, is_open_market_buy=True, is_open_market_sale=False,
    )
    e = event_from_form4(txn, resolver)
    assert e.source == "form4"
    assert "BUY" in e.title
    assert e.structured["transaction_code"] == "P"
    assert e.event_date.year == 2025 and e.event_date.month == 5


def test_bill_event_with_summary(resolver: EntityResolver) -> None:
    summary = SimpleNamespace(
        congress=119, bill_type="hr", number=1234, bill_id="hr-1234-119",
        title="A bill to expand solar tax credits", introduced_date=date(2025, 3, 1),
        latest_action_date=date(2025, 4, 1), latest_action_text="Referred to committee.",
        policy_area="Energy", url="https://api.congress.gov/x",
    )
    e = event_from_bill(summary, resolver)
    assert e.source == "congress_bill"
    assert e.structured["policy_area"] == "Energy"
    assert "FSLR" in {b.ticker for b in e.suggested_beneficiaries}


def test_federal_register_event(resolver: EntityResolver) -> None:
    doc = SimpleNamespace(
        document_number="2025-12345",
        title="New Standards for Potash Mining Operations",
        abstract="EPA finalizes potash effluent rules.",
        document_type="Rule", publication_date=date(2025, 8, 1),
        effective_date=date(2025, 9, 1),
        agencies=["Environmental Protection Agency"],
        topics=["mining", "fertilizer"],
        html_url="https://www.federalregister.gov/d/2025-12345",
        body_text="",
    )
    e = event_from_fr_doc(doc, resolver)
    assert e.source == "federal_register"
    assert e.structured["agencies"] == ["Environmental Protection Agency"]
    assert "fertilizer" in e.matched_themes


def test_news_event_uses_relevance_filter(resolver: EntityResolver) -> None:
    from llm_trade_lab.data.alphavantage_news_ingest import (
        NewsArticle,
        TickerSentiment,
    )

    article = NewsArticle(
        title="Fertilizer demand surges",
        url="https://x.com/y",
        time_published=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        summary="CF and MOS report records.",
        source="Reuters",
        source_domain="reuters.com",
        overall_sentiment_score=0.34,
        overall_sentiment_label="Somewhat-Bullish",
        topics=[("Earnings", 0.9)],
        ticker_sentiments=[
            TickerSentiment("CF", 0.95, 0.5, "Bullish"),
            TickerSentiment("AAPL", 0.10, -0.1, "Neutral"),
        ],
    )
    e = event_from_news(article, resolver)
    # AAPL has low relevance (0.10) and should be excluded from primary_tickers.
    assert e.primary_tickers == ["CF"]
    assert e.structured["overall_sentiment_label"] == "Somewhat-Bullish"
