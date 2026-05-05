"""Unified event stream — normalizes all source ingest modules into one Event schema.

In Phase 1, the LLM consumes Event objects to generate event-driven hypotheses.
The Event carries enough structured features for the LLM to reason about
catalyst type, plausible beneficiaries, and event probability.

What sources flow in:
  - SEC 8-K filings           -> Event(source="edgar_8k")
  - SEC Form 4 transactions   -> Event(source="form4")
  - Congress.gov bills         -> Event(source="congress_bill")
  - Federal Register docs      -> Event(source="federal_register")
  - FRED upcoming releases     -> Event(source="fred_release")
  - Alpha Vantage news         -> Event(source="alphavantage_news")

What does NOT flow in (deliberately):
  - 10-K / 10-Q                -> these are reference documents, not catalysts;
                                   accessed on-demand via edgar_ingest for context.
  - FRED time-series values    -> reference data; conditioned on, not events.
  - yfinance OHLCV             -> price data; events trigger trades on top of it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any, Literal

from llm_trade_lab.data.entity_resolver import EntityResolver, ResolvedBeneficiary, load_default

EventSource = Literal[
    "edgar_8k",
    "form4",
    "congress_bill",
    "federal_register",
    "fred_release",
    "alphavantage_news",
]


@dataclass
class Event:
    source: EventSource
    event_type: str                                # source-specific subtype
    event_date: datetime                            # UTC
    title: str
    body: str
    raw_id: str                                     # accession / doc / bill id (unique per source)
    primary_tickers: list[str] = field(default_factory=list)        # tickers explicitly named in the source
    suggested_beneficiaries: list[ResolvedBeneficiary] = field(default_factory=list)
    matched_themes: list[str] = field(default_factory=list)
    structured: dict[str, Any] = field(default_factory=dict)
    url: str = ""

    @property
    def event_id(self) -> str:
        return f"{self.source}:{self.raw_id}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": self.source,
            "event_type": self.event_type,
            "event_date": self.event_date.isoformat(),
            "title": self.title,
            "body": self.body,
            "raw_id": self.raw_id,
            "primary_tickers": list(self.primary_tickers),
            "suggested_beneficiaries": [asdict(b) for b in self.suggested_beneficiaries],
            "matched_themes": list(self.matched_themes),
            "structured": self.structured,
            "url": self.url,
        }
        return d


def _to_dt(d: date | datetime | None) -> datetime:
    if d is None:
        return datetime.now(timezone.utc)
    if isinstance(d, datetime):
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime.combine(d, time.min, tzinfo=timezone.utc)


def _enrich(event: Event, resolver: EntityResolver) -> Event:
    """Apply entity resolver to title+body; populate suggested_beneficiaries + matched_themes."""
    text = f"{event.title}\n{event.body}"
    event.suggested_beneficiaries = resolver.resolve(text)
    event.matched_themes = resolver.matched_themes(text)
    return event


# --------------------------------------------------------------------- adapters

def event_from_8k(filing: Any, resolver: EntityResolver) -> Event:
    """Build an Event from llm_trade_lab.data.edgar_ingest.Filing (form='8-K')."""
    items_str = ", ".join(filing.items) if filing.items else "-"
    e = Event(
        source="edgar_8k",
        event_type=f"Items: {items_str}" if filing.items else "8-K",
        event_date=_to_dt(filing.filing_date),
        title=f"{filing.ticker} 8-K  [{items_str}]",
        body=(filing.text or "")[:4000],
        raw_id=filing.accession_number,
        primary_tickers=[filing.ticker],
        structured={
            "ticker": filing.ticker,
            "cik": filing.cik,
            "form": filing.form,
            "items": list(filing.items),
        },
        url=filing.primary_url,
    )
    return _enrich(e, resolver)


def event_from_form4(txn: Any, resolver: EntityResolver) -> Event:
    """Build an Event from llm_trade_lab.data.form4_ingest.InsiderTransaction."""
    direction = (
        "BUY" if txn.is_open_market_buy
        else "SELL" if txn.is_open_market_sale
        else txn.transaction_code
    )
    title = (
        f"{txn.ticker} insider {direction}: {txn.insider_name} "
        f"({txn.insider_title or 'insider'})  "
        f"{int(txn.shares):,} sh @ ${txn.price_per_share:.2f}"
    )
    e = Event(
        source="form4",
        event_type=f"{txn.transaction_code} ({txn.transaction_type})" if txn.transaction_type else txn.transaction_code,
        event_date=_to_dt(txn.transaction_date or txn.filing_date),
        title=title,
        body="",  # Form 4 has no narrative; structured features carry the signal
        raw_id=f"{txn.accession_number}:{txn.transaction_code}:{int(txn.shares)}",
        primary_tickers=[txn.ticker],
        structured={
            "ticker": txn.ticker,
            "cik": txn.cik,
            "insider_name": txn.insider_name,
            "insider_title": txn.insider_title,
            "is_director": txn.is_director,
            "is_officer": txn.is_officer,
            "is_ten_percent_owner": txn.is_ten_percent_owner,
            "transaction_code": txn.transaction_code,
            "transaction_type": txn.transaction_type,
            "shares": txn.shares,
            "price_per_share": txn.price_per_share,
            "total_value": txn.total_value,
            "shares_owned_after": txn.shares_owned_after,
        },
    )
    return _enrich(e, resolver)


def event_from_bill(bill: Any, resolver: EntityResolver) -> Event:
    """Build an Event from llm_trade_lab.data.congress_ingest.BillSummary or BillDetail."""
    summary = getattr(bill, "summary", bill)
    text = ""
    sponsors_party = ""
    cosponsors_count = 0
    actions_count = 0
    if hasattr(bill, "text"):  # BillDetail
        text = bill.text or ""
        cosponsors_count = bill.cosponsors_count
        actions_count = bill.actions_count
        sponsors = bill.sponsors or []
        if sponsors:
            sponsors_party = sponsors[0].get("party", "") or ""
    e = Event(
        source="congress_bill",
        event_type=summary.latest_action_text[:80] or "introduced",
        event_date=_to_dt(summary.latest_action_date or summary.introduced_date),
        title=f"{summary.bill_id.upper()}: {summary.title[:160]}",
        body=text[:4000] if text else summary.title,
        raw_id=summary.bill_id,
        structured={
            "congress": summary.congress,
            "bill_type": summary.bill_type,
            "number": summary.number,
            "policy_area": summary.policy_area,
            "introduced_date": summary.introduced_date.isoformat() if summary.introduced_date else None,
            "latest_action_date": summary.latest_action_date.isoformat() if summary.latest_action_date else None,
            "latest_action_text": summary.latest_action_text,
            "sponsor_party": sponsors_party,
            "cosponsors_count": cosponsors_count,
            "actions_count": actions_count,
        },
        url=summary.url,
    )
    return _enrich(e, resolver)


def event_from_fr_doc(doc: Any, resolver: EntityResolver) -> Event:
    """Build an Event from llm_trade_lab.data.federal_register_ingest.FRDocument."""
    e = Event(
        source="federal_register",
        event_type=doc.document_type or "Notice",
        event_date=_to_dt(doc.publication_date),
        title=doc.title[:240],
        body=(doc.body_text or doc.abstract or "")[:4000],
        raw_id=doc.document_number,
        structured={
            "document_type": doc.document_type,
            "agencies": list(doc.agencies),
            "topics": list(doc.topics),
            "effective_date": doc.effective_date.isoformat() if doc.effective_date else None,
            "abstract": doc.abstract,
        },
        url=doc.html_url,
    )
    return _enrich(e, resolver)


def event_from_news(article: Any, resolver: EntityResolver) -> Event:
    """Build an Event from llm_trade_lab.data.alphavantage_news_ingest.NewsArticle."""
    primary = sorted(
        [
            ts.ticker
            for ts in article.ticker_sentiments
            if ts.relevance_score >= 0.5
        ]
    )
    e = Event(
        source="alphavantage_news",
        event_type=article.overall_sentiment_label or "news",
        event_date=article.time_published if article.time_published.tzinfo else article.time_published.replace(tzinfo=timezone.utc),
        title=article.title[:240],
        body=article.summary[:4000],
        raw_id=article.url,
        primary_tickers=primary,
        structured={
            "source": article.source,
            "source_domain": article.source_domain,
            "overall_sentiment_score": article.overall_sentiment_score,
            "overall_sentiment_label": article.overall_sentiment_label,
            "topics": article.topics,
            "ticker_sentiments": [asdict(t) for t in article.ticker_sentiments],
        },
        url=article.url,
    )
    return _enrich(e, resolver)


def event_from_fred_release(release_date: Any, resolver: EntityResolver) -> Event:
    """Build an Event from llm_trade_lab.data.fred_ingest.ReleaseDate (upcoming releases)."""
    e = Event(
        source="fred_release",
        event_type=release_date.release_name,
        event_date=_to_dt(release_date.date),
        title=f"FRED release: {release_date.release_name}",
        body=f"Scheduled release date {release_date.date.isoformat()} for {release_date.release_name}.",
        raw_id=f"{release_date.release_id}:{release_date.date.isoformat()}",
        structured={
            "release_id": release_date.release_id,
            "release_name": release_date.release_name,
        },
    )
    return _enrich(e, resolver)


# ---------------------------------------------------------------------- collect

def collect_events(
    *,
    since: str,
    until: str,
    tickers: list[str] | None = None,
    bill_congress: int | None = None,
    fr_agencies: list[str] | None = None,
    fr_topics_terms: str | None = None,
    fred_release_window: bool = False,
    news_tickers: list[str] | None = None,
    news_topics: list[str] | None = None,
    limit_per_source: int = 25,
    resolver: EntityResolver | None = None,
) -> list[Event]:
    """Poll all enabled sources and return a unified, descending-time-sorted event list.

    Each source is opt-in via its own kwarg so that no API quota is burned for
    sources you don't want.

    Args:
        since, until: YYYY-MM-DD bounds.
        tickers: if set, fetch 8-Ks and Form 4s for each ticker.
        bill_congress: if set (e.g. 119), fetch recent bills from that Congress.
        fr_agencies: if set, fetch Federal Register docs for these agency slugs.
        fr_topics_terms: optional full-text search filter for Federal Register.
        fred_release_window: if True, fetch upcoming/recent FRED releases in window.
        news_tickers: if set, fetch Alpha Vantage news for these tickers.
        news_topics: if set, fetch Alpha Vantage news for these topics.
        limit_per_source: per-source cap (most-recent first).
        resolver: entity resolver instance; defaults to bundled config.
    """
    resolver = resolver or load_default()
    events: list[Event] = []

    if tickers:
        from llm_trade_lab.data.edgar_ingest import fetch_filings
        from llm_trade_lab.data.form4_ingest import fetch_insider_transactions

        for ticker in tickers:
            for f in fetch_filings(ticker, form="8-K", start=since, end=until, limit=limit_per_source):
                events.append(event_from_8k(f, resolver))
            for txn in fetch_insider_transactions(ticker, start=since, end=until, limit=limit_per_source):
                events.append(event_from_form4(txn, resolver))

    if bill_congress:
        from llm_trade_lab.data.congress_ingest import list_recent_bills

        for b in list_recent_bills(
            congress=bill_congress, since=since, until=until, limit=limit_per_source
        ):
            events.append(event_from_bill(b, resolver))

    if fr_agencies or fr_topics_terms:
        from llm_trade_lab.data.federal_register_ingest import search_documents

        for d in search_documents(
            since=since, until=until,
            agencies=fr_agencies, terms=fr_topics_terms,
            per_page=limit_per_source,
        ):
            events.append(event_from_fr_doc(d, resolver))

    if fred_release_window:
        from llm_trade_lab.data.fred_ingest import (
            HIGH_IMPACT_RELEASE_PATTERNS,
            upcoming_release_dates,
        )

        # Curated allow-list keeps the unified stream focused on the ~20 macro
        # releases that actually move markets (FOMC, NFP, CPI, GDP, ISM, ...).
        for rd in upcoming_release_dates(
            start=since, end=until, name_patterns=HIGH_IMPACT_RELEASE_PATTERNS
        ):
            events.append(event_from_fred_release(rd, resolver))

    if news_tickers or news_topics:
        from llm_trade_lab.data.alphavantage_news_ingest import fetch_news_sentiment

        articles = fetch_news_sentiment(
            tickers=news_tickers, topics=news_topics, limit=limit_per_source
        )
        for a in articles:
            events.append(event_from_news(a, resolver))

    events.sort(key=lambda e: e.event_date, reverse=True)
    return events
