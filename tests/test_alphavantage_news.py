"""Alpha Vantage NEWS_SENTIMENT tests."""
from __future__ import annotations

import os
from datetime import timezone

import pytest

from llm_trade_lab.data import alphavantage_news_ingest as av


def test_get_key_raises_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        av._get_key()


def test_parse_dt_handles_av_format() -> None:
    dt = av._parse_dt("20260424T193000")
    assert dt.year == 2026 and dt.month == 4 and dt.day == 24
    assert dt.hour == 19 and dt.minute == 30
    assert dt.tzinfo == timezone.utc


def test_parse_dt_handles_short_format() -> None:
    dt = av._parse_dt("20260424T1930")
    assert dt.year == 2026 and dt.hour == 19 and dt.minute == 30


def test_parse_article_full() -> None:
    raw = {
        "title": "Apple beats earnings",
        "url": "https://r.com/x",
        "time_published": "20260424T193000",
        "summary": "AAPL beat est.",
        "source": "Reuters",
        "source_domain": "www.reuters.com",
        "overall_sentiment_score": 0.34,
        "overall_sentiment_label": "Somewhat-Bullish",
        "topics": [
            {"topic": "Earnings", "relevance_score": "0.99"},
            {"topic": "Technology", "relevance_score": "0.85"},
        ],
        "ticker_sentiment": [
            {
                "ticker": "AAPL",
                "relevance_score": "0.95",
                "ticker_sentiment_score": "0.42",
                "ticker_sentiment_label": "Bullish",
            }
        ],
    }
    a = av._parse_article(raw)
    assert a.title == "Apple beats earnings"
    assert a.overall_sentiment_label == "Somewhat-Bullish"
    assert len(a.topics) == 2
    assert a.topics[0] == ("Earnings", 0.99)
    assert len(a.ticker_sentiments) == 1
    assert a.ticker_sentiments[0].ticker == "AAPL"
    assert a.ticker_sentiments[0].sentiment_label == "Bullish"


def test_summary_filters_by_relevance() -> None:
    a1 = av._parse_article(
        {
            "title": "x",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.9",
                    "ticker_sentiment_score": "0.5",
                    "ticker_sentiment_label": "Bullish",
                }
            ],
        }
    )
    a2 = av._parse_article(
        {
            "title": "y",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.2",
                    "ticker_sentiment_score": "-0.4",
                    "ticker_sentiment_label": "Bearish",
                }
            ],
        }
    )
    s = av.ticker_sentiment_summary([a1, a2], "AAPL", min_relevance=0.5)
    assert s["n_articles"] == 1
    assert s["avg_sentiment_score"] == pytest.approx(0.5)
    assert s["label_counts"]["Bullish"] == 1
    assert s["label_counts"]["Bearish"] == 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ALPHAVANTAGE_API_KEY"),
    reason="ALPHAVANTAGE_API_KEY env var not set",
)
def test_fetch_aapl_integration(tmp_path) -> None:
    articles = av.fetch_news_sentiment(tickers=["AAPL"], limit=5, cache_dir=tmp_path)
    assert len(articles) >= 1
    a = articles[0]
    assert a.title
    assert a.url
    assert a.overall_sentiment_label in av.VALID_LABELS or a.overall_sentiment_label == ""
