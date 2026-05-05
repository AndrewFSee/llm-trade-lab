"""Pull Alpha Vantage news sentiment for a set of tickers + show aggregates.

One API call per query — careful with the 25/day free tier.
"""
from __future__ import annotations

from dotenv import load_dotenv

from llm_trade_lab.data.alphavantage_news_ingest import (
    fetch_news_sentiment,
    ticker_sentiment_summary,
)

TICKERS = ["AAPL", "MSFT", "NVDA"]


def main() -> None:
    load_dotenv()
    print(f"Fetching latest news for {TICKERS} (one API call) ...")
    articles = fetch_news_sentiment(tickers=TICKERS, limit=50)
    print(f"  -> {len(articles)} articles")

    if not articles:
        print("No articles returned.")
        return

    print("\nMost recent 8 articles:")
    for a in articles[:8]:
        topics = ", ".join(t for t, _ in a.topics[:3])[:40]
        print(
            f"  {a.time_published.strftime('%Y-%m-%d %H:%M')}  "
            f"[{a.overall_sentiment_label[:18]:<18s}]  "
            f"({a.source[:18]:<18s})  {a.title[:80]}"
        )
        if topics:
            print(f"    topics: {topics}")

    print("\nPer-ticker sentiment summary (relevance >= 0.5):")
    for t in TICKERS:
        s = ticker_sentiment_summary(articles, t, min_relevance=0.5)
        avg = (
            f"{s['avg_sentiment_score']:+.3f}"
            if s["avg_sentiment_score"] is not None
            else "  n/a"
        )
        labels = s["label_counts"]
        print(
            f"  {t:<5s}  n={s['n_articles']:>3d}  avg={avg}  "
            f"Bull={labels['Bullish']}  SoBull={labels['Somewhat-Bullish']}  "
            f"Neut={labels['Neutral']}  SoBear={labels['Somewhat-Bearish']}  "
            f"Bear={labels['Bearish']}"
        )


if __name__ == "__main__":
    main()
