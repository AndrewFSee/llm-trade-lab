"""Poll every event source for the last 30 days, normalize, sort, summarize."""
from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from dotenv import load_dotenv

from llm_trade_lab.events.event_stream import collect_events


def main() -> None:
    load_dotenv()
    today = date.today()
    since = (today - timedelta(days=30)).isoformat()
    until = today.isoformat()

    print(f"Collecting events {since} -> {until} ...")
    events = collect_events(
        since=since,
        until=until,
        tickers=["AAPL"],
        bill_congress=119,
        fr_agencies=["food-and-drug-administration"],
        fred_release_window=True,
        # news_tickers omitted to spare Alpha Vantage's 25/day quota; uncomment to include:
        # news_tickers=["AAPL", "MSFT", "NVDA"],
        limit_per_source=15,
    )
    print(f"  -> {len(events)} events total\n")

    by_source = Counter(e.source for e in events)
    print("By source:")
    for src, n in by_source.most_common():
        print(f"  {src:<22s}  {n}")
    print()

    by_theme: Counter[str] = Counter()
    for e in events:
        by_theme.update(e.matched_themes)
    print("Top matched themes (entity resolver):")
    for theme, n in by_theme.most_common(10):
        print(f"  {theme:<22s}  {n}")
    print()

    print("Most recent 12 events:")
    for e in events[:12]:
        themes = ",".join(e.matched_themes[:3]) or "-"
        tickers = ",".join(e.primary_tickers[:3]) or "-"
        print(
            f"  {e.event_date.strftime('%Y-%m-%d')}  [{e.source:<18s}]  "
            f"tickers={tickers:<14s}  themes={themes:<22s}  {e.title[:80]}"
        )


if __name__ == "__main__":
    main()
