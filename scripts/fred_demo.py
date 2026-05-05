"""Pull a few FRED macro series and show recent values."""
from __future__ import annotations

from dotenv import load_dotenv

from llm_trade_lab.data.fred_ingest import get_series, upcoming_release_dates

SERIES = [
    ("UNRATE", "Unemployment rate (%)"),
    ("CPIAUCSL", "CPI all items (1982-84=100)"),
    ("FEDFUNDS", "Effective federal funds rate (%)"),
    ("GS10", "10-Year Treasury yield (%)"),
    ("VIXCLS", "VIX close"),
]


def main() -> None:
    load_dotenv()
    print("Recent FRED macro series:")
    for sid, label in SERIES:
        try:
            s = get_series(sid, start="2024-01-01", end="2026-04-25")
            last = s.dropna()
            if len(last) == 0:
                print(f"  {sid:<10s}  {label:<40s}  (no observations)")
                continue
            print(
                f"  {sid:<10s}  {label:<40s}  latest={last.iloc[-1]:>8.2f}  "
                f"on {last.index[-1]}  ({len(last)} obs)"
            )
        except Exception as e:
            print(f"  {sid:<10s}  ERROR: {e}")

    print("\nUpcoming releases (next 14 days):")
    from datetime import date, timedelta

    today = date.today()
    end = today + timedelta(days=14)
    rels = upcoming_release_dates(start=today.isoformat(), end=end.isoformat())
    for r in rels[:15]:
        print(f"  {r.date}  {r.release_name[:60]}")


if __name__ == "__main__":
    main()
