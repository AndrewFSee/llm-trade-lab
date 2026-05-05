"""Fetch recent AAPL 8-Ks and print a summary.

Requires SEC_IDENTITY in .env (copy .env.example to .env first).
"""
from __future__ import annotations

from dotenv import load_dotenv

from llm_trade_lab.data.edgar_ingest import fetch_filings


def main() -> None:
    load_dotenv()
    filings = fetch_filings(
        "AAPL", form="8-K", start="2025-01-01", end="2026-04-25", limit=5
    )
    print(f"AAPL 8-Ks ({len(filings)}):")
    for f in filings:
        items_str = ", ".join(f.items) if f.items else "-"
        excerpt = f.text[:180].replace("\n", " ").strip() if f.text else "(no text)"
        print(f"  {f.filing_date}  {f.accession_number}  items=[{items_str}]")
        print(f"    {excerpt}{'...' if len(f.text) > 180 else ''}")


if __name__ == "__main__":
    main()
