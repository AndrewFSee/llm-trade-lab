"""Fetch recent AAPL Form 4 insider transactions."""
from __future__ import annotations

from dotenv import load_dotenv

from llm_trade_lab.data.form4_ingest import fetch_insider_transactions


def main() -> None:
    load_dotenv()
    txns = fetch_insider_transactions(
        "AAPL", start="2025-01-01", end="2026-04-25", limit=10
    )
    print(f"AAPL Form 4 transactions ({len(txns)}):")
    for t in txns:
        if t.is_open_market_buy:
            tag = "BUY "
        elif t.is_open_market_sale:
            tag = "SELL"
        else:
            tag = f"{t.transaction_code:<2s}  "
        title = t.insider_title[:24] if t.insider_title else "-"
        ttype = t.transaction_type[:16] if t.transaction_type else "-"
        print(
            f"  {t.transaction_date or t.filing_date}  {tag}  [{ttype:<16s}]  "
            f"{t.insider_name[:22]:<22s}  {title:<22s}  "
            f"{t.shares:>9.0f} sh @ ${t.price_per_share:>7.2f}  =${t.total_value:>14,.0f}"
        )


if __name__ == "__main__":
    main()
