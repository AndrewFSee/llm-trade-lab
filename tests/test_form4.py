"""Form 4 ingest tests. Offline by default."""
from __future__ import annotations

import os
from datetime import date

import pytest

from llm_trade_lab.data import form4_ingest


def test_to_float_handles_missing() -> None:
    assert form4_ingest._to_float(None) == 0.0
    assert form4_ingest._to_float("") == 0.0
    assert form4_ingest._to_float("12.5") == 12.5
    assert form4_ingest._to_float(12.5) == 12.5
    assert form4_ingest._to_float("nope") == 0.0


def test_row_to_transaction_buy_vs_sale() -> None:
    base = {
        "ticker": "AAPL",
        "cik": "320193",
        "accession_number": "x",
        "filing_date": "2025-06-01",
        "transaction_date": "2025-05-30",
        "insider_name": "Tim Cook",
        "insider_title": "CEO",
        "is_director": True,
        "is_officer": True,
        "is_ten_percent_owner": False,
        "shares": 1000.0,
        "price_per_share": 200.0,
        "total_value": 200_000.0,
        "shares_owned_after": 50_000.0,
        "transaction_type": "Purchase",
    }
    buy = form4_ingest._row_to_transaction({**base, "transaction_code": "P"})
    sell = form4_ingest._row_to_transaction({**base, "transaction_code": "S"})
    option_exercise = form4_ingest._row_to_transaction({**base, "transaction_code": "M"})
    assert buy.is_open_market_buy and not buy.is_open_market_sale
    assert sell.is_open_market_sale and not sell.is_open_market_buy
    assert not option_exercise.is_open_market_buy
    assert not option_exercise.is_open_market_sale
    assert buy.transaction_date == date(2025, 5, 30)
    assert buy.filing_date == date(2025, 6, 1)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("SEC_IDENTITY"),
    reason="SEC_IDENTITY env var not set",
)
def test_fetch_aapl_form4_integration(tmp_path) -> None:
    from llm_trade_lab.data import edgar_ingest

    edgar_ingest._IDENTITY_SET = False
    txns = form4_ingest.fetch_insider_transactions(
        "AAPL", start="2025-01-01", end="2025-12-31", limit=3, cache_dir=tmp_path
    )
    # Filings exist; transactions may be 0 if all are derivative or unparseable.
    assert isinstance(txns, list)
