"""EDGAR ingest tests. Offline by default; integration test runs only when
SEC_IDENTITY is set and `-m integration` is passed."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_trade_lab.data import edgar_ingest


def test_ensure_identity_raises_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_IDENTITY", raising=False)
    monkeypatch.setattr(edgar_ingest, "_IDENTITY_SET", False)
    with pytest.raises(RuntimeError, match="SEC_IDENTITY"):
        edgar_ingest._ensure_identity()


def test_filing_to_dict_serializes_date() -> None:
    from datetime import date

    f = edgar_ingest.Filing(
        ticker="AAPL",
        cik="320193",
        accession_number="0000000000-00-000000",
        form="8-K",
        filing_date=date(2025, 6, 1),
        items=["2.02", "9.01"],
        text="hello",
        primary_url="https://www.sec.gov/x",
    )
    d = f.to_dict()
    assert d["filing_date"] == "2025-06-01"
    assert d["items"] == ["2.02", "9.01"]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("SEC_IDENTITY"),
    reason="SEC_IDENTITY env var not set; skipping integration test",
)
def test_fetch_aapl_8k_integration(tmp_path: Path) -> None:
    edgar_ingest._IDENTITY_SET = False  # force re-init in case prior test cleared env
    filings = edgar_ingest.fetch_filings(
        "AAPL",
        form="8-K",
        start="2025-01-01",
        end="2025-12-31",
        limit=2,
        cache_dir=tmp_path,
    )
    assert len(filings) >= 1
    f = filings[0]
    assert f.ticker == "AAPL"
    assert f.form == "8-K"
    assert f.cik
    assert f.accession_number
