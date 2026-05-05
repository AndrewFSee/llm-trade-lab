"""Congress.gov ingest tests. Offline by default."""
from __future__ import annotations

import os
from datetime import date

import pytest

from llm_trade_lab.data import congress_ingest


def test_get_key_raises_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONGRESS_GOV_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CONGRESS_GOV_API_KEY"):
        congress_ingest._get_key()


def test_parse_summary_handles_minimal_bill() -> None:
    s = congress_ingest._parse_summary(
        {
            "congress": 119,
            "type": "HR",
            "number": 1234,
            "title": "Test Bill",
            "introducedDate": "2025-03-01",
            "latestAction": {"actionDate": "2025-04-15", "text": "Referred to committee."},
            "policyArea": {"name": "Agriculture and Food"},
            "url": "https://api.congress.gov/v3/bill/119/hr/1234?format=json",
        }
    )
    assert s.congress == 119
    assert s.bill_type == "hr"
    assert s.number == 1234
    assert s.bill_id == "hr-1234-119"
    assert s.introduced_date == date(2025, 3, 1)
    assert s.latest_action_date == date(2025, 4, 15)
    assert s.policy_area == "Agriculture and Food"


def test_parse_summary_handles_missing_fields() -> None:
    s = congress_ingest._parse_summary({})
    assert s.congress == 0
    assert s.bill_type == ""
    assert s.number == 0
    assert s.title == ""
    assert s.introduced_date is None
    assert s.latest_action_date is None
    assert s.policy_area is None


def test_get_bill_detail_validates_bill_type() -> None:
    with pytest.raises(ValueError, match="bill_type"):
        congress_ingest.get_bill_detail(119, "xyz", 1)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("CONGRESS_GOV_API_KEY"),
    reason="CONGRESS_GOV_API_KEY env var not set; skipping integration test",
)
def test_list_recent_bills_integration(tmp_path) -> None:
    bills = congress_ingest.list_recent_bills(
        congress=119,
        limit=5,
        cache_dir=tmp_path,
    )
    assert len(bills) >= 1
    b = bills[0]
    assert b.congress == 119
    assert b.bill_type in {"hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", "sres"}
    assert b.number > 0
    assert b.title
