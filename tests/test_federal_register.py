"""Federal Register ingest tests. Offline by default."""
from __future__ import annotations

from datetime import date

import pytest

from llm_trade_lab.data import federal_register_ingest as fr


def test_parse_doc_basic() -> None:
    raw = {
        "document_number": "2025-12345",
        "title": "Test Rule on Fertilizer",
        "abstract": "Establishes new standards.",
        "type": "Rule",
        "publication_date": "2025-08-01",
        "effective_on": "2025-09-01",
        "agencies": [{"name": "Environmental Protection Agency"}],
        "topics": ["agriculture", "fertilizer"],
        "html_url": "https://www.federalregister.gov/d/2025-12345",
    }
    doc = fr._parse_doc(raw, body_text="hello")
    assert doc.document_number == "2025-12345"
    assert doc.publication_date == date(2025, 8, 1)
    assert doc.effective_date == date(2025, 9, 1)
    assert doc.agencies == ["Environmental Protection Agency"]
    assert doc.topics == ["agriculture", "fertilizer"]
    assert doc.body_text == "hello"


def test_parse_doc_handles_missing() -> None:
    doc = fr._parse_doc({})
    assert doc.document_number == ""
    assert doc.publication_date is None
    assert doc.agencies == []


@pytest.mark.integration
def test_fda_filter_integration(tmp_path) -> None:
    """FDA mostly publishes Notices (drug approvals, etc.), not Rules — so don't
    filter by type here. The point is to verify the agency filter actually works."""
    docs = fr.search_documents(
        since="2025-01-01",
        until="2025-12-31",
        agencies=["food-and-drug-administration"],
        per_page=5,
        cache_dir=tmp_path,
    )
    assert len(docs) >= 1
    for d in docs:
        assert d.document_number
        assert d.title
        agencies_str = " ".join(d.agencies).lower()
        assert "food and drug" in agencies_str, f"agency filter failed: {d.agencies}"
