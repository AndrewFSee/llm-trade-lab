"""FRED ingest tests."""
from __future__ import annotations

import os

import pytest

from llm_trade_lab.data import fred_ingest


def test_get_key_raises_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        fred_ingest._get_key()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"),
    reason="FRED_API_KEY env var not set",
)
def test_unrate_integration(tmp_path) -> None:
    """Unemployment rate, monthly, recent window."""
    s = fred_ingest.get_series("UNRATE", start="2024-01-01", end="2024-12-31", cache_dir=tmp_path)
    assert len(s) >= 6
    assert s.min() >= 0 and s.max() <= 100
