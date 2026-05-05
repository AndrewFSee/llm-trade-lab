"""Tests for event resolution status classification."""
from __future__ import annotations

from datetime import date

import pytest

from llm_trade_lab.events.resolution import (
    _apply_stall,
    _classify_action,
    _parse_bill_id,
    _realized,
)


# ------------------------------------------------- parse_bill_id

@pytest.mark.parametrize(
    "bill_id, expected",
    [
        ("hr-1234-119", (119, "hr", 1234)),
        ("s-42-118", (118, "s", 42)),
        ("hjres-99-119", (119, "hjres", 99)),
        ("hres-27-119", (119, "hres", 27)),
        ("invalid", None),
        ("hr-1234", None),
        ("hr-abc-119", None),
        ("", None),
    ],
)
def test_parse_bill_id(bill_id, expected) -> None:
    assert _parse_bill_id(bill_id) == expected


# ------------------------------------------------- classify_action

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Became Public Law No: 119-42", "passed"),
        ("Signed by President.", "passed"),
        ("Vetoed by President.", "failed"),
        ("Failed of passage.", "failed"),
        ("Passed Senate without amendment.", "advanced"),
        ("Passed/agreed to in House: On motion...", "advanced"),
        ("Received in the Senate.", "advanced"),
        ("Referred to the Committee on Energy and Commerce.", "pending"),
        ("Read twice and referred to the Committee on Finance.", "pending"),
        ("Held at the desk.", "pending"),
        ("", "pending"),
    ],
)
def test_classify_action(text, expected) -> None:
    assert _classify_action(text) == expected


def test_classify_action_case_insensitive() -> None:
    assert _classify_action("BECAME PUBLIC LAW") == "passed"
    assert _classify_action("passed senate") == _classify_action("Passed Senate")


# ------------------------------------------------- apply_stall

def test_apply_stall_pending_old_becomes_stalled() -> None:
    today = date(2026, 5, 1)
    old_action = date(2025, 6, 1)  # ~330 days ago
    assert _apply_stall("pending", old_action, today, stall_days=180) == "stalled"


def test_apply_stall_pending_recent_stays_pending() -> None:
    today = date(2026, 5, 1)
    recent = date(2026, 4, 1)  # ~30 days ago
    assert _apply_stall("pending", recent, today, stall_days=180) == "pending"


def test_apply_stall_advanced_unaffected() -> None:
    today = date(2026, 5, 1)
    old_action = date(2025, 1, 1)  # very old
    # Advanced bills don't get re-classified to stalled
    assert _apply_stall("advanced", old_action, today, stall_days=180) == "advanced"


def test_apply_stall_no_action_date_stays_pending() -> None:
    today = date(2026, 5, 1)
    assert _apply_stall("pending", None, today, stall_days=180) == "pending"


# ------------------------------------------------- realized

def test_realized_outcomes() -> None:
    assert _realized("passed") == 1.0
    assert _realized("advanced") == 1.0
    assert _realized("stalled") == 0.0
    assert _realized("failed") == 0.0
    assert _realized("pending") is None
    assert _realized("unknown") is None
