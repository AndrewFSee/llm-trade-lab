"""Resolution tracking for event-driven hypotheses.

Looks up the current status of a trigger event (initially Congress.gov bills)
and maps it to a discrete resolution bucket so we can score the model's
predicted `event_probability` against the realized outcome.

Buckets:
  pending   - recently introduced, no significant action
  advanced  - passed at least one chamber
  passed    - became law (signed or veto-overridden)
  failed    - explicitly defeated or vetoed without override
  stalled   - no advancement after `stall_days` (default 180)
  unknown   - bill_id couldn't be parsed or fetch failed

Binary realized_outcome for calibration scoring:
  passed/advanced  -> 1.0
  stalled/failed   -> 0.0
  pending/unknown  -> None  (excluded from Brier/ECE)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Literal

from llm_trade_lab.data.congress_ingest import get_bill_detail

logger = logging.getLogger(__name__)

ResolutionStatus = Literal["pending", "advanced", "passed", "failed", "stalled", "unknown"]

DEFAULT_STALL_DAYS = 180


@dataclass
class Resolution:
    status: ResolutionStatus
    evidence: str
    resolution_date: date | None
    realized_outcome: float | None


def _parse_bill_id(bill_id: str) -> tuple[int, str, int] | None:
    """Parse 'hr-1234-119' format into (congress, bill_type, number)."""
    parts = bill_id.split("-")
    if len(parts) != 3:
        return None
    bill_type, number_s, congress_s = parts
    try:
        return int(congress_s), bill_type, int(number_s)
    except ValueError:
        return None


def _classify_action(action_text: str) -> ResolutionStatus:
    """Substring-match the latest_action_text against canonical patterns."""
    text = (action_text or "").lower()
    if any(p in text for p in (
        "became public law",
        "signed by president",
        "signed by the president",
        "veto override",
    )):
        return "passed"
    if any(p in text for p in (
        "vetoed by president",
        "vetoed by the president",
        "motion to table agreed",
        "failed of passage",
        "failed to pass",
    )):
        return "failed"
    if any(p in text for p in (
        "passed senate",
        "passed/agreed to in senate",
        "passed house",
        "passed/agreed to in house",
        "agreed to in senate",
        "agreed to in house",
        "received in the senate",
        "received in the house",
        "presented to president",
        "presented to the president",
    )):
        return "advanced"
    return "pending"


def _apply_stall(
    status: ResolutionStatus,
    action_date: date | None,
    today: date,
    stall_days: int,
) -> ResolutionStatus:
    """If still pending and last action is older than stall_days, mark stalled."""
    if status == "pending" and action_date is not None:
        if (today - action_date).days >= stall_days:
            return "stalled"
    return status


def _realized(status: ResolutionStatus) -> float | None:
    if status in ("passed", "advanced"):
        return 1.0
    if status in ("stalled", "failed"):
        return 0.0
    return None


def resolve_congress_bill(
    bill_id: str,
    *,
    today: date | None = None,
    stall_days: int = DEFAULT_STALL_DAYS,
) -> Resolution:
    """Look up current status of a Congress.gov bill and map to a Resolution.

    On any fetch error or unparseable bill_id, returns status='unknown' so the
    caller can decide whether to retry later or skip.
    """
    today = today or date.today()

    parsed = _parse_bill_id(bill_id)
    if parsed is None:
        return Resolution(
            status="unknown",
            evidence=f"bill_id {bill_id!r} did not parse as 'type-number-congress'",
            resolution_date=None,
            realized_outcome=None,
        )
    congress, bill_type, number = parsed

    try:
        detail = get_bill_detail(congress, bill_type, number, fetch_text=False)
    except Exception as e:
        logger.warning("Congress.gov fetch failed for %s: %s", bill_id, e)
        return Resolution(
            status="unknown",
            evidence=f"Congress.gov fetch failed: {e}",
            resolution_date=None,
            realized_outcome=None,
        )

    summary = detail.summary
    action_text = summary.latest_action_text or ""
    action_date = summary.latest_action_date

    status = _classify_action(action_text)
    status = _apply_stall(status, action_date, today, stall_days)

    return Resolution(
        status=status,
        evidence=action_text[:240],
        resolution_date=action_date,
        realized_outcome=_realized(status),
    )
