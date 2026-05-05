"""SQLite hypothesis ledger.

Two tables:
  - hypothesis: one row per generated hypothesis, stored as canonical JSON.
    `id` is a content hash so re-inserting the same hypothesis is idempotent.
  - backtest_result: one row per (hypothesis, run) — a hypothesis can be
    backtested on multiple universes/windows.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import TypeAdapter

from llm_trade_lab.backtest.engine import BacktestResult
from llm_trade_lab.schema.hypothesis import Hypothesis

_HYPOTHESIS_ADAPTER: TypeAdapter[Hypothesis] = TypeAdapter(Hypothesis)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypothesis (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    model_version_hash TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hypothesis_type ON hypothesis(type);
CREATE INDEX IF NOT EXISTS idx_hypothesis_model ON hypothesis(model_version_hash);

CREATE TABLE IF NOT EXISTS backtest_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    universe_ticker TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    sharpe REAL,
    sortino REAL,
    return_pct REAL,
    max_drawdown_pct REAL,
    n_trades INTEGER,
    win_rate_pct REAL,
    cost_bps REAL,
    raw_stats_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypothesis(id)
);

CREATE INDEX IF NOT EXISTS idx_bt_hypothesis ON backtest_result(hypothesis_id);

CREATE TABLE IF NOT EXISTS event_resolution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL UNIQUE,
    trigger_event_doc_id TEXT NOT NULL,
    trigger_event_source TEXT NOT NULL,
    resolution_status TEXT NOT NULL,
    resolution_evidence TEXT,
    resolution_date TEXT,
    realized_outcome REAL,
    p_event_predicted REAL,
    checked_at TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypothesis(id)
);

CREATE INDEX IF NOT EXISTS idx_er_status ON event_resolution(resolution_status);
"""


def _hypothesis_id(h: Hypothesis) -> str:
    canonical = _HYPOTHESIS_ADAPTER.dump_json(h, by_alias=False).decode()
    canonical = json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert_hypothesis(self, h: Hypothesis) -> str:
        hid = _hypothesis_id(h)
        payload = _HYPOTHESIS_ADAPTER.dump_json(h).decode()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO hypothesis
                    (id, type, name, model_version_hash, generated_at, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hid,
                    h.type,
                    h.name,
                    h.model_version_hash,
                    h.generated_at.isoformat(),
                    payload,
                    datetime.utcnow().isoformat(),
                ),
            )
        return hid

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hypothesis WHERE id = ?",
                (hypothesis_id,),
            ).fetchone()
        if row is None:
            return None
        return _HYPOTHESIS_ADAPTER.validate_json(row[0])

    def insert_backtest_result(
        self,
        hypothesis_id: str,
        universe_ticker: str,
        window_start: str,
        window_end: str,
        result: BacktestResult,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_result
                    (hypothesis_id, universe_ticker, window_start, window_end,
                     sharpe, sortino, return_pct, max_drawdown_pct,
                     n_trades, win_rate_pct, cost_bps, raw_stats_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis_id,
                    universe_ticker,
                    window_start,
                    window_end,
                    result.sharpe,
                    result.sortino,
                    result.return_pct,
                    result.max_drawdown_pct,
                    result.n_trades,
                    result.win_rate_pct,
                    result.cost_bps,
                    json.dumps(result.raw_stats),
                    datetime.utcnow().isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def query_results(self, hypothesis_id: str) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM backtest_result WHERE hypothesis_id = ? ORDER BY id",
                (hypothesis_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def iter_hypotheses(
        self, *, hypothesis_type: str | None = None
    ) -> list[tuple[str, Hypothesis]]:
        """Return all (id, Hypothesis) pairs in the ledger, optionally filtered by type.

        Order is by created_at ascending. Returns a list (not a true iterator) so
        callers can take len(); fine through millions of rows on SQLite.
        """
        with self._conn() as conn:
            if hypothesis_type:
                rows = conn.execute(
                    "SELECT id, payload_json FROM hypothesis WHERE type = ? ORDER BY created_at",
                    (hypothesis_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, payload_json FROM hypothesis ORDER BY created_at"
                ).fetchall()
        out: list[tuple[str, Hypothesis]] = []
        for hid, payload in rows:
            try:
                out.append((hid, _HYPOTHESIS_ADAPTER.validate_json(payload)))
            except Exception:
                continue
        return out

    def query_all_results(self) -> list[dict]:
        """All backtest_result rows in the ledger. Used by eval harness."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM backtest_result ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    # -------- event_resolution --------

    def upsert_event_resolution(
        self,
        hypothesis_id: str,
        *,
        trigger_event_doc_id: str,
        trigger_event_source: str,
        resolution_status: str,
        resolution_evidence: str = "",
        resolution_date: str | None = None,
        realized_outcome: float | None = None,
        p_event_predicted: float,
    ) -> None:
        """Insert or replace the resolution row for one event-driven hypothesis."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO event_resolution
                    (hypothesis_id, trigger_event_doc_id, trigger_event_source,
                     resolution_status, resolution_evidence, resolution_date,
                     realized_outcome, p_event_predicted, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hypothesis_id) DO UPDATE SET
                    resolution_status = excluded.resolution_status,
                    resolution_evidence = excluded.resolution_evidence,
                    resolution_date = excluded.resolution_date,
                    realized_outcome = excluded.realized_outcome,
                    checked_at = excluded.checked_at
                """,
                (
                    hypothesis_id,
                    trigger_event_doc_id,
                    trigger_event_source,
                    resolution_status,
                    resolution_evidence,
                    resolution_date,
                    realized_outcome,
                    p_event_predicted,
                    datetime.utcnow().isoformat(),
                ),
            )

    def get_event_resolution(self, hypothesis_id: str) -> dict | None:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM event_resolution WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
        return dict(row) if row else None

    def iter_event_resolutions(
        self, *, status_filter: str | None = None
    ) -> list[dict]:
        """All event_resolution rows, optionally filtered by status."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM event_resolution WHERE resolution_status = ? ORDER BY id",
                    (status_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM event_resolution ORDER BY id"
                ).fetchall()
        return [dict(r) for r in rows]
