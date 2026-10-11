"""SQLite storage: the four data tables, run records (for /trace) and a data version counter.

Data tables are written with pandas so extra columns in a judge's CSV are kept, as Annex B allows.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import settings

DATA_TABLES = ("orders", "plants", "transport_duties", "capacity_events")


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a connection (one per request; SQLite is cheap to open)."""
    path = db_path or settings.DB_PATH
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the bookkeeping tables. Data tables are created on first ingest."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT,
            request TEXT,
            answer_type TEXT,
            response_json TEXT,
            trace_json TEXT
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    conn.commit()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def read_table(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Read a data table; an absent table comes back as an empty frame."""
    if not table_exists(conn, name):
        return pd.DataFrame()
    return pd.read_sql_query(f"SELECT * FROM {name}", conn)


def write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame, mode: str = "replace") -> None:
    """Write a validated frame. mode: 'replace' or 'append'."""
    df.to_sql(name, conn, if_exists=mode, index=False)
    bump_data_version(conn)
    conn.commit()


def get_data_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key='data_version'").fetchone()
    return int(row["value"]) if row else 0


def bump_data_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('data_version', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
    )


def save_run(conn: sqlite3.Connection, run_id: str, request: str, answer_type: str,
             response: dict, trace: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, datetime.now(timezone.utc).isoformat(), request, answer_type,
         json.dumps(response, default=str), json.dumps(trace, default=str)),
    )
    conn.commit()


def load_trace(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT trace_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return json.loads(row["trace_json"]) if row else None
