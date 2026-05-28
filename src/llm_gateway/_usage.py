"""Optional SQLite usage tracking at /root/.llm-gateway/usage.db."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path("/root/.llm-gateway/usage.db")
_CONN: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _CONN.execute(
            """CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                app TEXT,
                task TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER,
                tokens_out INTEGER,
                success INTEGER NOT NULL DEFAULT 1
            )"""
        )
        _CONN.commit()
    return _CONN


def record(
    *,
    provider: str,
    model: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    task: str | None = None,
    app: str | None = None,
    success: bool = True,
) -> None:
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO usage (ts, app, task, provider, model, tokens_in, tokens_out, success) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                datetime.now(tz=timezone.utc).isoformat(),
                app, task, provider, model,
                tokens_in, tokens_out, int(success),
            ],
        )
        conn.commit()
    except Exception:
        pass  # never let tracking errors break the caller
