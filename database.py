import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "fake_news_history.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            article TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            reliability REAL NOT NULL,
            signals TEXT NOT NULL,
            metrics TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_analysis(article, label, confidence, reliability, signals, metrics):
    conn = _connect()
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO analyses (created_at, article, label, confidence, reliability, signals, metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_iso,
            article,
            label,
            float(confidence),
            float(reliability),
            json.dumps(signals),
            json.dumps(metrics),
        ),
    )
    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()
    return analysis_id


def get_history():
    conn = _connect()
    rows = conn.execute(
        """
        SELECT id, created_at, article, label, confidence, reliability, signals, metrics
        FROM analyses
        ORDER BY id DESC
        """
    ).fetchall()
    conn.close()

    result = []
    for row in rows:
        try:
            signals_data = json.loads(row["signals"])
        except Exception:
            signals_data = []
        try:
            metrics_data = json.loads(row["metrics"])
        except Exception:
            metrics_data = {}

        result.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "article": row["article"],
                "label": row["label"],
                "confidence": row["confidence"],
                "reliability": row["reliability"],
                "signals": signals_data,
                "metrics": metrics_data,
            }
        )
    return result


def get_analysis_by_id(analysis_id):
    conn = _connect()
    row = conn.execute(
        """
        SELECT id, created_at, article, label, confidence, reliability, signals, metrics
        FROM analyses
        WHERE id = ?
        """,
        (analysis_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    try:
        signals_data = json.loads(row["signals"])
    except Exception:
        signals_data = []
    try:
        metrics_data = json.loads(row["metrics"])
    except Exception:
        metrics_data = {}

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "article": row["article"],
        "label": row["label"],
        "confidence": row["confidence"],
        "reliability": row["reliability"],
        "signals": signals_data,
        "metrics": metrics_data,
    }


def delete_analysis(analysis_id):
    conn = _connect()
    cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def clear_history():
    conn = _connect()
    conn.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()
