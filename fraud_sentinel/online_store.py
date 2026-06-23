"""Online feature store — low-latency serving for real-time fraud scoring."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class OnlineFeatureStore:
    """SQLite-backed online store for sub-millisecond feature lookups at inference."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_vectors (
                    transaction_id INTEGER,
                    dataset_id INTEGER,
                    amount_log REAL,
                    hour_of_day REAL,
                    amount_zscore REAL,
                    fraud_score_rf REAL,
                    fraud_score_mlp REAL,
                    updated_at TEXT,
                    PRIMARY KEY (transaction_id, dataset_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_version TEXT,
                    rows_synced INTEGER,
                    synced_at TEXT
                )
                """
            )

    def upsert_batch(self, df: pd.DataFrame, source_version: str) -> int:
        """Sync enriched features from offline store to online serving table."""
        now = datetime.now(timezone.utc).isoformat()
        cols = ["transaction_id", "dataset_id", "amount_log", "hour_of_day", "amount_zscore",
                "fraud_score_rf", "fraud_score_mlp", "updated_at"]
        rows = []
        ds_id = int(df["dataset_id"].iloc[0]) if "dataset_id" in df.columns else 1
        for _, r in df.iterrows():
            rows.append((
                int(r["transaction_id"]),
                int(r.get("dataset_id", ds_id)),
                float(r.get("amount_log", 0)),
                float(r.get("hour_of_day", 0)),
                float(r.get("amount_zscore", 0)),
                float(r.get("fraud_score_rf", 0.0)),
                float(r.get("fraud_score_mlp", 0.0)),
                now,
            ))
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                f"""
                INSERT OR REPLACE INTO feature_vectors
                ({", ".join(cols)})
                VALUES ({", ".join("?" * len(cols))})
                """,
                rows,
            )
            conn.execute(
                "INSERT INTO sync_log (source_version, rows_synced, synced_at) VALUES (?, ?, ?)",
                (source_version, len(rows), now),
            )
        return len(rows)

    def get_features(self, transaction_id: int, dataset_id: int = 1) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM feature_vectors WHERE transaction_id = ? AND dataset_id = ?",
                (transaction_id, dataset_id),
            ).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM feature_vectors").fetchone()[0]
            last_sync = conn.execute(
                "SELECT source_version, rows_synced, synced_at FROM sync_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return {
            "served_transactions": n,
            "last_sync": {
                "source_version": last_sync[0] if last_sync else None,
                "rows_synced": last_sync[1] if last_sync else 0,
                "synced_at": last_sync[2] if last_sync else None,
            },
        }

    def export_sample(self, n: int = 5) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM feature_vectors LIMIT {n}", conn
            )
