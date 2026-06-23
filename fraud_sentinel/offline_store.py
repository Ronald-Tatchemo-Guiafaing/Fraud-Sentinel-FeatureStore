"""Offline feature store — batch historical storage for model training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class OfflineFeatureStore:
    """Parquet-based offline store with versioned dataset snapshots."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"versions": [], "batches": []}

    def clear_manifest(self) -> None:
        """Reset manifest (removes stale v1_raw / v2 / v3 entries)."""
        self._save_manifest({"versions": [], "batches": []})

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def write_version(self, name: str, df: pd.DataFrame, meta: dict | None = None) -> Path:
        """Persist one institution dataset snapshot to Parquet."""
        out = self.root / f"{name}.parquet"
        df.to_parquet(out, index=False)
        manifest = self._load_manifest()
        entry = {
            "name": name,
            "path": str(out.name),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "fraud_rate": float(df["Class"].mean()) if "Class" in df.columns else (
                float(df["isFraud"].mean()) if "isFraud" in df.columns else None
            ),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
        }
        manifest["versions"] = [v for v in manifest["versions"] if v["name"] != name]
        manifest["versions"].append(entry)
        self._save_manifest(manifest)
        return out

    def write_batch(self, batch_id: str, df: pd.DataFrame) -> Path:
        """Temporal batch for point-in-time training windows."""
        batch_dir = self.root / "batches"
        batch_dir.mkdir(exist_ok=True)
        out = batch_dir / f"{batch_id}.parquet"
        df.to_parquet(out, index=False)
        manifest = self._load_manifest()
        manifest["batches"] = [b for b in manifest["batches"] if b.get("batch_id") != batch_id]
        manifest["batches"].append(
            {
                "batch_id": batch_id,
                "rows": int(len(df)),
                "path": str(out.relative_to(self.root)),
                "written_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_manifest(manifest)
        return out

    def read_version(self, name: str) -> pd.DataFrame:
        path = self.root / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    def summary(self) -> dict:
        return self._load_manifest()
