"""
Metadata registry — Fraud Sentinel Feature Store.

Form: JSON files under data_store/metadata/
Professor questions answered by: python show_metadata.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .feature_registry import REGISTRY

METADATA_FIELDS = [
    "id",
    "type",
    "name",
    "form",
    "domain",
    "stage",
    "status",
    "rows",
    "columns_count",
    "fraud_rate",
    "schema_hash",
    "source_file",
    "parent_version",
    "transformations_applied",
    "quality_checks",
    "known_issues",
    "lessons_from_failure",
    "created_at",
    "owner",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Versions we studied and REJECTED (professor: talk about failures of other versions)
REJECTED_VERSIONS: list[dict[str, Any]] = [
    {
        "id": "meta_rejected_001",
        "type": "dataset_version",
        "name": "mixed_creditcard_paysim",
        "form": "JSON metadata record",
        "domain": "mixed_mobile_money_and_card",
        "stage": "abandoned",
        "status": "FAILED",
        "rows": None,
        "failure_reason": "Professor feedback (8/20): datasets must be the SAME type. "
        "PaySim is mobile-money (M-Pesa); creditcard.csv is card-not-present banking. "
        "Different schema, fraud mechanism, and feature semantics.",
        "lessons_from_failure": "Use three distinct CNP datasets from different institutions (ULB, IEEE-CIS, Sparkov), not cross-domain mixing.",
        "created_at": "2026-06-17",
    },
    {
        "id": "meta_rejected_002",
        "type": "architecture",
        "name": "diagram_only_feature_store",
        "form": "JSON metadata record",
        "domain": "credit_card_fraud",
        "stage": "design_only",
        "status": "FAILED",
        "failure_reason": "Feature store existed only as PNG diagram — no offline Parquet or online SQLite.",
        "lessons_from_failure": "Implement data_store/offline + data_store/online with manifest.json (classmates scored 16/20).",
        "created_at": "2026-06-17",
    },
    {
        "id": "meta_rejected_003",
        "type": "evaluation",
        "name": "accuracy_only_metrics",
        "form": "JSON metadata record",
        "domain": "credit_card_fraud",
        "stage": "modeling",
        "status": "FAILED",
        "failure_reason": "Naive classifier reaches 99.83% accuracy with 0% fraud recall on 0.172% fraud rate.",
        "lessons_from_failure": "Metadata must track fraud-class F1, Precision, Recall, PR-AUC — not accuracy alone.",
        "created_at": "2026-06-17",
    },
    {
        "id": "meta_rejected_004",
        "type": "feature_store",
        "name": "no_metadata_manifest",
        "form": "JSON metadata record",
        "domain": "credit_card_fraud",
        "stage": "governance",
        "status": "FAILED",
        "failure_reason": "No manifest, no feature registry JSON, no version lineage — professor cannot audit versions.",
        "lessons_from_failure": "Add metadata_catalog.json + per-version sidecar .meta.json + feature_metadata.json.",
        "created_at": "2026-06-18",
    },
    {
        "id": "meta_rejected_005",
        "type": "dataset_version",
        "name": "v1_v2_v3_single_csv",
        "form": "JSON metadata record",
        "domain": "credit_card_fraud",
        "stage": "abandoned",
        "status": "FAILED",
        "failure_reason": "Treating V1 Raw / V2 Cleaned / V3 Enriched as three datasets — that is one CSV with three pipeline stages, not three institutions.",
        "lessons_from_failure": "Present three separate CNP corpora (ULB, IEEE-CIS, Sparkov), each fully described before the next.",
        "created_at": "2026-06-19",
    },
]


def build_feature_metadata() -> list[dict[str, Any]]:
    records = []
    for i, f in enumerate(REGISTRY, 1):
        records.append({
            "id": f"meta_feature_{i:03d}",
            "type": "feature",
            "name": f.name,
            "form": "JSON (feature_metadata.json)",
            "domain": "credit_card_fraud",
            "layer": f.layer,
            "source_columns": list(f.source_columns),
            "description": f.description,
            "dtype": "float64",
            "nullable": f.name not in ("fraud_score_rf", "fraud_score_mlp"),
            "served_online": f.layer in ("transformation", "aggregate", "prediction"),
            "point_in_time_correct": f.layer != "aggregate" or f.name == "tx_count_1h",
            "status": "ACTIVE",
            "created_at": _now(),
        })
    return records


def build_dataset_metadata(
    name: str,
    df,
    institution: str,
    source: str,
    role: str,
    transformations: list[str],
    quality_checks: dict,
    fs_challenges: list[str],
) -> dict[str, Any]:
    import hashlib
    cols = list(df.columns)
    schema_sig = hashlib.md5(",".join(cols).encode()).hexdigest()[:12]
    fraud_col = "Class" if "Class" in df.columns else "isFraud"
    return {
        "id": f"meta_{name}",
        "type": "dataset",
        "name": name,
        "form": "JSON sidecar + Parquet data file",
        "domain": "cnp_card_fraud",
        "institution": institution,
        "source": source,
        "role": role,
        "status": "ACTIVE",
        "rows": int(len(df)),
        "columns_count": len(cols),
        "columns": cols,
        "fraud_rate": round(float(df[fraud_col].mean()), 6),
        "schema_hash": schema_sig,
        "transformations_applied": transformations,
        "quality_checks": quality_checks,
        "fs_challenges_addressed": fs_challenges,
        "parquet_path": f"data_store/offline/{name}.parquet",
        "sidecar_path": f"data_store/metadata/datasets/{name}.meta.json",
        "created_at": _now(),
        "owner": "Ronald Tatchemo Guiafaing / IUC",
    }


def build_dataset_version_metadata(
    name: str,
    df,
    stage: str,
    parent: str | None,
    transformations: list[str],
    quality_checks: dict,
    known_issues: list[str],
) -> dict[str, Any]:
    import hashlib
    cols = list(df.columns)
    schema_sig = hashlib.md5(",".join(cols).encode()).hexdigest()[:12]
    return {
        "id": f"meta_dataset_{stage}",
        "type": "dataset_version",
        "name": name,
        "form": "JSON sidecar + Parquet data file",
        "domain": "credit_card_fraud",
        "stage": stage,
        "status": "ACTIVE",
        "rows": int(len(df)),
        "columns_count": len(cols),
        "columns": cols,
        "fraud_rate": round(float(df["Class"].mean()), 6) if "Class" in df.columns else None,
        "schema_hash": schema_sig,
        "source_file": "creditcard.csv" if stage == "raw" else f"parent:{parent}",
        "parent_version": parent,
        "transformations_applied": transformations,
        "quality_checks": quality_checks,
        "known_issues": known_issues,
        "lessons_from_failure": [],
        "parquet_path": f"data_store/offline/{name}.parquet",
        "sidecar_path": f"data_store/metadata/versions/{name}.meta.json",
        "created_at": _now(),
        "owner": "Ronald Tatchemo Guiafaing / IUC",
    }


def build_batch_metadata(batch_id: str, rows: int, time_range: str) -> dict[str, Any]:
    return {
        "id": f"meta_batch_{batch_id}",
        "type": "temporal_batch",
        "name": batch_id,
        "form": "JSON + Parquet",
        "domain": "credit_card_fraud",
        "stage": "offline_training_window",
        "status": "ACTIVE",
        "rows": rows,
        "time_range": time_range,
        "parquet_path": f"data_store/offline/batches/{batch_id}.parquet",
        "created_at": _now(),
    }


def build_sync_metadata(source_version: str, rows_synced: int) -> dict[str, Any]:
    return {
        "id": "meta_sync_001",
        "type": "online_sync",
        "name": "offline_to_online_sync",
        "form": "JSON + SQLite sync_log table",
        "domain": "credit_card_fraud",
        "source_version": source_version,
        "rows_synced": rows_synced,
        "target": "data_store/online/features.db",
        "features_synced": ["amount_log", "hour_of_day", "amount_zscore", "fraud_score_rf", "fraud_score_mlp"],
        "status": "ACTIVE",
        "created_at": _now(),
    }


def write_metadata_catalog(root: Path, catalog: dict) -> Path:
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "metadata_catalog.json"
    path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return path


def export_dataset_sidecar(meta_dir: Path, record: dict) -> None:
    datasets_dir = meta_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    name = record["name"]
    (datasets_dir / f"{name}.meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def export_version_sidecar(meta_dir: Path, record: dict) -> None:
    versions_dir = meta_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    name = record["name"]
    (versions_dir / f"{name}.meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def build_full_catalog(
    dataset_metas: list[dict],
    batch_metas: list[dict],
    sync_meta: dict,
) -> dict[str, Any]:
    feature_metas = build_feature_metadata()
    all_active = feature_metas + dataset_metas + batch_metas + [sync_meta]
    all_records = all_active + REJECTED_VERSIONS

    return {
        "catalog_id": "fraud_sentinel_metadata_catalog_v2",
        "form": "JSON",
        "description": "Master index — 3 institution datasets (ULB, IEEE-CIS, Sparkov)",
        "generated_at": _now(),
        "total_metadata_records": len(all_records),
        "counts_by_type": {
            "feature": len(feature_metas),
            "dataset_institution": len(dataset_metas),
            "dataset_version_rejected": len(REJECTED_VERSIONS),
            "temporal_batch": len(batch_metas),
            "online_sync": 1,
            "total_active": len(all_active),
            "total_including_rejected": len(all_records),
        },
        "metadata_files": {
            "catalog": "data_store/metadata/metadata_catalog.json",
            "features": "data_store/metadata/feature_metadata.json",
            "rejected_versions": "data_store/metadata/rejected_version_log.json",
            "dataset_sidecars": "data_store/metadata/datasets/*.meta.json",
            "offline_manifest": "data_store/offline/manifest.json",
            "pipeline_summary": "data_store/pipeline_summary.json",
        },
        "how_metadata_is_handled": [
            "Each of the 3 institution datasets has a JSON sidecar in metadata/datasets/.",
            "Feature definitions are stored in feature_metadata.json (7 features).",
            "Rejected/failed approaches are logged in rejected_version_log.json.",
            "offline/manifest.json tracks all 3 Parquet dataset files.",
            "metadata_catalog.json indexes ALL records for audit.",
        ],
        "active_records": all_active,
        "rejected_records": REJECTED_VERSIONS,
    }


def persist_metadata_bundle(
    store_root: Path,
    dataset_metas: list[dict],
    batch_metas: list[dict],
    sync_meta: dict,
) -> dict[str, Any]:
    meta_dir = store_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    feature_metas = build_feature_metadata()
    (meta_dir / "feature_metadata.json").write_text(
        json.dumps({"count": len(feature_metas), "records": feature_metas}, indent=2), encoding="utf-8"
    )
    (meta_dir / "rejected_version_log.json").write_text(
        json.dumps({"count": len(REJECTED_VERSIONS), "records": REJECTED_VERSIONS}, indent=2), encoding="utf-8"
    )

    for rec in dataset_metas:
        export_dataset_sidecar(meta_dir, rec)

    (meta_dir / "batch_metadata.json").write_text(
        json.dumps({"count": len(batch_metas), "records": batch_metas}, indent=2), encoding="utf-8"
    )
    (meta_dir / "sync_metadata.json").write_text(
        json.dumps(sync_meta, indent=2), encoding="utf-8"
    )

    catalog = build_full_catalog(dataset_metas, batch_metas, sync_meta)
    write_metadata_catalog(store_root, catalog)
    return catalog
