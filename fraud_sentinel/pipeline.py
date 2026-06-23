"""
End-to-end pipeline: 3 real CNP fraud datasets + offline/online Feature Store.

Each dataset: load real data → Feature Engineering → Parquet offline → metadata.
ULB + IEEE + Sparkov all get amount_log, hour_of_day, amount_zscore, fraud_score_rf/mlp.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loaders import load_ieee_cis, load_sparkov, load_ulb
from .ml_utils import RANDOM_STATE, train_and_score
from .offline_store import OfflineFeatureStore
from .online_store import OnlineFeatureStore
from .metadata_registry import (
    build_batch_metadata,
    build_dataset_metadata,
    build_sync_metadata,
    persist_metadata_bundle,
)

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data_store"
OFFLINE = STORE / "offline"
ONLINE_DB = STORE / "online" / "features.db"
FIGURES = ROOT / "figures"

DS1 = "dataset_1_ulb_creditcard"
DS2 = "dataset_2_ieee_cis"
DS3 = "dataset_3_sparkov"


def enrich_ulb(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["amount_log"] = np.log1p(out["Amount"])
    out["hour_of_day"] = (out["Time"] / 3600) % 24
    out["amount_zscore"] = (out["Amount"] - out["Amount"].mean()) / (out["Amount"].std() + 1e-9)
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return train_and_score(out, feature_cols)


def enrich_ieee(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["amount_log"] = np.log1p(out["TransactionAmt"].clip(lower=0))
    out["hour_of_day"] = (out["TransactionDT"] / 3600) % 24
    out["amount_zscore"] = (
        (out["TransactionAmt"] - out["TransactionAmt"].mean())
        / (out["TransactionAmt"].std() + 1e-9)
    )
    out["card1_freq"] = out.groupby("card1")["card1"].transform("count").astype(float)
    out["email_domain_enc"] = pd.factorize(out["P_emaildomain"])[0].astype(float)
    out["device_mobile"] = (out["DeviceType"].str.lower() == "mobile").astype(int)
    out["product_high_risk"] = out["ProductCD"].isin(["H", "S"]).astype(int)

    numeric = [
        "TransactionDT", "TransactionAmt", "card1", "addr1",
        "amount_log", "hour_of_day", "amount_zscore",
        "card1_freq", "email_domain_enc", "device_mobile", "product_high_risk",
    ]
    return train_and_score(out, numeric)


def enrich_sparkov(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["amount_log"] = np.log1p(out["amt"].clip(lower=0))
    if "unix_time" in out.columns:
        out["hour_of_day"] = (pd.to_datetime(out["unix_time"], unit="s").dt.hour).astype(float)
    else:
        out["hour_of_day"] = (
            pd.to_datetime(out["trans_date"] + " " + out["trans_time"]).dt.hour
        ).astype(float)
    out["amount_zscore"] = (out["amt"] - out["amt"].mean()) / (out["amt"].std() + 1e-9)
    out["category_enc"] = pd.factorize(out["category"])[0].astype(float)
    out["customer_velocity"] = out.groupby("cc_num")["cc_num"].transform("count").astype(float)
    out["state_risk"] = out.groupby("state")["is_fraud"].transform("mean").astype(float)

    numeric = [
        "amt", "amount_log", "hour_of_day", "amount_zscore",
        "category_enc", "customer_velocity", "state_risk",
        "city_pop", "lat", "long", "merch_lat", "merch_long",
    ]
    numeric = [c for c in numeric if c in out.columns]
    return train_and_score(out, numeric)


def _quality_checks(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "duplicates": int(df.duplicated().sum()),
        "nulls": int(df.isnull().sum().sum()),
        "fraud_rate": round(float(df["Class"].mean()), 6),
    }


def run_pipeline(sync_online_per_dataset: int = 2000) -> dict:
    print("Loading Dataset 1 — ULB…")
    ulb, m1 = enrich_ulb(load_ulb())
    print(f"  ULB: {len(ulb):,} rows, fraud {ulb['Class'].mean():.4%}")

    print("Loading Dataset 2 — IEEE-CIS…")
    ieee, m2 = enrich_ieee(load_ieee_cis())
    print(f"  IEEE: {len(ieee):,} rows, fraud {ieee['Class'].mean():.4%}")

    print("Loading Dataset 3 — Sparkov…")
    sparkov, m3 = enrich_sparkov(load_sparkov())
    print(f"  Sparkov: {len(sparkov):,} rows, fraud {sparkov['Class'].mean():.4%}")

    offline = OfflineFeatureStore(OFFLINE)
    offline.clear_manifest()
    offline.write_version(DS1, ulb, {"domain": "cnp_card_fraud", "institution": "ULB", "dataset_number": 1})
    offline.write_version(DS2, ieee, {"domain": "cnp_card_fraud", "institution": "IEEE-CIS/Vesta", "dataset_number": 2})
    offline.write_version(DS3, sparkov, {"domain": "cnp_card_fraud", "institution": "Sparkov/FDB", "dataset_number": 3})

    ds1_meta = build_dataset_metadata(
        DS1, ulb, institution="ULB — Université Libre de Bruxelles",
        source="creditcard.csv (Kaggle mlg-ulb/creditcardfraud)",
        role="Dataset 1: full EDA, FE, ML, online sync",
        transformations=["amount_log", "hour_of_day", "amount_zscore", "fraud_score_rf", "fraud_score_mlp"],
        quality_checks=_quality_checks(ulb),
        fs_challenges=["#2 Data Quality", "#3 Batch vs Real-Time", "#4 Scalability"],
    )
    ds2_meta = build_dataset_metadata(
        DS2, ieee, institution="IEEE CIS + Vesta Corporation",
        source="IEEE-CIS Kaggle corpus via HuggingFace (590,540 rows)",
        role="Dataset 2: cross-institution FE + schema harmonization",
        transformations=[
            "amount_log", "hour_of_day", "amount_zscore", "card1_freq",
            "email_domain_enc", "device_mobile", "product_high_risk",
            "fraud_score_rf", "fraud_score_mlp",
        ],
        quality_checks=_quality_checks(ieee),
        fs_challenges=["#1 Feature Consistency", "#2 Data Quality", "#3 Batch vs Real-Time"],
    )
    ds3_meta = build_dataset_metadata(
        DS3, sparkov, institution="Sparkov — Fraud Dataset Benchmark",
        source="Kaggle kartik2112/fraud-detection (fraudTrain + fraudTest merged)",
        role="Dataset 3: scale ingestion + merchant/location FE",
        transformations=[
            "amount_log", "hour_of_day", "amount_zscore", "category_enc",
            "customer_velocity", "state_risk", "fraud_score_rf", "fraud_score_mlp",
        ],
        quality_checks=_quality_checks(sparkov),
        fs_challenges=["#2 Data Quality", "#3 Batch vs Real-Time", "#4 Scalability"],
    )

    batch_metas = []
    for i, (name, frame) in enumerate([(DS1, ulb), (DS2, ieee), (DS3, sparkov)], 1):
        sort_col = "Time" if "Time" in frame.columns else (
            "TransactionDT" if "TransactionDT" in frame.columns else "unix_time"
        )
        sorted_df = frame.sort_values(sort_col).reset_index(drop=True)
        n = len(sorted_df)
        chunk = n // 3
        for j in range(3):
            start = j * chunk
            end = n if j == 2 else (j + 1) * chunk
            bid = f"{name}_batch_{j + 1}"
            offline.write_batch(bid, sorted_df.iloc[start:end])
            batch_metas.append(build_batch_metadata(bid, end - start, f"{name} temporal window {j + 1}"))

    if ONLINE_DB.exists():
        ONLINE_DB.unlink()

    online = OnlineFeatureStore(ONLINE_DB)
    sync_totals = {}
    for name, frame in [(DS1, ulb), (DS2, ieee), (DS3, sparkov)]:
        n = online.upsert_batch(frame.head(sync_online_per_dataset), name)
        sync_totals[name] = n

    sync_meta = build_sync_metadata("all_three_datasets", sum(sync_totals.values()))
    catalog = persist_metadata_bundle(STORE, [ds1_meta, ds2_meta, ds3_meta], batch_metas, sync_meta)

    all_metrics = {"dataset_1_ulb": m1, "dataset_2_ieee": m2, "dataset_3_sparkov": m3}
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (FIGURES / "metrics_ulb.json").write_text(json.dumps(m1, indent=2), encoding="utf-8")

    summary = {
        "three_institution_datasets": True,
        "all_real_data": True,
        "datasets": {
            DS1: {"institution": "ULB", "rows": len(ulb), "fraud_rate": round(ulb["Class"].mean(), 6), "ml": m1},
            DS2: {"institution": "IEEE-CIS", "rows": len(ieee), "fraud_rate": round(ieee["Class"].mean(), 6), "ml": m2},
            DS3: {"institution": "Sparkov", "rows": len(sparkov), "fraud_rate": round(sparkov["Class"].mean(), 6), "ml": m3},
        },
        "offline_store": offline.summary(),
        "online_store": online.stats(),
        "online_sync_by_dataset": sync_totals,
        "metadata_total": catalog["total_metadata_records"],
    }
    (STORE / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2, default=str))
