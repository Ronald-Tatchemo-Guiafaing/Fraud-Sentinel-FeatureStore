"""
Load real CNP fraud datasets — no synthetic stand-ins.

Dataset 1: ULB creditcard.csv (local)
Dataset 2: IEEE-CIS / Vesta — 590,540 rows via HuggingFace (Kaggle corpus)
Dataset 3: Sparkov — fraudTrain + fraudTest via kagglehub (Kaggle kartik2112/fraud-detection)
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ULB_CSV = ROOT / "creditcard.csv"
IEEE_CACHE = RAW / "ieee" / "ieee_cis.parquet"
SPARKOV_CACHE = RAW / "sparkov" / "sparkov_merged.parquet"
SPARKOV_DIR = RAW / "sparkov"
KAGGLE_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "kartik2112" / "fraud-detection"


def load_ulb() -> pd.DataFrame:
    if not ULB_CSV.exists():
        raise FileNotFoundError(f"ULB file missing: {ULB_CSV}")
    df = pd.read_csv(ULB_CSV)
    df["Class"] = df["Class"].astype(int)
    df["transaction_id"] = np.arange(len(df), dtype=np.int64)
    df["dataset_id"] = 1
    df["institution"] = "ULB"
    return df


def load_ieee_cis() -> pd.DataFrame:
    """IEEE-CIS train_transaction corpus (590,540 rows). Cached as Parquet after first load."""
    IEEE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if IEEE_CACHE.exists():
        return pd.read_parquet(IEEE_CACHE)

    from datasets import load_dataset

    hf = load_dataset("Kshitijbhatt1998/ieee-fraud-detection-pipeline-features", split="train")
    df = hf.to_pandas()

    out = pd.DataFrame({
        "TransactionID": df["transaction_id"].astype(np.int64),
        "TransactionDT": df["transaction_dt"].astype(np.int64),
        "TransactionAmt": df["transaction_amt"].astype(float),
        "card1": df["card1"].fillna(-1).astype(float),
        "card4": df["card4"].fillna("unknown").astype(str),
        "ProductCD": df["product_cd"].fillna("unknown").astype(str),
        "addr1": df["addr1"].fillna(-1).astype(float),
        "P_emaildomain": df["purchaser_email_domain"].fillna("unknown").astype(str),
        "DeviceType": df["device_type"].fillna("unknown").astype(str),
        "isFraud": df["is_fraud"].astype(int),
        "Class": df["is_fraud"].astype(int),
        "transaction_id": df["transaction_id"].astype(np.int64),
        "dataset_id": 2,
        "institution": "IEEE-CIS/Vesta",
    })
    out.to_parquet(IEEE_CACHE, index=False)
    return out


def _find_sparkov_csvs() -> tuple[Path, Path] | None:
    SPARKOV_DIR.mkdir(parents=True, exist_ok=True)
    train = SPARKOV_DIR / "fraudTrain.csv"
    test = SPARKOV_DIR / "fraudTest.csv"
    if train.exists() and test.exists():
        return train, test

    for folder in [KAGGLE_CACHE, SPARKOV_DIR]:
        if not folder.exists():
            continue
        trains = list(folder.rglob("fraudTrain.csv"))
        tests = list(folder.rglob("fraudTest.csv"))
        if trains and tests:
            return trains[0], tests[0]

    archive = KAGGLE_CACHE / "1.archive"
    if archive.exists():
        with zipfile.ZipFile(archive, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                target = SPARKOV_DIR / Path(name).name
                if not target.exists():
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        if train.exists() and test.exists():
            return train, test
    return None


def _download_sparkov_csvs() -> tuple[Path, Path]:
    found = _find_sparkov_csvs()
    if found:
        return found

    import kagglehub

    try:
        cache_dir = Path(kagglehub.dataset_download("kartik2112/fraud-detection"))
    except PermissionError:
        cache_dir = KAGGLE_CACHE

    found = _find_sparkov_csvs()
    if found:
        return found

    csvs = list(cache_dir.rglob("*.csv")) if cache_dir.exists() else []
    train_files = [p for p in csvs if "train" in p.name.lower()]
    test_files = [p for p in csvs if "test" in p.name.lower()]
    if train_files and test_files:
        return train_files[0], test_files[0]

    raise FileNotFoundError(
        "Sparkov CSVs not found. Download kartik2112/fraud-detection from Kaggle "
        f"into {SPARKOV_DIR}"
    )


def load_sparkov() -> pd.DataFrame:
    """Sparkov simulated CNP corpus — fraudTrain + fraudTest merged (~1.85M rows)."""
    SPARKOV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if SPARKOV_CACHE.exists():
        return pd.read_parquet(SPARKOV_CACHE)

    train_path, test_path = _download_sparkov_csvs()
    parts = [pd.read_csv(train_path), pd.read_csv(test_path)]
    raw = pd.concat(parts, ignore_index=True)
    raw["is_fraud"] = raw["is_fraud"].astype(int)
    raw["Class"] = raw["is_fraud"]
    raw["transaction_id"] = np.arange(len(raw), dtype=np.int64)
    raw["dataset_id"] = 3
    raw["institution"] = "Sparkov/FDB"
    raw.to_parquet(SPARKOV_CACHE, index=False)
    return raw
