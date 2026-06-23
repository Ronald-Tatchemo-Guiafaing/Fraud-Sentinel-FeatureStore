"""Generate EDA and ML figures for all three datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fraud_sentinel.offline_store import OfflineFeatureStore

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OFFLINE = ROOT / "data_store" / "offline"
DS1, DS2, DS3 = "dataset_1_ulb_creditcard", "dataset_2_ieee_cis", "dataset_3_sparkov"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

PLOT_SAMPLE = 80_000


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def _read_cols(name: str, cols: list[str]) -> pd.DataFrame:
    path = OFFLINE / f"{name}.parquet"
    return pd.read_parquet(path, columns=[c for c in cols if c in pd.read_parquet(path, engine="pyarrow").columns])


def _load_for_plots(name: str, amount_col: str, hour_col: str) -> pd.DataFrame:
    path = OFFLINE / f"{name}.parquet"
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    cols = ["Class", amount_col, hour_col]
    parts = []
    n = 0
    for rg in range(pf.num_row_groups):
        parts.append(pf.read_row_group(rg, columns=cols).to_pandas())
        n += len(parts[-1])
        if n >= PLOT_SAMPLE:
            break
    df = pd.concat(parts, ignore_index=True)
    if len(df) > PLOT_SAMPLE:
        return df.sample(PLOT_SAMPLE, random_state=42)
    return df


def _class_balance_from_counts(legit: int, fraud: int, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Legitimate (0)", "Fraud (1)"], [legit, fraud], color=["#58a6ff", "#f85149"])
    ax.set_title(title)
    ax.set_ylabel("Transactions")
    total = legit + fraud
    rate = (fraud / total * 100) if total else 0
    ax.text(0.5, 0.92, f"Fraud rate: {rate:.3f}% ({fraud:,}/{total:,})", transform=ax.transAxes, ha="center", fontsize=9)
    _save(fig, fname)


def _amount_dist(df: pd.DataFrame, amount_col: str, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    cap = df[amount_col].quantile(0.99)
    for label, color in [(0, "#58a6ff"), (1, "#f85149")]:
        sub = df.loc[df["Class"] == label, amount_col].clip(upper=cap)
        ax.hist(sub, bins=50, alpha=0.6, label=f"Class {label}", color=color, density=True)
    ax.set_title(title + f" (sample n={len(df):,})")
    ax.set_xlabel(amount_col)
    ax.legend()
    _save(fig, fname)


def _hourly_fraud(df: pd.DataFrame, hour_col: str, title: str, fname: str) -> None:
    hourly = df.groupby(hour_col)["Class"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    hourly.plot(kind="bar", ax=ax, color="#3fb950")
    ax.set_title(title + f" (sample n={len(df):,})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate (%)")
    _save(fig, fname)


def _metrics_bar(metrics: dict, title: str, fname: str) -> None:
    models = list(metrics.keys())
    f1s = [metrics[m]["f1"] for m in models]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, f1s, color=["#58a6ff", "#3fb950"])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("F1 score")
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    _save(fig, fname)


def _three_institutions(summary: dict) -> None:
    d = summary["datasets"] if "datasets" in summary else summary
    names = ["ULB", "IEEE-CIS", "Sparkov"]
    keys = [DS1, DS2, DS3]
    rows = [d[k]["rows"] for k in keys]
    rates = [d[k]["fraud_rate"] * 100 for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, rows, color=["#58a6ff", "#d29922", "#3fb950"])
    axes[0].set_title("Three CNP institutions — REAL row counts")
    axes[0].set_ylabel("Transactions")
    for i, v in enumerate(rows):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(names, rates, color=["#58a6ff", "#d29922", "#3fb950"])
    axes[1].set_title("Fraud rate per institution (%)")
    fig.suptitle("Three distinct sources — NOT V1/V2/V3 pipeline stages", fontsize=11, y=1.02)
    _save(fig, "19_three_homogeneous_datasets.png")


def generate_all() -> None:
    summary = json.loads((ROOT / "data_store" / "pipeline_summary.json").read_text(encoding="utf-8"))
    d = summary["datasets"]

    # ULB
    ulb_counts = pd.read_parquet(OFFLINE / f"{DS1}.parquet", columns=["Class"])["Class"].value_counts()
    _class_balance_from_counts(int(ulb_counts.get(0, 0)), int(ulb_counts.get(1, 0)),
                              "Dataset 1 ULB — class balance", "d1_ulb_class_balance.png")
    ulb = _load_for_plots(DS1, "Amount", "hour_of_day")
    _amount_dist(ulb, "Amount", "Dataset 1 ULB — amount distribution", "d1_ulb_amount_distribution.png")
    _hourly_fraud(ulb, "hour_of_day", "Dataset 1 ULB — hourly fraud rate", "d1_ulb_hourly_fraud.png")

    # IEEE
    ieee_counts = pd.read_parquet(OFFLINE / f"{DS2}.parquet", columns=["Class"])["Class"].value_counts()
    _class_balance_from_counts(int(ieee_counts.get(0, 0)), int(ieee_counts.get(1, 0)),
                              "Dataset 2 IEEE-CIS — class balance", "d2_ieee_class_balance.png")
    ieee = _load_for_plots(DS2, "TransactionAmt", "hour_of_day")
    _amount_dist(ieee, "TransactionAmt", "Dataset 2 IEEE-CIS — amount distribution", "d2_ieee_amount_distribution.png")
    _hourly_fraud(ieee, "hour_of_day", "Dataset 2 IEEE-CIS — hourly fraud rate", "d2_ieee_hourly_fraud.png")

    # Sparkov
    sp_counts = pd.read_parquet(OFFLINE / f"{DS3}.parquet", columns=["Class"])["Class"].value_counts()
    _class_balance_from_counts(int(sp_counts.get(0, 0)), int(sp_counts.get(1, 0)),
                              "Dataset 3 Sparkov — class balance", "d3_sparkov_class_balance.png")
    sparkov = _load_for_plots(DS3, "amt", "hour_of_day")
    _amount_dist(sparkov, "amt", "Dataset 3 Sparkov — amount distribution", "d3_sparkov_amount_distribution.png")
    _hourly_fraud(sparkov, "hour_of_day", "Dataset 3 Sparkov — hourly fraud rate", "d3_sparkov_hourly_fraud.png")

    for legacy, new in [
        ("01_class_balance.png", "d1_ulb_class_balance.png"),
        ("02_amount_distribution.png", "d1_ulb_amount_distribution.png"),
        ("03_temporal_patterns.png", "d1_ulb_hourly_fraud.png"),
    ]:
        src = FIG / new
        if src.exists():
            shutil.copy(src, FIG / legacy)

    metrics = json.loads((FIG / "metrics.json").read_text(encoding="utf-8"))
    _metrics_bar(metrics["dataset_1_ulb"], "Dataset 1 ULB — ML metrics", "d1_ulb_ml_metrics.png")
    _metrics_bar(metrics["dataset_2_ieee"], "Dataset 2 IEEE-CIS — ML metrics", "d2_ieee_ml_metrics.png")
    _metrics_bar(metrics["dataset_3_sparkov"], "Dataset 3 Sparkov — ML metrics", "d3_sparkov_ml_metrics.png")
    _metrics_bar(metrics["dataset_1_ulb"], "ULB ML benchmark", "08_model_metrics_comparison.png")
    _three_institutions(summary)

    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    generate_all()
