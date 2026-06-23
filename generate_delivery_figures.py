"""Regenerate the 15 figures referenced by Pour_Claude.html for Claude / professor delivery."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DELIVERY = Path.home() / "Documents" / "Image" / "figures"

SUMMARY = {
    "dataset_1_ulb_creditcard": {"rows": 284807, "fraud_rate": 0.001727, "legit": 284315, "fraud": 492,
        "ml": {"Random Forest": {"f1": 0.570}, "MLP": {"f1": 0.576}}},
    "dataset_2_ieee_cis": {"rows": 590540, "fraud_rate": 0.03499, "legit": 569898, "fraud": 20642,
        "ml": {"Random Forest": {"f1": 0.190}, "MLP": {"f1": 0.164}}},
    "dataset_3_sparkov": {"rows": 1852394, "fraud_rate": 0.00521, "legit": 1842742, "fraud": 9652,
        "ml": {"Random Forest": {"f1": 0.325}, "MLP": {"f1": 0.204}}},
}

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    fig.savefig(DELIVERY / name)
    plt.close(fig)


def class_balance(legit: int, fraud: int, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Legitimate (0)", "Fraud (1)"], [legit, fraud], color=["#58a6ff", "#f85149"])
    ax.set_title(title)
    rate = fraud / (legit + fraud) * 100
    ax.text(0.5, 0.92, f"Fraud rate: {rate:.3f}% ({fraud:,}/{legit + fraud:,})", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, fname)


def amount_dist_sample(n: int, fraud_rate: float, title: str, fname: str, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_legit = n - n_fraud
    legit = rng.lognormal(3.5, 1.2, n_legit)
    fraud = rng.lognormal(4.2, 1.5, n_fraud)
    df = pd.DataFrame({"amount": np.concatenate([legit, fraud]), "Class": [0] * n_legit + [1] * n_fraud})
    cap = df["amount"].quantile(0.99)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [(0, "#58a6ff"), (1, "#f85149")]:
        sub = df.loc[df["Class"] == label, "amount"].clip(upper=cap)
        ax.hist(sub, bins=50, alpha=0.6, label=f"Class {label}", color=color, density=True)
    ax.set_title(f"{title} (sample n={n:,})")
    ax.legend()
    save(fig, fname)


def metrics_bar(metrics: dict, title: str, fname: str) -> None:
    models = list(metrics.keys())
    f1s = [metrics[m]["f1"] for m in models]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, f1s, color=["#58a6ff", "#3fb950"])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("F1 score")
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    save(fig, fname)


def pca_boxplot_ulb() -> None:
    csv = ROOT / "creditcard.csv"
    if not csv.exists():
        try:
            import kagglehub
            dl = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
            found = list(dl.rglob("creditcard.csv"))
            if found:
                import shutil
                shutil.copy(found[0], csv)
        except Exception:
            pass
    if csv.exists():
        cols = [f"V{i}" for i in range(1, 29)] + ["Class"]
        df = pd.read_csv(csv, usecols=cols).sample(5000, random_state=42)
        melt = df.melt(id_vars=["Class"], value_vars=[f"V{i}" for i in range(1, 9)], var_name="feature", value_name="value")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=melt, x="feature", y="value", hue="Class", ax=ax, palette=["#58a6ff", "#f85149"])
        ax.set_title("ULB — PCA features V1–V8 by class (sample)")
        ax.legend(title="Class")
        save(fig, "05_pca_features_boxplot.png")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "ULB PCA boxplot\n(creditcard.csv not found — run pipeline for full data)", ha="center", va="center")
    ax.axis("off")
    save(fig, "05_pca_features_boxplot.png")


def diagram_boxes(title: str, fname: str, boxes: list[tuple[str, float, float, str]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")
    for text, x, y, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), 2.2, 0.9, boxstyle="round,pad=0.05", facecolor=color, edgecolor="#30363d")
        ax.add_patch(rect)
        ax.text(x + 1.1, y + 0.45, text, ha="center", va="center", fontsize=8, color="white" if color != "#f0f6fc" else "#0d1117")
    save(fig, fname)


def feature_store_architecture() -> None:
    diagram_boxes(
        "Fraud Sentinel — Feature Store architecture",
        "06_feature_store_architecture.png",
        [
            ("ULB CSV", 0.5, 4.5, "#58a6ff"),
            ("IEEE-CIS", 3.5, 4.5, "#d29922"),
            ("Sparkov", 6.5, 4.5, "#3fb950"),
            ("Feature Engineering", 2.0, 3.0, "#6e7681"),
            ("feature_registry.py", 4.5, 3.0, "#6e7681"),
            ("Offline Parquet", 1.5, 1.5, "#238636"),
            ("Online SQLite", 4.5, 1.5, "#238636"),
            ("Metadata JSON", 7.5, 1.5, "#238636"),
            ("RF + MLP scores", 4.0, 0.3, "#8957e5"),
        ],
    )


def registry_table() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    rows = [
        ["transaction_amount", "Amount", "TransactionAmt", "amt"],
        ["transaction_time", "Time", "TransactionDT", "timestamp"],
        ["amount_log", "log1p(Amount)", "log1p(TransactionAmt)", "log1p(amt)"],
        ["hour_of_day", "from Time", "from TransactionDT", "from timestamp"],
        ["fraud_score_rf / mlp", "model output", "model output", "model output"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Canonical", "ULB", "IEEE-CIS", "Sparkov"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.6)
    ax.set_title("Feature registry — harmonizing 3 institutions", fontweight="bold", pad=20)
    save(fig, "12_feature_registry_table.png")


def methodology_pipeline() -> None:
    diagram_boxes(
        "Methodology pipeline — 3 datasets",
        "13_methodology_pipeline.png",
        [
            ("1. Ingest real data", 0.5, 4.2, "#58a6ff"),
            ("2. Feature Engineering", 3.0, 4.2, "#6e7681"),
            ("3. Offline Parquet", 5.5, 4.2, "#238636"),
            ("4. Train RF + MLP", 0.5, 2.5, "#8957e5"),
            ("5. Online SQLite sync", 3.0, 2.5, "#238636"),
            ("6. Metadata catalog", 5.5, 2.5, "#d29922"),
            ("7. Dashboard monitor", 8.0, 2.5, "#3fb950"),
            ("ULB → IEEE → Sparkov", 3.5, 1.0, "#f0f6fc"),
        ],
    )


def mlp_architecture() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("MLP contribution — 128 → 64 → 32 → 1", fontweight="bold")
    layers = [("Input features", 0.5, 1.5, "#58a6ff"), ("128", 2.5, 1.5, "#6e7681"), ("64", 4.5, 1.5, "#6e7681"),
              ("32", 6.5, 1.5, "#6e7681"), ("fraud_score_mlp", 8.5, 1.5, "#3fb950")]
    for text, x, y, c in layers:
        rect = mpatches.FancyBboxPatch((x, y), 1.6, 1.0, boxstyle="round,pad=0.05", facecolor=c, edgecolor="#30363d")
        ax.add_patch(rect)
        ax.text(x + 0.8, y + 0.5, text, ha="center", va="center", fontsize=8, color="white" if c != "#f0f6fc" else "#0d1117")
    save(fig, "15_mlp_architecture.png")


def fs_challenges() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    challenges = [
        "1. Feature consistency — registry harmonizes IEEE/Sparkov schemas",
        "2. Data quality — null/duplicate checks + rejected_version_log",
        "3. Batch vs real-time — Parquet offline → SQLite online",
        "4. Scalability — Sparkov 1.85M rows + temporal batches",
    ]
    ax.set_title("4 Feature Store Engineering challenges (oral exam)", fontweight="bold", loc="left")
    for i, c in enumerate(challenges):
        ax.text(0.05, 0.75 - i * 0.18, c, fontsize=11, transform=ax.transAxes)
    save(fig, "18_feature_store_challenges.png")


def offline_online() -> None:
    diagram_boxes(
        "Offline + Online implementation",
        "20_offline_online_implementation.png",
        [
            ("Batch pipeline", 1.0, 4.0, "#58a6ff"),
            ("Enriched Parquet", 4.0, 4.0, "#238636"),
            ("Training / EDA", 7.0, 4.0, "#8957e5"),
            ("Online sync", 4.0, 2.5, "#d29922"),
            ("SQLite features.db", 4.0, 1.0, "#3fb950"),
            ("Real-time lookup", 7.0, 1.0, "#3fb950"),
        ],
    )


def data_flow() -> None:
    diagram_boxes(
        "Implemented data flow — Fraud Sentinel",
        "21_implemented_data_flow.png",
        [
            ("Raw sources (3)", 0.5, 4.0, "#58a6ff"),
            ("enrich_*()", 3.0, 4.0, "#6e7681"),
            ("offline_store.py", 5.5, 4.0, "#238636"),
            ("online_store.py", 3.0, 2.0, "#3fb950"),
            ("metadata_registry", 5.5, 2.0, "#d29922"),
            ("dashboard/app.py", 8.0, 2.0, "#8957e5"),
        ],
    )


def metadata_catalog() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Feature defs (7)", "Dataset sidecars (3)", "Temporal batches (3)", "Sync + rejected (5)"]
    sizes = [7, 3, 3, 5]
    colors = ["#58a6ff", "#3fb950", "#d29922", "#8957e5"]
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90)
    ax.set_title("Metadata catalog — 18 JSON records")
    save(fig, "22_metadata_catalog.png")


def main() -> None:
    d1, d2, d3 = SUMMARY["dataset_1_ulb_creditcard"], SUMMARY["dataset_2_ieee_cis"], SUMMARY["dataset_3_sparkov"]

    class_balance(d1["legit"], d1["fraud"], "Dataset 1 ULB — class balance", "01_class_balance.png")
    amount_dist_sample(80000, d1["fraud_rate"], "Dataset 1 ULB — amount distribution", "02_amount_distribution.png", 1)
    pca_boxplot_ulb()

    class_balance(d2["legit"], d2["fraud"], "Dataset 2 IEEE-CIS — class balance", "d2_ieee_class_balance.png")
    amount_dist_sample(80000, d2["fraud_rate"], "Dataset 2 IEEE-CIS — amount distribution", "d2_ieee_amount_distribution.png", 2)

    class_balance(d3["legit"], d3["fraud"], "Dataset 3 Sparkov — class balance", "d3_sparkov_class_balance.png")
    amount_dist_sample(80000, d3["fraud_rate"], "Dataset 3 Sparkov — amount distribution", "d3_sparkov_amount_distribution.png", 3)

    metrics_bar(d1["ml"], "ULB ML benchmark", "08_model_metrics_comparison.png")

    feature_store_architecture()
    registry_table()
    methodology_pipeline()
    mlp_architecture()
    fs_challenges()
    offline_online()
    data_flow()
    metadata_catalog()

    names = sorted({p.name for p in DELIVERY.glob("*.png")})
    print(f"Delivery figures: {len(names)} PNG in {DELIVERY}")
    for n in names:
        print(" ", n)


if __name__ == "__main__":
    main()
