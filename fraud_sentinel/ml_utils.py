"""Shared ML training / scoring for all three datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CHUNK = 20_000


def train_and_score(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "Class",
    sample_n: int = 40_000,
) -> tuple[pd.DataFrame, dict]:
    """SMOTE-Tomek + RF/MLP; add fraud_score_rf / fraud_score_mlp columns; return metrics."""
    out = df.copy()
    out["fraud_score_rf"] = np.nan
    out["fraud_score_mlp"] = np.nan

    usable = [c for c in feature_cols if c in out.columns]
    work = out[usable + [label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise ValueError("No usable rows for ML after cleaning")

    n0 = min((work[label_col] == 0).sum(), sample_n // 2)
    n1 = min((work[label_col] == 1).sum(), max(sample_n // 2, 1))
    train_df = pd.concat([
        work[work[label_col] == 0].sample(n0, random_state=RANDOM_STATE),
        work[work[label_col] == 1].sample(n1, random_state=RANDOM_STATE),
    ])

    X = train_df[usable].values
    y = train_df[label_col].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    smote = SMOTETomek(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_s, y)

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, random_state=RANDOM_STATE)
    rf.fit(X_res, y_res)
    mlp.fit(X_res, y_res)

    for start in range(0, len(out), CHUNK):
        idx = out.index[start : start + CHUNK]
        block = out.loc[idx, usable].replace([np.inf, -np.inf], np.nan)
        valid = block.notna().all(axis=1)
        if not valid.any():
            continue
        Xb = scaler.transform(block.loc[valid].values)
        out.loc[block.loc[valid].index, "fraud_score_rf"] = rf.predict_proba(Xb)[:, 1]
        out.loc[block.loc[valid].index, "fraud_score_mlp"] = mlp.predict_proba(Xb)[:, 1]

    eval_df = out.dropna(subset=["fraud_score_rf", "fraud_score_mlp"])
    if len(eval_df) < 500:
        eval_df = work.sample(min(10_000, len(work)), random_state=RANDOM_STATE)
        Xe = scaler.transform(eval_df[usable].values)
        y_true = eval_df[label_col].values
        pr_rf = rf.predict_proba(Xe)[:, 1]
        pr_mlp = mlp.predict_proba(Xe)[:, 1]
    else:
        X_hold, _, y_hold, _ = train_test_split(
            eval_df[usable], eval_df[label_col],
            test_size=0.2, random_state=RANDOM_STATE, stratify=eval_df[label_col],
        )
        y_true = y_hold.values
        pr_rf = rf.predict_proba(scaler.transform(X_hold))[:, 1]
        pr_mlp = mlp.predict_proba(scaler.transform(X_hold))[:, 1]

    def _metrics(name: str, proba: np.ndarray) -> dict:
        pred = (proba >= 0.5).astype(int)
        return {
            "model": name,
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else 0.0,
            "pr_auc": float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else 0.0,
        }

    metrics = {
        "Random Forest": _metrics("Random Forest", pr_rf),
        "MLP": _metrics("MLP", pr_mlp),
    }
    return out, metrics
