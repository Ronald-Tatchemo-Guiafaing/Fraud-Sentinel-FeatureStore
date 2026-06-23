"""Central registry — same feature definitions for training and serving."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class FeatureDefinition:
    name: str
    source_columns: tuple[str, ...]
    layer: str  # raw | transformation | aggregate | prediction
    description: str
    transform: Callable | None = None


REGISTRY: list[FeatureDefinition] = [
    FeatureDefinition(
        "amount_log",
        ("Amount",),
        "transformation",
        "Log-transform of transaction amount for skew reduction.",
    ),
    FeatureDefinition(
        "hour_of_day",
        ("Time",),
        "transformation",
        "Hour of day derived from Time (seconds since first transaction).",
    ),
    FeatureDefinition(
        "amount_zscore",
        ("Amount",),
        "transformation",
        "Z-score of Amount within the offline training batch.",
    ),
    FeatureDefinition(
        "v1_v28_pca",
        tuple(f"V{i}" for i in range(1, 29)),
        "raw",
        "PCA-anonymized card transaction components (ULB dataset).",
    ),
    FeatureDefinition(
        "tx_count_1h",
        ("Time",),
        "aggregate",
        "Rolling count of transactions in the last hour (point-in-time).",
    ),
    FeatureDefinition(
        "fraud_score_rf",
        (),
        "prediction",
        "Random Forest fraud probability served from online store.",
    ),
    FeatureDefinition(
        "fraud_score_mlp",
        (),
        "prediction",
        "MLP neural network fraud probability served from online store.",
    ),
]


def registry_table_rows() -> list[list[str]]:
    rows = [["Feature", "Source", "Layer", "Description"]]
    for f in REGISTRY:
        src = ", ".join(f.source_columns) if f.source_columns else "model output"
        rows.append([f.name, src, f.layer, f.description])
    return rows
