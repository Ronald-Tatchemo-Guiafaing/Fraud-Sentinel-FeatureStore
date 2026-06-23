"""
Four challenges in Feature Store Engineering (professor oral question).
Distinct from Feature Engineering (creating ML features for fraud detection).
"""

CHALLENGES = [
    {
        "id": 1,
        "name": "Feature Consistency",
        "question": "Ensuring the same feature computation logic is used during both "
        "model training and production inference.",
        "risk": "Training-serving skew — features differ between environments.",
        "our_solution": "Central feature registry (feature_registry.py) maps ULB / IEEE / Sparkov "
        "columns to canonical names in offline Parquet and online SQLite.",
        "code": "fraud_sentinel/feature_registry.py",
        "dataset_example": "Dataset 2 IEEE: TransactionAmt → canonical transaction_amount",
    },
    {
        "id": 2,
        "name": "Data Quality and Reliability",
        "question": "Handling missing, incorrect, duplicated, or delayed data.",
        "risk": "Silent fraud detection failures in production.",
        "our_solution": "Per-dataset validation + quality_checks in metadata sidecars "
        "+ rejected_version_log.json for documented failures.",
        "code": "fraud_sentinel/pipeline.py + metadata_registry.py",
        "dataset_example": "Dataset 1 ULB: null/duplicate checks before Parquet write",
    },
    {
        "id": 3,
        "name": "Real-Time vs. Batch Processing",
        "question": "Supporting both batch-generated features and real-time features.",
        "risk": "Latency vs cost imbalance; train/prod feature mismatch.",
        "our_solution": "Offline store (Parquet, batch) + Online store (SQLite, real-time lookup) "
        "with controlled sync (sync_log).",
        "code": "offline_store.py + online_store.py",
        "dataset_example": "Dataset 1 ULB: enriched features batch → 5,000 rows synced online",
    },
    {
        "id": 4,
        "name": "Scalability",
        "question": "Managing large volumes of data and thousands of features across models.",
        "risk": "Slow retrieval and storage as data grows.",
        "our_solution": "Three institution corpora + temporal batches + metadata catalog "
        "+ Parquet columnar storage.",
        "code": "data_store/metadata/metadata_catalog.json",
        "dataset_example": "Dataset 3 Sparkov: ~1.3M rows ingested offline",
    },
]


def as_table_rows() -> list[dict]:
    return [
        {
            "challenge": c["name"],
            "professor_focus": c["question"],
            "fraud_sentinel_solution": c["our_solution"],
            "implementation": c["code"],
        }
        for c in CHALLENGES
    ]
