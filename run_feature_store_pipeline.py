#!/usr/bin/env python3
"""Run Fraud Sentinel feature store pipeline."""
from fraud_sentinel.pipeline import run_pipeline

if __name__ == "__main__":
    summary = run_pipeline()
    print("Pipeline OK — datasets:", list(summary["datasets"].keys()))
    print("Online sync per dataset:", summary["online_sync_by_dataset"])
