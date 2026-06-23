#!/usr/bin/env python3
"""
Print metadata answers for professor / oral defense.
Run: python show_metadata.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META = ROOT / "data_store" / "metadata"
CATALOG = META / "metadata_catalog.json"


def main():
    print("=" * 60)
    print("FRAUD SENTINEL — METADATA SUMMARY (for professor)")
    print("=" * 60)

    if not CATALOG.exists():
        print("\n[!] Run first: python run_feature_store_pipeline.py\n")
        return

    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    counts = cat["counts_by_type"]

    print("\n1. DO WE HAVE METADATA?")
    print("   YES — JSON files under data_store/metadata/")
    for k, v in cat["metadata_files"].items():
        print(f"   - {k}: {v}")

    print("\n2. WHAT FORM IS METADATA IN?")
    for line in cat["how_metadata_is_handled"]:
        print(f"   • {line}")

    print("\n3. HOW MANY METADATA RECORDS?")
    print(f"   Total (active + rejected): {cat['total_metadata_records']}")
    print(f"   Active only:               {counts['total_active']}")
    print(f"   - Features:                {counts['feature']}")
    print(f"   - Dataset institutions:    {counts.get('dataset_institution', counts.get('dataset_version_active', 3))}")
    print(f"   - Temporal batches:        {counts['temporal_batch']}")
    print(f"   - Online sync:             {counts['online_sync']}")
    print(f"   - Rejected (failures):     {counts['dataset_version_rejected']}")

    print("\n4. WHAT INFORMATION IS IN METADATA?")
    print("   Per dataset version: id, name, domain, stage, rows, columns, fraud_rate,")
    print("   schema_hash, parent_version, transformations, quality_checks, known_issues")
    print("   Per feature: name, layer, source_columns, dtype, served_online, point_in_time")
    print("   Per rejected version: failure_reason, lessons_from_failure")

    print("\n5. FAILED VERSIONS WE STUDIED (and fixed):")
    for r in cat["rejected_records"]:
        print(f"\n   [{r['status']}] {r['name']}")
        print(f"   Reason: {r['failure_reason']}")
        print(f"   Lesson: {r['lessons_from_failure']}")

    print("\n6. ACTIVE VERSION LINEAGE:")
    for r in cat["active_records"]:
        if r.get("type") == "dataset_version":
            print(f"   {r['name']} | stage={r['stage']} | rows={r['rows']} | parent={r.get('parent_version')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
