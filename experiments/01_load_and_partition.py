#!/usr/bin/env python3
"""
Experiment 01: Load and Partition
=================================
Load the OCEL 2.0 Logistics log, print summary stats, apply the 3-org
partition, and save per-org event tables as CSVs.
"""

import sys
from pathlib import Path

# Resolve project root for imports and file paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fedsfd.ocel.loader import (
    load_ocel,
    extract_events_df,
    extract_relations_df,
    extract_objects_df,
    print_summary,
)
from fedsfd.ocel.partitioning import (
    partition_events,
    partition_relations,
    print_partition_summary,
)
from fedsfd.utils.config import load_config


def main(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "logistics_3org.yaml")

    config = load_config(config_path)
    data_path = PROJECT_ROOT / config.data_path

    # --- Load ---
    print(f"Loading OCEL log from: {data_path}")
    ocel = load_ocel(data_path)
    print_summary(ocel)

    # --- Extract DataFrames ---
    events_df = extract_events_df(ocel)
    relations_df = extract_relations_df(ocel)
    objects_df = extract_objects_df(ocel)

    # --- Partition ---
    print("\nPartitioning events by organization...")
    org_events = partition_events(events_df, config)
    org_relations = partition_relations(relations_df, org_events)
    print_partition_summary(org_events, org_relations)

    # --- Save CSVs ---
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    events_df.to_csv(results_dir / "all_events.csv", index=False)
    print(f"\nSaved: results/all_events.csv ({len(events_df)} rows)")

    for org_name, ev_df in org_events.items():
        fname = f"events_{org_name.lower().replace('/', '_').replace(' ', '_')}.csv"
        ev_df.to_csv(results_dir / fname, index=False)
        print(f"Saved: results/{fname} ({len(ev_df)} rows)")

    print("\n✓ Load and Partition complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)