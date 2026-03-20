#!/usr/bin/env python3
"""
Experiment 02: Aggregate Time Series
=====================================
Compute aggregated time series for SFD discovery.

When ``sfd_variables`` is present in the config, computes analyst-defined
variables (backlogs, rates, ratios).  Otherwise falls back to legacy
WIP/throughput/arrival per scope.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fedsfd.ocel.loader import (
    load_ocel,
    extract_events_df,
    extract_relations_df,
    get_time_horizon,
)
from fedsfd.ocel.partitioning import partition_events
from fedsfd.ocel.scoping import assign_scopes, print_scope_summary
from fedsfd.sfd.aggregation import (
    parse_time_delta,
    aggregate_all_scopes,
    print_aggregation_summary,
    compute_all_sfd_variables,
    generate_time_windows,
    print_variable_summary,
    variables_to_long_df,
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
    events_df = extract_events_df(ocel)
    relations_df = extract_relations_df(ocel)

    t_min, t_max = get_time_horizon(events_df)
    delta = parse_time_delta(config.time_window_delta)
    print(f"\nTime window Δt = {delta}")
    print(f"Time range: {t_min} → {t_max}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    if config.has_sfd_variables:
        # --- Analyst-defined variable path ---
        print("\nMode: analyst-defined SFD variables")
        windows = generate_time_windows(t_min, t_max, delta)
        print(f"Time windows: {len(windows)}")

        all_var_ts = compute_all_sfd_variables(
            events_df=events_df,
            relations_df=relations_df,
            sfd_config=config.sfd_variables,
            windows=windows,
            delta=delta,
        )
        print_variable_summary(all_var_ts, config.sfd_variables)

        # Save to CSV (long format)
        var_df = variables_to_long_df(all_var_ts, config.sfd_variables, windows)
        var_df.to_csv(results_dir / "sfd_variables.csv", index=False)
        print(f"\nSaved: results/sfd_variables.csv ({len(var_df)} rows)")

        # Also save per-org wide CSVs
        for org_name, org_ts in all_var_ts.items():
            import pandas as pd
            import numpy as np
            wide_df = pd.DataFrame({"time_window_start": windows})
            for var_name, ts in sorted(org_ts.items()):
                wide_df[var_name] = ts
            fname = f"variables_{org_name.lower().replace(' ', '_')}.csv"
            wide_df.to_csv(results_dir / fname, index=False)
            print(f"Saved: results/{fname}")

    else:
        # --- Legacy scope-based path ---
        print("\nMode: legacy scope-based aggregation")
        org_events = partition_events(events_df, config)
        scoped_events = assign_scopes(org_events, config)
        print_scope_summary(scoped_events)

        agg_df = aggregate_all_scopes(
            scoped_events=scoped_events,
            relations_df=relations_df,
            scopes_config=config.scopes,
            t_min=t_min,
            t_max=t_max,
            delta=delta,
        )
        print_aggregation_summary(agg_df)

        agg_df.to_csv(results_dir / "aggregated_timeseries.csv", index=False)
        print(f"Saved: results/aggregated_timeseries.csv ({len(agg_df)} rows)")

        for org in sorted(agg_df["org"].unique()):
            for scope in sorted(agg_df[agg_df["org"] == org]["scope"].unique()):
                scope_df = agg_df[
                    (agg_df["org"] == org) & (agg_df["scope"] == scope)
                ].copy()
                wide = scope_df.pivot(
                    index="time_window_start", columns="metric", values="value"
                ).reset_index()
                fname = (
                    f"ts_{org.lower().replace(' ', '_')}_"
                    f"{scope.lower().replace(' ', '_')}.csv"
                )
                wide.to_csv(results_dir / fname, index=False)

    print("\n✓ Aggregate Time Series complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)