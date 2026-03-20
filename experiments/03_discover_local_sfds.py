#!/usr/bin/env python3
"""
Experiment 03: Discover Local SFDs
===================================
For each organization, discover an SFD from aggregated time series.
Run local simulation, export to Vensim .mdl, and save parameters.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fedsfd.ocel.loader import load_ocel, extract_events_df, extract_relations_df, get_time_horizon
from fedsfd.ocel.partitioning import partition_events
from fedsfd.ocel.scoping import assign_scopes
from fedsfd.sfd.aggregation import (
    parse_time_delta,
    aggregate_all_scopes,
    compute_all_sfd_variables,
    generate_time_windows,
    print_variable_summary,
    variables_to_long_df,
)
from fedsfd.sfd.discovery import (
    discover_sfd,
    discover_sfd_from_variables,
)
from fedsfd.sfd.simulation import simulate_sfd
from fedsfd.sfd.vensim_export import export_to_mdl, export_sfd_parameters
from fedsfd.utils.config import load_config


def main(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "logistics_3org.yaml")

    config = load_config(config_path)
    data_path = PROJECT_ROOT / config.data_path
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    np.random.seed(config.random_seed)

    # --- Load data ---
    print("Loading OCEL log...")
    ocel = load_ocel(data_path)
    events_df = extract_events_df(ocel)
    relations_df = extract_relations_df(ocel)
    t_min, t_max = get_time_horizon(events_df)
    delta = parse_time_delta(config.time_window_delta)

    print(f"Time range: {t_min} → {t_max}  (Δt = {config.time_window_delta})")

    use_analyst_variables = config.has_sfd_variables
    print(f"Variable mode: {'analyst-defined' if use_analyst_variables else 'legacy scope-based'}")

    if use_analyst_variables:
        windows = generate_time_windows(t_min, t_max, delta)
        n_windows = len(windows)
        print(f"Time windows: {n_windows}")

        all_var_ts = compute_all_sfd_variables(
            events_df=events_df,
            relations_df=relations_df,
            sfd_config=config.sfd_variables,
            windows=windows,
            delta=delta,
        )
        print_variable_summary(all_var_ts, config.sfd_variables)

        var_df = variables_to_long_df(all_var_ts, config.sfd_variables, windows)
        var_df.to_csv(results_dir / "sfd_variables.csv", index=False)
        print(f"Saved variable time series to results/sfd_variables.csv")

        # --- Discover per-org local SFDs ---
        local_sfds = {}
        for org_name in config.organizations:
            print(f"\n{'─' * 60}")
            print(f"Discovering SFD for: {org_name}")
            print(f"{'─' * 60}")

            org_ts = all_var_ts.get(org_name, {})
            variable_roles = config.get_variable_roles(org_name)
            org_sfd_config = config.sfd_variables.get(org_name, {})

            sfd = discover_sfd_from_variables(
                var_ts=org_ts,
                variable_roles=variable_roles,
                org=org_name,
                correlation_threshold=config.discovery.correlation_threshold,
                max_lag=config.discovery.max_lag,
                sfd_variables_config=org_sfd_config,
            )
            sfd.print_summary()
            local_sfds[org_name] = sfd

            # Export to .mdl
            mdl_path = results_dir / f"sfd_{org_name.lower().replace(' ', '_')}.mdl"
            export_to_mdl(sfd, mdl_path, time_horizon=n_windows)
            print(f"  Exported: {mdl_path.name}")

            params_path = results_dir / f"sfd_{org_name.lower().replace(' ', '_')}_params.txt"
            export_sfd_parameters(sfd, params_path)
            print(f"  Parameters: {params_path.name}")

            # Local simulation
            print(f"  Simulating {n_windows} steps...")
            trajectories = simulate_sfd(sfd, n_steps=n_windows, dt=1.0)

            # Save trajectories
            max_len = max(len(v) for v in trajectories.values()) if trajectories else 0
            traj_df = pd.DataFrame({"time_step": range(max_len)})
            for var_name, traj in sorted(trajectories.items()):
                padded = np.full(max_len, np.nan)
                padded[:len(traj)] = traj
                traj_df[var_name] = padded
            traj_path = results_dir / f"trajectories_{org_name.lower().replace(' ', '_')}.csv"
            traj_df.to_csv(traj_path, index=False)
            print(f"  Trajectories: {traj_path.name}")

            # Stock comparison
            print(f"\n  Stock comparison (simulated vs actual):")
            for stock in sfd.stocks:
                actual_ts = org_ts.get(stock.name)
                simulated = trajectories.get(stock.name, np.array([]))
                if actual_ts is not None and len(actual_ts) > 0 and len(simulated) > 1:
                    n_compare = min(len(actual_ts), len(simulated) - 1)
                    sim_vals = simulated[1:n_compare + 1]
                    act_vals = actual_ts[:n_compare]
                    if len(sim_vals) > 0:
                        rmse = np.sqrt(np.mean((sim_vals - act_vals) ** 2))
                        mae = np.mean(np.abs(sim_vals - act_vals))
                        print(
                            f"    {stock.name:35s} "
                            f"RMSE={rmse:8.2f}  MAE={mae:8.2f}  "
                            f"actual_mean={act_vals.mean():7.1f}  "
                            f"sim_mean={sim_vals.mean():7.1f}"
                        )


    print("\n✓ Experiment 03 complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)