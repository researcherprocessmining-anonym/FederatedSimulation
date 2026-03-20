#!/usr/bin/env python3
"""
Experiment 04: Discover Federation
====================================
Run flow matching, boundary equation discovery, residual computation.
Build the federated SFD and export to .mdl.
"""

import sys
from pathlib import Path

import numpy as np

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
)
from fedsfd.sfd.discovery import discover_sfd, discover_sfd_from_variables
from fedsfd.sfd.vensim_export import export_to_mdl, export_sfd_parameters
from fedsfd.federation.flow_matching import discover_flow_matches, print_flow_matches
from fedsfd.federation.boundary import discover_boundary_equations, print_boundary_equations
from fedsfd.federation.residual import compute_residuals, print_residuals
from fedsfd.federation.federated_model import build_federated_sfd
from fedsfd.mpc.local_mock import LocalMockBackend
from fedsfd.mpc.factory import create_backend_from_experiment_config
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

    use_analyst_variables = config.has_sfd_variables

    if use_analyst_variables:
        windows = generate_time_windows(t_min, t_max, delta)
        n_windows = len(windows)

        print("Computing analyst-defined variables...")
        all_var_ts = compute_all_sfd_variables(
            events_df, relations_df, config.sfd_variables, windows, delta,
        )

        print("Discovering local SFDs...")
        local_sfds = {}
        for org_name in config.organizations:
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
            local_sfds[org_name] = sfd
            print(f"  {org_name}: {len(sfd.stocks)} stocks, {len(sfd.flows)} flows, "
                  f"{len(sfd.dependencies)} deps")

        var_ts = all_var_ts

    else:
        org_events = partition_events(events_df, config)
        scoped_events = assign_scopes(org_events, config)
        agg_df = aggregate_all_scopes(scoped_events, relations_df, config.scopes, t_min, t_max, delta)
        n_windows = agg_df["time_window_start"].nunique()

        print("Discovering local SFDs (legacy)...")
        local_sfds = {}
        for org_name in config.organizations:
            sfd = discover_sfd(agg_df, org_name, config.scopes[org_name], 0.3, 3)
            local_sfds[org_name] = sfd
            print(f"  {org_name}: {len(sfd.stocks)} stocks, {len(sfd.flows)} flows")

        var_ts = _build_var_ts_from_agg(agg_df, config)

    # --- Setup MPC backend ---
    mpc_backend = create_backend_from_experiment_config(config)
    mpc_backend.setup(len(config.organizations), {})
    org_party_map = {name: i for i, name in enumerate(config.organizations.keys())}

    # --- Flow matching ---
    print("\nDiscovering cross-org flow matches...")
    matches = discover_flow_matches(local_sfds, config, mpc_backend)
    print_flow_matches(matches)

    # --- Boundary equations ---
    persist = config.mpc.persist_shares
    if persist:
        print("\nDiscovering boundary equations (with persistent secret shares)...")
    else:
        print("\nDiscovering boundary equations...")
    boundary_equations = discover_boundary_equations(
        matches, local_sfds, var_ts, org_party_map, mpc_backend,
        persist=persist,
    )
    print_boundary_equations(boundary_equations)

    # --- Residuals ---
    print("\nComputing residuals...")
    residuals = compute_residuals(boundary_equations, local_sfds, var_ts, model_type="constant")
    print_residuals(residuals)

    # --- Build federated model ---
    print("\nBuilding federated SFD...")
    fed_sfd = build_federated_sfd(local_sfds, boundary_equations, residuals)
    fed_sfd.print_summary()

    # --- Export ---
    mdl_path = results_dir / "sfd_federated.mdl"
    export_to_mdl(fed_sfd, mdl_path, time_horizon=n_windows)
    print(f"\nExported: {mdl_path.name}")

    params_path = results_dir / "sfd_federated_params.txt"
    export_sfd_parameters(fed_sfd, params_path)
    print(f"Parameters: {params_path.name}")

    # Note: if persist_shares is True, the boundary equation secret shares
    # are stored in MP-SPDZ's Persistence/ directory. They will be read by
    # experiment 05 (simulation). To clean up after the full session, call
    # mpc_backend.clear_persisted_params().

    print("\n✓ Experiment 04 complete.")


def _build_var_ts_from_agg(agg_df, config):
    """Build var_ts dict from legacy agg_df for backward compatibility."""
    var_ts = {}
    for org_name in config.organizations:
        org_ts = {}
        org_df = agg_df[agg_df["org"] == org_name]
        for scope in org_df["scope"].unique():
            scope_df = org_df[org_df["scope"] == scope]
            for metric in ["wip", "throughput", "arrival"]:
                name = f"{scope}_{metric}".lower().replace(" ", "_")
                series = (
                    scope_df[scope_df["metric"] == metric]
                    .sort_values("time_window_start")["value"]
                    .values.astype(float)
                )
                if len(series) > 0:
                    org_ts[name] = series
        var_ts[org_name] = org_ts
    return var_ts


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)