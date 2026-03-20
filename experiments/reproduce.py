#!/usr/bin/env python3
"""
Reproduce paper experiments
============================
Three experiments for the federated SFD paper:

**Experiment 1 — What-If Simulation**
  Run federated simulation under baseline and what-if (+30% demand surge)
  scenarios.  When CPN ground-truth data is available, produce a 4-way
  comparison (real baseline, sim baseline, real what-if, sim what-if).

**Experiment 2 — CPN vs SFD vs local**
  Compare federated vs CPN
  vs local-only (RMSE, MAE, R²).

**Experiment 3 — Scalability / MPC Overhead**
  Measure wall-clock time of the federated simulation, broken down into
  MPC boundary-flow computation vs local internal-flow computation.

Outputs (all saved to results/):
  What-if:
    fed_trajectories_*.csv              trajectory CSVs
    fed_*_comparison.{pdf,png}          comparison plots
    fed_four_way_comparison.csv         4-way stock comparison

  CPN vs SFD vs local:
    eval_metrics_federated.csv          per-stock accuracy (federated)
    eval_metrics_local.csv              per-stock accuracy (local-only)
    eval_metrics_one_step_ahead.csv     per-stock accuracy (one-step-ahead)
    eval_trajectory_comparison.csv      trajectory comparison
    eval_equation_quality.csv           per-flow R² report
    results/plots/                      trajectory plots

  Scalability:
    eval_scalability.csv                MPC overhead measurements

Usage:
    python experiments/reproduce.py [config.yaml]
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
)
from fedsfd.sfd.discovery import discover_sfd, discover_sfd_from_variables
from fedsfd.sfd.simulation import simulate_sfd
from fedsfd.federation.flow_matching import discover_flow_matches
from fedsfd.federation.boundary import discover_boundary_equations
from fedsfd.federation.residual import compute_residuals
from fedsfd.federation.federated_sim_actors import federated_simulate_actors
from fedsfd.mpc.factory import create_backend_from_experiment_config
from fedsfd.utils.config import load_config

from fedsfd.evaluation.metrics import (
    compute_all_stock_metrics,
    collect_equation_quality,
    print_ts_metrics_table,
    print_equation_quality_table,
)
from fedsfd.evaluation.export import (
    save_ts_metrics,
    save_trajectory_comparison,
    save_equation_quality,
    save_scalability_results,
)
from fedsfd.evaluation.plotting import plot_trajectory_comparisons


# =========================================================================
# Shared helpers
# =========================================================================

def _save_trajectories(trajectories: dict, path: Path, n_steps: int) -> None:
    """Save trajectories to CSV with proper handling of different lengths."""
    max_len = max(len(v) for v in trajectories.values()) if trajectories else 0
    df = pd.DataFrame({"time_step": range(max_len)})
    for name, traj in sorted(trajectories.items()):
        padded = np.full(max_len, np.nan)
        padded[:len(traj)] = traj
        df[name] = padded
    df.to_csv(path, index=False)


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


def _build_what_if_mods(config) -> dict:
    """Build what-if modification dict from config."""
    what_if_mods = {}
    for wi in config.simulation.what_if:
        if "variable" in wi:
            flow_name = wi["variable"]
        elif "scope" in wi:
            scope_key = wi["scope"].lower().replace(" ", "_")
            metric = wi["metric"]
            flow_name = f"{scope_key}_{metric}"
        else:
            print(f"  Warning: unrecognized what-if entry: {wi}")
            continue
        what_if_mods[flow_name] = wi["multiplier"]
        print(f"  Modifying: {flow_name} x {wi['multiplier']}")
    return what_if_mods


def _warm_start_stocks(local_sfds: dict, var_ts: dict, use_first: bool = False) -> None:
    """Set each stock's initial_value from observed data."""
    for org_name, sfd in local_sfds.items():
        org_ts = var_ts.get(org_name, {})
        for stock in sfd.stocks:
            ts = org_ts.get(stock.name)
            if ts is None or len(ts) == 0:
                continue
            stock.initial_value = float(ts[0]) if use_first else float(np.mean(ts))
    mode = "first time-step" if use_first else "observed means"
    print(f"  Warm-started stocks with {mode}")


def _get_stock_names(local_sfds: dict) -> dict:
    """Return {org_name: [full_trajectory_key, ...]} for stocks only."""
    org_stocks = {}
    for org_name, sfd in local_sfds.items():
        prefix = org_name.lower()
        org_stocks[org_name] = [f"{prefix}_{s.name}" for s in sfd.stocks]
    return org_stocks


def _pretty_label(stock_key: str) -> str:
    """Turn 'company_order_backlog' into 'Order Backlog'."""
    parts = stock_key.split("_", 1)
    return parts[1].replace("_", " ").title() if len(parts) > 1 else stock_key


def _var_ts_to_trajectories(var_ts: dict, local_sfds: dict) -> dict:
    """Convert var_ts dict to flat trajectory dict keyed like simulation output."""
    traj = {}
    for org_name, sfd in local_sfds.items():
        prefix = org_name.lower()
        org_ts = var_ts.get(org_name, {})
        for stock in sfd.stocks:
            ts = org_ts.get(stock.name)
            if ts is not None:
                traj[f"{prefix}_{stock.name}"] = np.array(ts)
        for flow in sfd.flows:
            ts = org_ts.get(flow.name)
            if ts is not None:
                traj[f"{prefix}_{flow.name}"] = np.array(ts)
    return traj


def _truncate_trajectories(traj: dict, n: int) -> dict:
    """Truncate all arrays in a trajectory dict to length n."""
    return {k: v[:n] for k, v in traj.items()}


def _collect_stock_keys(local_sfds):
    """Get federated trajectory keys for all stocks."""
    keys = []
    for org_name, sfd in local_sfds.items():
        for stock in sfd.stocks:
            keys.append(f"{org_name.lower()}_{stock.name}")
    return keys



def _run_local_only_simulation(local_sfds, n_steps, var_ts_train=None):
    """Run local-only simulation (no cross-org flows) for each org."""
    from fedsfd.federation.federated_sim_actors import _compute_training_bounds

    training_bounds = _compute_training_bounds(var_ts_train) if var_ts_train else {}

    all_traj = {}
    for org_name, sfd in local_sfds.items():
        traj = simulate_sfd(sfd, n_steps=n_steps, dt=1.0)
        org_bounds = training_bounds.get(org_name, {})
        for var_name, values in traj.items():
            bounds = org_bounds.get(var_name)
            if bounds is not None:
                values = np.clip(values, bounds[0], bounds[1])
            all_traj[f"{org_name.lower()}_{var_name}"] = values
    return all_traj


def _timed_federated_simulate(
    local_sfds, boundary_equations, residuals,
    mpc_backend, org_party_map, n_steps, dt=1.0,
    what_if_modifications=None,
):
    """Run federated simulation and return (trajectories, elapsed_seconds)."""
    t0 = time.perf_counter()
    traj = federated_simulate_actors(
        local_sfds=local_sfds,
        boundary_equations=boundary_equations,
        residuals=residuals,
        mpc_backend=mpc_backend,
        org_party_map=org_party_map,
        n_steps=n_steps,
        dt=dt,
        what_if_modifications=what_if_modifications,
    )
    elapsed = time.perf_counter() - t0
    return traj, elapsed


def _estimate_mpc_fraction(
    local_sfds, boundary_equations, residuals,
    mpc_backend, org_party_map, n_steps,
):
    """Estimate MPC vs local time using actor-based timing instrumentation."""
    t0 = time.perf_counter()
    _traj, timing_info = federated_simulate_actors(
        local_sfds=local_sfds,
        boundary_equations=boundary_equations,
        residuals=residuals,
        mpc_backend=mpc_backend,
        org_party_map=org_party_map,
        n_steps=n_steps,
        dt=1.0,
        report_timing=True,
    )
    total_time = time.perf_counter() - t0
    mpc_time = 0.0
    local_time = 0.0
    for entry in timing_info:
        if entry["actor"] == "mpc_platform":
            mpc_time = entry["mpc_compute_sec"]
        else:
            local_time = max(local_time, entry["local_compute_sec"])

    return {
        "n_orgs": len(local_sfds),
        "n_boundary_flows": len(boundary_equations),
        "n_steps": n_steps,
        "wall_time_sec": round(total_time, 6),
        "mpc_time_sec": round(mpc_time, 6),
        "local_time_sec": round(local_time, 6),
        "mpc_fraction": round(mpc_time / total_time, 4) if total_time > 0 else 0.0,
        "total_flows": sum(len(sfd.flows) for sfd in local_sfds.values()),
        "total_stocks": sum(len(sfd.stocks) for sfd in local_sfds.values()),
    }


# =========================================================================
# Plotting helpers (what-if experiment)
# =========================================================================

def _plot_four_way_comparison(
    real_baseline: dict, sim_baseline: dict,
    real_whatif: dict, sim_whatif: dict,
    local_sfds: dict, results_dir: Path,
) -> None:
    """Combined 4-way figure: 2 columns (Baseline | What-if), one row per org."""
    org_stocks = _get_stock_names(local_sfds)
    org_names = list(org_stocks.keys())
    n_orgs = len(org_names)

    all_stock_keys = [s for org in org_names for s in org_stocks[org]]
    cmap = matplotlib.colormaps["tab10"]
    stock_colors = {s: cmap(i) for i, s in enumerate(all_stock_keys)}

    fig, axes = plt.subplots(
        n_orgs, 2, figsize=(13, 2.6 * n_orgs), sharex=True,
        constrained_layout=True,
    )
    if n_orgs == 1:
        axes = axes.reshape(1, 2)

    col_data = [
        (real_baseline, sim_baseline, "Baseline"),
        (real_whatif, sim_whatif, "What-if (+30 % demand)"),
    ]

    for col_idx, (real_t, sim_t, title_suffix) in enumerate(col_data):
        for row_idx, org_name in enumerate(org_names):
            ax = axes[row_idx, col_idx]
            for skey in org_stocks[org_name]:
                r = real_t.get(skey, np.array([]))
                s = sim_t.get(skey, np.array([]))
                if len(r) == 0 and len(s) == 0:
                    continue
                label = _pretty_label(skey)
                color = stock_colors[skey]
                if len(r) > 0:
                    ax.plot(r, color=color, linewidth=1.3, label=label)
                if len(s) > 0:
                    ax.plot(s, color=color, linewidth=1.3, linestyle="--", alpha=0.85)
                if col_idx == 1 and len(s) > 0:
                    bl_s = sim_baseline.get(skey, np.array([]))
                    if len(bl_s) > 0:
                        n = min(len(s), len(bl_s))
                        mean_bl = np.mean(bl_s[:n])
                        mean_wi = np.mean(s[:n])
                        if mean_bl != 0:
                            pct = 100.0 * (mean_wi - mean_bl) / abs(mean_bl)
                            ax.annotate(
                                f"avg {pct:+.1f}%",
                                xy=(len(s) - 1, s[-1]),
                                xytext=(4, 0), textcoords="offset points",
                                fontsize=6, color=color, fontweight="bold",
                                va="center",
                            )
            if col_idx == 1:
                ax.set_ylabel("Stock level")
            ax.grid(True, linewidth=0.3, alpha=0.6)
            if row_idx == 0:
                ax.set_title(title_suffix, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.annotate(
                    org_name, xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-45, 0), textcoords="offset points",
                    fontsize=10, fontweight="bold", va="center", rotation=90,
                )
            ax.legend(fontsize=6, ncol=2, loc="best", framealpha=0.7)

    for row_idx in range(n_orgs):
        ymin = min(axes[row_idx, c].get_ylim()[0] for c in range(2))
        ymax = max(axes[row_idx, c].get_ylim()[1] for c in range(2))
        for c in range(2):
            axes[row_idx, c].set_ylim(ymin, ymax)

    for col_idx in range(2):
        axes[-1, col_idx].set_xlabel("Simulation step (days)")

    fig.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1.3, linestyle="-", label="Real (CPN)"),
            Line2D([0], [0], color="black", linewidth=1.3, linestyle="--", alpha=0.85, label="SFD simulated"),
        ],
        loc="upper center", ncol=2, fontsize=9, frameon=True, framealpha=0.7,
        bbox_to_anchor=(0.5, -0.02),
    )
    for fmt in ("pdf", "png"):
        fig.savefig(results_dir / f"fed_four_way_comparison.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: fed_four_way_comparison.pdf / .png")


# =========================================================================
# Shared data loading and SFD discovery
# =========================================================================

def _load_data_and_compute_variables(config, data_path, delta):
    """Load OCEL, compute time-series variables."""
    print("Loading OCEL log...")
    ocel = load_ocel(data_path)
    events_df = extract_events_df(ocel)
    relations_df = extract_relations_df(ocel)
    t_min, t_max = get_time_horizon(events_df)

    if config.has_sfd_variables:
        windows = generate_time_windows(t_min, t_max, delta)
        var_ts = compute_all_sfd_variables(
            events_df, relations_df, config.sfd_variables, windows, delta,
        )
    else:
        org_events = partition_events(events_df, config)
        scoped_events = assign_scopes(org_events, config)
        agg_df = aggregate_all_scopes(scoped_events, relations_df, config.scopes, t_min, t_max, delta)
        var_ts = _build_var_ts_from_agg(agg_df, config)

    return var_ts


def _discover_local_sfds(config, var_ts):
    """Discover local SFDs for each organization."""
    local_sfds = {}
    use_analyst_variables = config.has_sfd_variables

    if use_analyst_variables:
        for org_name in config.organizations:
            org_ts = var_ts.get(org_name, {})
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
    else:
        for org_name in config.organizations:
            local_sfds[org_name] = discover_sfd(
                var_ts.get(org_name, {}), org_name,
                config.scopes[org_name], 0.3, 3,
            )

    for org_name, sfd in local_sfds.items():
        print(f"  {org_name}: {len(sfd.stocks)} stocks, {len(sfd.flows)} flows, "
              f"{len(sfd.dependencies)} deps")
    return local_sfds



# =========================================================================
# Experiment 1: What-If Simulation
# =========================================================================

def run_whatif_experiment(config, var_ts, local_sfds, boundary_equations,
                         residuals, mpc_backend, org_party_map,
                         results_dir, delta):
    """Run what-if simulation experiment."""
    print(f"\n{'#' * 70}")
    print("  EXPERIMENT 1: What-If Simulation")
    print(f"{'#' * 70}")

    warmup_days = config.simulation.warmup_days

    def _load_cpn_ocel(path_str, label):
        """Load a CPN OCEL, skip warmup windows, and compute SFD variables."""
        cpn_path = PROJECT_ROOT / path_str
        print(f"\nLoading CPN {label} OCEL from {cpn_path}...")
        cpn_ocel = load_ocel(cpn_path)
        ev_df = extract_events_df(cpn_ocel)
        rel_df = extract_relations_df(cpn_ocel)
        t0_raw, t1 = get_time_horizon(ev_df)

        if warmup_days > 0:
            t0 = t0_raw + pd.Timedelta(days=warmup_days)
            print(f"  Skipping {warmup_days}-day warm-up: "
                  f"windows start at {t0.date()} (data starts {t0_raw.date()})")
        else:
            t0 = t0_raw

        wins = generate_time_windows(t0, t1, delta)
        print(f"Computing SFD variables from CPN {label} data...")
        vts = compute_all_sfd_variables(
            ev_df, rel_df, config.sfd_variables, wins, delta,
        )
        print(f"  CPN {label} spans {len(wins)} time windows "
              f"({t0.date()} to {t1.date()})")
        return vts, len(wins)

    cpn_baseline_var_ts = None
    cpn_whatif_var_ts = None

    if config.simulation.baseline_data_path:
        cpn_baseline_var_ts, _ = _load_cpn_ocel(
            config.simulation.baseline_data_path, "baseline")
    if config.simulation.whatif_data_path:
        cpn_whatif_var_ts, _ = _load_cpn_ocel(
            config.simulation.whatif_data_path, "what-if")

    has_both_cpn = cpn_baseline_var_ts is not None and cpn_whatif_var_ts is not None

    if has_both_cpn:
        # 4-way comparison: CPN baseline, sim baseline, CPN what-if, sim what-if
        real_baseline = _var_ts_to_trajectories(cpn_baseline_var_ts, local_sfds)
        real_whatif = _var_ts_to_trajectories(cpn_whatif_var_ts, local_sfds)

        n_windows = 300
        real_baseline = _truncate_trajectories(real_baseline, n_windows)
        real_whatif = _truncate_trajectories(real_whatif, n_windows)

        _save_trajectories(real_baseline, results_dir / "fed_trajectories_real_baseline.csv", n_windows)
        _save_trajectories(real_whatif, results_dir / "fed_trajectories_real_whatif.csv", n_windows)

        _warm_start_stocks(local_sfds, cpn_baseline_var_ts, use_first=True)

        print(f"\nRunning BASELINE federated simulation ({n_windows} steps)...")
        sim_baseline, _ = _timed_federated_simulate(
            local_sfds, boundary_equations, residuals,
            mpc_backend, org_party_map, n_steps=n_windows, dt=1.0,
        )
        _save_trajectories(sim_baseline, results_dir / "fed_trajectories_sim_baseline.csv", n_windows)

        what_if_mods = _build_what_if_mods(config) if config.simulation.what_if else {}
        print(f"\nRunning WHAT-IF federated simulation ({n_windows} steps)...")
        sim_whatif, _ = _timed_federated_simulate(
            local_sfds, boundary_equations, residuals,
            mpc_backend, org_party_map, n_steps=n_windows, dt=1.0,
            what_if_modifications=what_if_mods,
        )
        _save_trajectories(sim_whatif, results_dir / "fed_trajectories_sim_whatif.csv", n_windows)

        # Print 4-way stock summary
        print(f"\n{'=' * 90}")
        print("4-Way Stock Comparison (mean values)")
        print(f"{'=' * 90}")
        print(f"{'Stock':<35s} {'Real BL':>10s} {'Sim BL':>10s} {'Real WI':>10s} {'Sim WI':>10s}")
        print("-" * 90)
        for name in sorted(real_baseline.keys()):
            is_stock = any(
                name == f"{org.lower()}_{s.name}"
                for org, sfd in local_sfds.items() for s in sfd.stocks
            )
            if not is_stock:
                continue
            rb = np.mean(real_baseline.get(name, [0]))
            sb = np.mean(sim_baseline.get(name, [0]))
            rw = np.mean(real_whatif.get(name, [0]))
            sw = np.mean(sim_whatif.get(name, [0]))
            print(f"  {name:<33s} {rb:>10.1f} {sb:>10.1f} {rw:>10.1f} {sw:>10.1f}")

        # Save 4-way comparison CSV
        comp_rows = []
        for name in sorted(real_baseline.keys()):
            is_stock = any(
                name == f"{org.lower()}_{s.name}"
                for org, sfd in local_sfds.items() for s in sfd.stocks
            )
            if not is_stock:
                continue
            rb = real_baseline.get(name, np.array([]))
            sb = sim_baseline.get(name, np.array([]))
            rw = real_whatif.get(name, np.array([]))
            sw = sim_whatif.get(name, np.array([]))
            n = min(len(rb), len(sb), len(rw), len(sw))
            for i in range(n):
                comp_rows.append({
                    "time_step": i, "stock": name,
                    "real_baseline": rb[i], "sim_baseline": sb[i],
                    "real_whatif": rw[i], "sim_whatif": sw[i],
                })
        if comp_rows:
            pd.DataFrame(comp_rows).to_csv(
                results_dir / "fed_four_way_comparison.csv", index=False)
            print(f"\nSaved: fed_four_way_comparison.csv")

        # Plot
        _plot_four_way_comparison(
            real_baseline, sim_baseline, real_whatif, sim_whatif,
            local_sfds, results_dir,
        )

    else:
        # Legacy 2-way: simulated baseline vs simulated what-if
        _warm_start_stocks(local_sfds, var_ts)
        n_windows = 30

        print(f"\nRunning BASELINE federated simulation ({n_windows} steps)...")
        sim_baseline, _ = _timed_federated_simulate(
            local_sfds, boundary_equations, residuals,
            mpc_backend, org_party_map, n_steps=n_windows, dt=1.0,
        )
        _save_trajectories(sim_baseline, results_dir / "fed_trajectories_baseline.csv", n_windows)

        if config.simulation.what_if:
            what_if_mods = _build_what_if_mods(config)
            print(f"\nRunning WHAT-IF federated simulation ({n_windows} steps)...")
            sim_whatif, _ = _timed_federated_simulate(
                local_sfds, boundary_equations, residuals,
                mpc_backend, org_party_map, n_steps=n_windows, dt=1.0,
                what_if_modifications=what_if_mods,
            )
            _save_trajectories(sim_whatif, results_dir / "fed_trajectories_whatif.csv", n_windows)

    print("  Experiment 1 (What-If) complete.")


# =========================================================================
# Experiment 2: CPN vs SFD vs local
# =========================================================================

def run_predictive_fitness(config, local_sfds, boundary_equations, residuals,
                           mpc_backend, org_party_map, var_ts,
                           results_dir, delta):
    """Run CPN vs SFD vs local evaluation.

    SFDs are discovered from the full OCEL dataset.  Accuracy is evaluated
    against a separately generated CPN baseline simulation (configured via
    ``simulation.baseline_data_path``).
    """
    print(f"\n{'#' * 70}")
    print("  EXPERIMENT 2: CPN vs SFD vs local")
    print(f"{'#' * 70}")

    # --- Load CPN baseline ground truth ---
    if not config.simulation.baseline_data_path:
        print("  SKIPPED: no baseline_data_path configured — "
              "cannot evaluate CPN vs SFD vs local without CPN ground truth.")
        return None

    warmup_days = config.simulation.warmup_days
    cpn_path = PROJECT_ROOT / config.simulation.baseline_data_path
    print(f"  Loading CPN baseline OCEL from {cpn_path}...")
    cpn_ocel = load_ocel(cpn_path)
    ev_df = extract_events_df(cpn_ocel)
    rel_df = extract_relations_df(cpn_ocel)
    t0_raw, t1 = get_time_horizon(ev_df)

    if warmup_days > 0:
        t0 = t0_raw + pd.Timedelta(days=warmup_days)
        print(f"  Skipping {warmup_days}-day warm-up: "
              f"windows start at {t0.date()} (data starts {t0_raw.date()})")
    else:
        t0 = t0_raw

    wins = generate_time_windows(t0, t1, delta)
    cpn_var_ts = compute_all_sfd_variables(
        ev_df, rel_df, config.sfd_variables, wins, delta,
    )
    n_test = len(wins)
    print(f"  CPN baseline: {n_test} time windows ({t0.date()} to {t1.date()})")

    # --- Warm-start stocks from CPN baseline first time-step ---
    _warm_start_stocks(local_sfds, cpn_var_ts, use_first=True)

    # --- Simulate ---
    print(f"\n  Running federated simulation ({n_test} steps)...")
    federated_traj, fed_time = _timed_federated_simulate(
        local_sfds, boundary_equations, residuals,
        mpc_backend, org_party_map, n_test,
    )
    print(f"    Federated sim wall time: {fed_time:.4f}s")

    print(f"  Running local-only simulation ({n_test} steps)...")
    local_only_traj = _run_local_only_simulation(local_sfds, n_test, var_ts_train=var_ts)

    # --- Metrics (compare against CPN baseline) ---
    fed_metrics = compute_all_stock_metrics(cpn_var_ts, federated_traj, local_sfds)
    print_ts_metrics_table(fed_metrics, title="Federated Simulation vs CPN Baseline")
    save_ts_metrics(fed_metrics, results_dir / "eval_metrics_federated.csv", label="federated")

    local_metrics = compute_all_stock_metrics(cpn_var_ts, local_only_traj, local_sfds)
    print_ts_metrics_table(local_metrics, title="Local-Only Simulation vs CPN Baseline")
    save_ts_metrics(local_metrics, results_dir / "eval_metrics_local.csv", label="local_only")

    # Comparative summary
    print(f"\n{'=' * 80}")
    print("  Accuracy Comparison Summary (mean across stocks)")
    print(f"{'=' * 80}")

    def _mean_metric(metrics_list, attr):
        vals = [getattr(m, attr) for m in metrics_list if not np.isnan(getattr(m, attr))]
        return np.mean(vals) if vals else np.nan

    for label, mlist in [("Federated", fed_metrics),
                         ("Local-only", local_metrics)]:
        avg_rmse = _mean_metric(mlist, "rmse")
        avg_mae = _mean_metric(mlist, "mae")
        avg_r2 = _mean_metric(mlist, "r_squared")
        print(f"  {label:<15s}  avg RMSE={avg_rmse:8.3f}  avg MAE={avg_mae:8.3f}  avg R2={avg_r2:8.3f}")

    # Federation value
    print(f"\n{'=' * 80}")
    print("  Federation Value: Federated RMSE / Local-only RMSE per stock")
    print(f"{'=' * 80}")
    local_by_var = {m.variable: m for m in local_metrics}
    print(f"  {'Stock':<42s} {'Local RMSE':>10s} {'Fed RMSE':>10s} {'Ratio':>8s} {'Verdict':>10s}")
    print("  " + "-" * 82)
    for fm in fed_metrics:
        lm = local_by_var.get(fm.variable)
        if lm is None:
            continue
        if lm.rmse > 1e-12:
            ratio = fm.rmse / lm.rmse
            verdict = "BETTER" if ratio < 0.9 else ("SAME" if ratio < 1.1 else "WORSE")
        else:
            ratio = np.nan
            verdict = "N/A"
        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
        print(f"  {fm.variable:<42s} {lm.rmse:>10.3f} {fm.rmse:>10.3f} {ratio_str:>8s} {verdict:>10s}")

    # Trajectory comparison
    stock_keys = _collect_stock_keys(local_sfds)
    save_trajectory_comparison(
        actual_ts=cpn_var_ts,
        federated_traj=federated_traj,
        stock_names=stock_keys,
        path=results_dir / "eval_trajectory_comparison.csv",
        local_only_traj=local_only_traj,
    )

    # Trajectory plots
    plots_dir = results_dir / "plots"
    plot_paths = plot_trajectory_comparisons(
        actual_ts=cpn_var_ts,
        simulated_trajectories={
            "Federated": federated_traj,
            "Local-only": local_only_traj,
        },
        local_sfds=local_sfds,
        output_dir=plots_dir,
    )
    if plot_paths:
        print(f"  Saved {len(plot_paths)} trajectory plots to results/plots/")

    # Equation quality
    eq_quality = collect_equation_quality(local_sfds)
    print_equation_quality_table(eq_quality)
    save_equation_quality(eq_quality, results_dir / "eval_equation_quality.csv")

    weak_flows = [q for q in eq_quality if not np.isnan(q.r_squared) and q.r_squared < 0.2]
    if weak_flows:
        print(f"\n  Flows with R2 < 0.2 (weak fit):")
        for q in weak_flows:
            print(f"    {q.org}/{q.flow_name}: R2={q.r_squared:.3f} ({q.equation_type})")

    # Error propagation
    print(f"\n{'=' * 80}")
    print("  Error Propagation: per-step absolute error trajectory")
    print(f"{'=' * 80}")
    for fm in fed_metrics:
        if fm.n_points < 4:
            continue
        errors = np.abs(fm.normalized_errors[~np.isnan(fm.normalized_errors)])
        if len(errors) < 4:
            continue
        mid = len(errors) // 2
        first_half = np.mean(errors[:mid])
        second_half = np.mean(errors[mid:])
        if first_half > 1e-12:
            growth = (second_half - first_half) / first_half * 100
        else:
            growth = 0.0
        trend = "STABLE" if abs(growth) < 20 else ("GROWING" if growth > 0 else "SHRINKING")
        print(f"  {fm.variable:<42s}  1st-half={first_half:.3f}  2nd-half={second_half:.3f}  "
              f"change={growth:+.0f}%  -> {trend}")

    print("  Experiment 2 (Predictive Fitness) complete.")


# =========================================================================
# Experiment 3: Scalability / MPC Overhead
# =========================================================================

def run_scalability(local_sfds, boundary_equations, residuals,
                    mpc_backend, org_party_map, results_dir,
                    sfd_discovery_time, flow_matching_time,
                    boundary_eq_time, residual_time):
    """Run scalability / MPC overhead experiment."""
    print(f"\n{'#' * 70}")
    print("  EXPERIMENT 3: Scalability / MPC Overhead")
    print(f"{'#' * 70}")

    step_counts = sorted(set([3, 5, 7, 10, 15, 30, 50, 100]))
    scalability_results = []

    for n_steps in step_counts:
        result = _estimate_mpc_fraction(
            local_sfds, boundary_equations, residuals,
            mpc_backend, org_party_map, n_steps,
        )
        scalability_results.append(result)
        print(
            f"  steps={result['n_steps']:>4d}  "
            f"total={result['wall_time_sec']:.4f}s  "
            f"mpc={result['mpc_time_sec']:.4f}s  "
            f"local={result['local_time_sec']:.4f}s  "
            f"mpc_frac={result['mpc_fraction']:.2%}"
        )

    federation_discovery_time = flow_matching_time + boundary_eq_time + residual_time
    total_discovery_time = sfd_discovery_time + federation_discovery_time
    scalability_results.append({
        "n_orgs": len(local_sfds),
        "n_boundary_flows": len(boundary_equations),
        "n_steps": 0,
        "wall_time_sec": round(total_discovery_time, 6),
        "mpc_time_sec": round(flow_matching_time + boundary_eq_time, 6),
        "local_time_sec": round(sfd_discovery_time + residual_time, 6),
        "mpc_fraction": round(
            (flow_matching_time + boundary_eq_time) / total_discovery_time, 4
        ) if total_discovery_time > 0 else 0.0,
        "total_flows": sum(len(sfd.flows) for sfd in local_sfds.values()),
        "total_stocks": sum(len(sfd.stocks) for sfd in local_sfds.values()),
        "mode": "discovery",
    })
    save_scalability_results(scalability_results, results_dir / "eval_scalability.csv")

    n_total_vars = sum(len(sfd.stocks) + len(sfd.flows) for sfd in local_sfds.values())
    n_boundary = len(boundary_equations)

    print(f"\n  Summary:")
    print(f"    Total model variables:    {n_total_vars}")
    print(f"    Boundary flows (MPC):     {n_boundary}")
    print(f"    Internal variables:        {n_total_vars - n_boundary}")
    if n_total_vars > 0:
        print(f"    MPC variables / total:     {n_boundary}/{n_total_vars} "
              f"= {n_boundary / n_total_vars:.1%}")

    print(f"\n  Discovery phase timing:")
    print(f"    Local SFD discovery:      {sfd_discovery_time:.4f}s")
    print(f"    Flow matching (MPC):      {flow_matching_time:.4f}s")
    print(f"    Boundary equations (MPC): {boundary_eq_time:.4f}s")
    print(f"    Residual fitting:         {residual_time:.4f}s")
    print(f"    Total discovery:          {total_discovery_time:.4f}s")

    sim_results = [r for r in scalability_results if r.get("mode") != "discovery"]
    if sim_results:
        last = sim_results[-1]
        mpc_per_boundary_per_step = (
            last["mpc_time_sec"] / (last["n_boundary_flows"] * last["n_steps"])
            if last["n_boundary_flows"] > 0 and last["n_steps"] > 0 else 0.0
        )
        full_mpc_estimate = mpc_per_boundary_per_step * n_total_vars * last["n_steps"]
        print(f"\n    Estimated full-MPC simulation time: {full_mpc_estimate:.4f}s "
              f"(if all {n_total_vars} variables ran in MPC)")
        print(f"    Actual federated time:              {last['wall_time_sec']:.4f}s")
        if full_mpc_estimate > 0:
            speedup = full_mpc_estimate / last["wall_time_sec"]
            print(f"    Speedup from selective MPC:          {speedup:.1f}x")

    print("  Experiment 3 (Scalability) complete.")


# =========================================================================
# Main
# =========================================================================

def main(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "logistics_3org.yaml")

    config = load_config(config_path)
    data_path = PROJECT_ROOT / config.data_path
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    delta = parse_time_delta(config.time_window_delta)

    np.random.seed(config.random_seed)

    # --- Load data and compute variables ---
    var_ts = _load_data_and_compute_variables(config, data_path, delta)

    # --- Setup MPC backend ---
    mpc_backend = create_backend_from_experiment_config(config)
    mpc_backend.setup(len(config.organizations), {})

    # =====================================================================
    # Shared: Discover SFDs and federation from full OCEL data (timed)
    # =====================================================================
    print("\nDiscovering local SFDs (full data)...")
    t_sfd_start = time.perf_counter()
    local_sfds = _discover_local_sfds(config, var_ts)
    sfd_discovery_time = time.perf_counter() - t_sfd_start

    org_party_map = {name: i for i, name in enumerate(config.organizations.keys())}
    persist = config.mpc.persist_shares

    t_fm_start = time.perf_counter()
    matches = discover_flow_matches(local_sfds, config, mpc_backend)
    flow_matching_time = time.perf_counter() - t_fm_start

    t_be_start = time.perf_counter()
    boundary_equations = discover_boundary_equations(
        matches, local_sfds, var_ts, org_party_map, mpc_backend,
        persist=persist,
    )
    boundary_eq_time = time.perf_counter() - t_be_start

    t_res_start = time.perf_counter()
    residuals = compute_residuals(boundary_equations, local_sfds, var_ts, model_type="constant")
    residual_time = time.perf_counter() - t_res_start

    # =====================================================================
    # Experiment 1: What-If Simulation
    # =====================================================================
    run_whatif_experiment(
        config, var_ts, local_sfds, boundary_equations, residuals,
        mpc_backend, org_party_map, results_dir, delta,
    )

    # =====================================================================
    # Experiment 2: Predictive Fitness (SFDs from full OCEL, test on CPN)
    # =====================================================================
    run_predictive_fitness(
        config, local_sfds, boundary_equations, residuals,
        mpc_backend, org_party_map, var_ts,
        results_dir, delta,
    )

    # =====================================================================
    # Experiment 3: Scalability
    # =====================================================================
    run_scalability(
        local_sfds=local_sfds,
        boundary_equations=boundary_equations,
        residuals=residuals,
        mpc_backend=mpc_backend,
        org_party_map=org_party_map,
        results_dir=results_dir,
        sfd_discovery_time=sfd_discovery_time,
        flow_matching_time=flow_matching_time,
        boundary_eq_time=boundary_eq_time,
        residual_time=residual_time,
    )

    # --- Cleanup ---
    if config.mpc.persist_shares:
        print("\nCleaning up persisted secret shares...")
        mpc_backend.clear_persisted_params()

    print(f"\n{'=' * 70}")
    print("  All experiments complete. Results saved to results/:")
    for f in sorted(results_dir.glob("*.csv")):
        print(f"    {f.name}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    config_path = args[0] if args else None
    main(config_path)
