"""
Export evaluation results to CSV files.

Saves:
  - Per-variable accuracy metrics (RMSE, MAE, MAPE, R²)
  - Trajectory comparisons: actual vs federated per stock
  - Equation quality report
  - MPC scalability measurements
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fedsfd.evaluation.metrics import (
    EquationQuality,
    TimeSeriesMetrics,
)


def save_ts_metrics(
    metrics: List[TimeSeriesMetrics],
    path: Path,
    label: str = "",
) -> None:
    """Save time-series accuracy metrics to CSV.

    Columns: variable, label, rmse, mae, mape, r_squared,
             actual_mean, simulated_mean, n_points
    """
    rows = []
    for m in metrics:
        rows.append({
            "variable": m.variable,
            "label": label,
            "rmse": m.rmse,
            "mae": m.mae,
            "mape": m.mape,
            "r_squared": m.r_squared,
            "actual_mean": m.actual_mean,
            "simulated_mean": m.simulated_mean,
            "n_points": m.n_points,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def save_trajectory_comparison(
    actual_ts: Dict[str, Dict[str, np.ndarray]],
    federated_traj: Dict[str, np.ndarray],
    stock_names: List[str],
    path: Path,
    local_only_traj: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save per-stock trajectory comparison to CSV.

    For each stock at each time step, save:
        time, stock, actual, simulated_federated, [simulated_local]

    Parameters
    ----------
    actual_ts : dict of {org_name: {variable_name: np.ndarray}}
    federated_traj : dict of {qualified_stock_name: np.ndarray}
    stock_names : list of str
        Federated trajectory keys for stocks to compare.
    path : Path
    local_only_traj : dict or None
        Local-only simulation trajectories (optional).
    """
    rows = []
    for stock_key in stock_names:
        # Find actual time series
        actual = None
        for org_name, org_ts in actual_ts.items():
            prefix = f"{org_name.lower()}_"
            if stock_key.startswith(prefix):
                local_name = stock_key[len(prefix):]
                actual = org_ts.get(local_name)
                break

        # Get simulated trajectories
        fed = federated_traj.get(stock_key)

        if actual is None:
            continue

        # Align lengths: simulated trajectories may have n+1 entries
        n = len(actual)

        def _align(traj):
            if traj is None:
                return None
            return traj[1:n + 1] if len(traj) > n else traj[:n]

        fed_aligned = _align(fed)
        local_aligned = _align(local_only_traj.get(stock_key) if local_only_traj else None)

        for t in range(n):
            row = {
                "time_step": t,
                "stock": stock_key,
                "actual": actual[t],
            }
            row["simulated_federated"] = (
                fed_aligned[t] if fed_aligned is not None and t < len(fed_aligned) else np.nan
            )
            if local_only_traj is not None:
                row["simulated_local"] = (
                    local_aligned[t] if local_aligned is not None and t < len(local_aligned) else np.nan
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)



def save_equation_quality(qualities: List[EquationQuality], path: Path) -> None:
    """Save equation quality report to CSV."""
    rows = []
    for q in qualities:
        rows.append({
            "org": q.org,
            "flow_name": q.flow_name,
            "equation_type": q.equation_type,
            "r_squared": q.r_squared,
            "n_dependencies": q.n_dependencies,
            "dep_names": "; ".join(q.dep_names),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def save_scalability_results(results: List[dict], path: Path) -> None:
    """Save MPC scalability measurements to CSV.

    Each dict should contain at least:
        n_orgs, n_boundary_flows, n_steps, wall_time_sec,
        mpc_time_sec, local_time_sec, mpc_fraction
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)