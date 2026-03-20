"""
Evaluation metrics for federated SFD simulation.

Provides:
  - Time-series accuracy: RMSE, MAE, MAPE, R², normalized error
  - Per-variable R² reporting for equation quality diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from fedsfd.sfd.model import SFD


# =========================================================================
# Time-series accuracy metrics
# =========================================================================

@dataclass
class TimeSeriesMetrics:
    """Accuracy metrics for a single variable's trajectory.

    Attributes
    ----------
    variable : str
        Variable name (e.g., "company_order_backlog").
    rmse : float
        Root Mean Squared Error.
    mae : float
        Mean Absolute Error.
    mape : float
        Mean Absolute Percentage Error (in %, NaN if actual is zero).
    r_squared : float
        Coefficient of determination.
    actual_mean : float
        Mean of the actual (ground truth) series.
    simulated_mean : float
        Mean of the simulated series.
    n_points : int
        Number of compared time points.
    normalized_errors : np.ndarray
        Per-step (sim - actual) / actual, NaN where actual is zero.
    """
    variable: str
    rmse: float
    mae: float
    mape: float
    r_squared: float
    actual_mean: float
    simulated_mean: float
    n_points: int
    normalized_errors: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def compute_ts_metrics(
    actual: np.ndarray,
    simulated: np.ndarray,
    variable_name: str = "",
) -> TimeSeriesMetrics:
    """Compute accuracy metrics between actual and simulated time series.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth time series.
    simulated : np.ndarray
        Simulated time series (same length as actual).
    variable_name : str
        Label for reporting.

    Returns
    -------
    TimeSeriesMetrics
    """
    n = min(len(actual), len(simulated))
    if n == 0:
        return TimeSeriesMetrics(
            variable=variable_name,
            rmse=np.nan, mae=np.nan, mape=np.nan, r_squared=np.nan,
            actual_mean=np.nan, simulated_mean=np.nan, n_points=0,
        )

    act = actual[:n].astype(float)
    sim = simulated[:n].astype(float)

    errors = sim - act
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # MAPE: skip zero actuals
    nonzero = np.abs(act) > 1e-12
    if np.any(nonzero):
        mape = float(np.mean(np.abs(errors[nonzero] / act[nonzero])) * 100)
    else:
        mape = np.nan

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((act - np.mean(act)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    # Normalized errors
    norm_errors = np.full(n, np.nan)
    norm_errors[nonzero] = errors[nonzero] / act[nonzero]

    return TimeSeriesMetrics(
        variable=variable_name,
        rmse=rmse,
        mae=mae,
        mape=mape,
        r_squared=r_squared,
        actual_mean=float(np.mean(act)),
        simulated_mean=float(np.mean(sim)),
        n_points=n,
        normalized_errors=norm_errors,
    )


def compute_all_stock_metrics(
    actual_ts: Dict[str, np.ndarray],
    simulated_traj: Dict[str, np.ndarray],
    sfds: Dict[str, SFD],
) -> List[TimeSeriesMetrics]:
    """Compute accuracy metrics for all stocks across all organizations.

    Parameters
    ----------
    actual_ts : dict of {org_name: {variable_name: np.ndarray}}
        Per-org ground-truth time series.
    simulated_traj : dict of {trajectory_key: np.ndarray}
        Simulated trajectories (keys like "company_order_backlog").
    sfds : dict of {org_name: SFD}
        Per-org SFDs (to identify stock variables).

    Returns
    -------
    list of TimeSeriesMetrics
    """
    metrics = []
    for org_name, sfd in sfds.items():
        org_ts = actual_ts.get(org_name, {})
        for stock in sfd.stocks:
            actual = org_ts.get(stock.name)
            traj_key = f"{org_name.lower()}_{stock.name}"
            simulated = simulated_traj.get(traj_key)

            if actual is None or simulated is None:
                continue

            # Simulated stock trajectories have n_steps+1 entries (including
            # initial value); actual has n_windows entries.  Align by
            # comparing simulated[1:] with actual.
            sim_aligned = simulated[1:] if len(simulated) > len(actual) else simulated
            m = compute_ts_metrics(actual, sim_aligned, variable_name=traj_key)
            metrics.append(m)

    return metrics



# =========================================================================
# Equation quality (R² reporting)
# =========================================================================

@dataclass
class EquationQuality:
    """R² and fit quality for a single flow's equation."""
    org: str
    flow_name: str
    equation_type: str  # "linear", "draining", "constant", etc.
    r_squared: float
    n_dependencies: int
    dep_names: List[str] = field(default_factory=list)


def collect_equation_quality(sfds: Dict[str, SFD]) -> List[EquationQuality]:
    """Collect R² and equation type for all flows across all SFDs.

    Parameters
    ----------
    sfds : dict of {org_name: SFD}

    Returns
    -------
    list of EquationQuality
    """
    qualities = []
    for org_name, sfd in sfds.items():
        for flow in sfd.flows:
            params = flow.equation_params or {}
            eq_type = params.get("type", "unknown")
            r2 = params.get("r_squared", np.nan)
            dep_names = params.get("dep_names", [])

            qualities.append(EquationQuality(
                org=org_name,
                flow_name=flow.name,
                equation_type=eq_type,
                r_squared=r2,
                n_dependencies=len(dep_names),
                dep_names=list(dep_names),
            ))
    return qualities


# =========================================================================
# Pretty-printing
# =========================================================================

def print_ts_metrics_table(metrics: List[TimeSeriesMetrics], title: str = "") -> None:
    """Print a formatted table of time-series metrics."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")

    header = (
        f"  {'Variable':<42s} {'RMSE':>8s} {'MAE':>8s} "
        f"{'MAPE%':>8s} {'R²':>8s} {'ActMean':>8s} {'SimMean':>8s}"
    )
    print(header)
    print("  " + "-" * 78)

    for m in metrics:
        mape_str = f"{m.mape:8.1f}" if not np.isnan(m.mape) else "     N/A"
        r2_str = f"{m.r_squared:8.3f}" if not np.isnan(m.r_squared) else "     N/A"
        print(
            f"  {m.variable:<42s} {m.rmse:8.2f} {m.mae:8.2f} "
            f"{mape_str} {r2_str} {m.actual_mean:8.1f} {m.simulated_mean:8.1f}"
        )



def print_equation_quality_table(qualities: List[EquationQuality]) -> None:
    """Print equation quality for all flows."""
    print(f"\n{'=' * 75}")
    print(f"  Equation Quality (R² per flow)")
    print(f"{'=' * 75}")
    print(f"  {'Org':<12s} {'Flow':<30s} {'Type':<12s} {'R²':>8s} {'#Deps':>6s}")
    print("  " + "-" * 73)

    for q in qualities:
        r2_str = f"{q.r_squared:.3f}" if not np.isnan(q.r_squared) else "   N/A"
        print(
            f"  {q.org:<12s} {q.flow_name:<30s} "
            f"{q.equation_type:<12s} {r2_str:>8s} {q.n_dependencies:>6d}"
        )