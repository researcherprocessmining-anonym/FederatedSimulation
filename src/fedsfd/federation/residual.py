"""
Residual external flow computation and modeling.

For each external flow of an org, the residual is the portion not
explained by boundary equations:
    ε_f[t] = f[t] - Σ_{matched} eq(s_source[t - lag])

Updated to work with var_ts dicts from analyst-defined variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from fedsfd.federation.boundary import BoundaryEquation
from fedsfd.sfd.model import SFD


@dataclass
class ResidualModel:
    """A model for the residual (unexplained) component of an external flow."""
    org: str
    flow_name: str
    model_type: str = "constant"
    mean: float = 0.0
    trend_slope: float = 0.0
    residual_ts: Optional[np.ndarray] = None

    def predict(self, t: int) -> float:
        """Predict residual at time step t."""
        if self.model_type == "linear_trend":
            return self.mean + self.trend_slope * t
        return self.mean


def compute_residuals(
    boundary_equations: List[BoundaryEquation],
    local_sfds: Dict[str, SFD],
    var_ts: Dict[str, Dict[str, np.ndarray]],
    model_type: str = "constant",
) -> List[ResidualModel]:
    """Compute and model residual flows for all organizations.

    Parameters
    ----------
    boundary_equations : list of BoundaryEquation
    local_sfds : dict of {org_name: SFD}
    var_ts : dict of {org_name: {variable_name: np.ndarray}}
        Per-org variable time series.
    model_type : str
        "constant" or "linear_trend"

    Returns
    -------
    list of ResidualModel
    """
    # Group boundary equations by receiving flow
    eq_by_inflow: Dict[tuple, List[BoundaryEquation]] = {}
    for eq in boundary_equations:
        key = (eq.match.inflow_org, eq.match.inflow_name)
        eq_by_inflow.setdefault(key, []).append(eq)

    residuals = []

    for org_name, sfd in local_sfds.items():
        org_ts = var_ts.get(org_name, {})

        # Process all external inflows
        flow_names_to_check = set()
        for f in sfd.get_external_inflows():
            flow_names_to_check.add(f.name)
        for f in sfd.flows:
            if (org_name, f.name) in eq_by_inflow:
                flow_names_to_check.add(f.name)

        for flow_name in flow_names_to_check:
            actual_ts = org_ts.get(flow_name)
            if actual_ts is None:
                continue

            # Compute predicted portion from boundary equations
            eqs = eq_by_inflow.get((org_name, flow_name), [])
            predicted = np.zeros(len(actual_ts))

            for eq in eqs:
                out_org_ts = var_ts.get(eq.match.outflow_org, {})
                source_ts = out_org_ts.get(eq.source_stock_name)
                if source_ts is None:
                    continue
                for t in range(len(actual_ts)):
                    t_lagged = t - eq.lag
                    if 0 <= t_lagged < len(source_ts):
                        predicted[t] += eq.predict(source_ts[t_lagged])

            residual_ts = actual_ts - predicted

            mean_val = float(np.mean(residual_ts))
            trend = 0.0
            if model_type == "linear_trend" and len(residual_ts) > 1:
                t_idx = np.arange(len(residual_ts), dtype=float)
                if np.std(t_idx) > 0:
                    slope = np.polyfit(t_idx, residual_ts, 1)[0]
                    trend = float(slope)

            residuals.append(ResidualModel(
                org=org_name,
                flow_name=flow_name,
                model_type=model_type,
                mean=mean_val,
                trend_slope=trend,
                residual_ts=residual_ts,
            ))

    return residuals


def print_residuals(residuals: List[ResidualModel]) -> None:
    """Print a summary of residual models."""
    print(f"\n{'=' * 60}")
    print(f"Residual Models ({len(residuals)} total)")
    print(f"{'=' * 60}")
    for r in residuals:
        extra = ""
        if r.model_type == "linear_trend":
            extra = f"  trend={r.trend_slope:.4f}"
        ts_stats = ""
        if r.residual_ts is not None:
            ts_stats = f"  std={np.std(r.residual_ts):.2f}"
        print(
            f"  {r.org}/{r.flow_name:35s} "
            f"model={r.model_type}  mean={r.mean:.2f}{extra}{ts_stats}"
        )