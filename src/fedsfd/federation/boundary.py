"""
Boundary equation discovery for federated SFD.

For each matched cross-org flow pair (outflow of org_i → inflow of org_j),
fit a linear regression:
    inflow[t] = α + β · source_stock[t - lag]

The source stock is the stock that the outflow drains from in org_i's SFD.

Updated to work with analyst-defined variable names — no longer depends
on _wip/_throughput/_arrival naming conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from fedsfd.federation.flow_matching import FlowMatch
from fedsfd.mpc.interface import MPCBackend
from fedsfd.sfd.model import SFD


@dataclass
class BoundaryEquation:
    """A learned boundary equation connecting two organizations.

    Attributes
    ----------
    match : FlowMatch
        The flow match this equation describes.
    source_stock_name : str
        Name of the stock in the source org.
    intercept : float
        Regression intercept α.
    slope : float
        Regression slope β.
    lag : int
        Time lag τ*.
    r_squared : float
        Coefficient of determination of the fit.
    eq_id : int or None
        Persistence channel ID. Set when the equation was discovered
        with persist=True (parameters stored as MP-SPDZ secret shares).
        None means parameters are in plaintext (classic mode).
    """
    match: FlowMatch
    source_stock_name: str
    intercept: float
    slope: float
    lag: int
    r_squared: float = 0.0
    eq_id: Optional[int] = None

    @property
    def is_persisted(self) -> bool:
        """Whether this equation's parameters are persisted as secret shares."""
        return self.eq_id is not None

    def predict(self, source_stock_value: float) -> float:
        """Compute boundary flow rate from a source stock value."""
        return max(self.intercept + self.slope * source_stock_value, 0.0)


def discover_boundary_equations(
    matches: List[FlowMatch],
    local_sfds: Dict[str, SFD],
    var_ts: Dict[str, Dict[str, np.ndarray]],
    org_party_map: Dict[str, int],
    mpc_backend: MPCBackend,
    persist: bool = False,
) -> List[BoundaryEquation]:
    """Discover boundary equations for all matched cross-org flows.

    Parameters
    ----------
    matches : list of FlowMatch
        Cross-org flow matches.
    local_sfds : dict of {org_name: SFD}
        Per-org SFDs.
    var_ts : dict of {org_name: {variable_name: np.ndarray}}
        Per-org variable time series (analyst-defined or legacy).
    org_party_map : dict of {org_name: party_index}
    mpc_backend : MPCBackend
    persist : bool
        If True, use secure_regression_persist() to write (alpha, beta)
        as secret shares to MP-SPDZ persistence files. The boundary
        flow program can then read them back without the parameters
        ever leaving the secret-shared domain. Each equation is assigned
        a sequential eq_id (0, 1, 2, ...) used as the persistence
        channel. Default: False (classic mode, reveals params).

    Returns
    -------
    list of BoundaryEquation
    """
    equations = []
    eq_id_counter = 0

    for match in matches:
        out_sfd = local_sfds.get(match.outflow_org)
        in_sfd = local_sfds.get(match.inflow_org)
        if out_sfd is None or in_sfd is None:
            print(f"  Warning: SFD not found for match {match.outflow_org} → {match.inflow_org}")
            continue

        # Find the outflow in the source org's SFD
        out_flow = out_sfd.get_flow(match.outflow_name)
        if out_flow is None:
            print(f"  Warning: outflow '{match.outflow_name}' not found in {match.outflow_org}")
            continue

        # The source stock is the stock the outflow drains from
        source_stock = out_flow.source
        if source_stock is None or source_stock.is_cloud:
            # If the outflow comes from cloud, use the first non-cloud stock
            if out_sfd.stocks:
                source_stock = out_sfd.stocks[0]
            else:
                print(f"  Warning: no source stock for {match.outflow_name}")
                continue

        # Get time series directly from var_ts dict
        out_org_ts = var_ts.get(match.outflow_org, {})
        in_org_ts = var_ts.get(match.inflow_org, {})

        source_stock_ts = out_org_ts.get(source_stock.name)
        inflow_ts = in_org_ts.get(match.inflow_name)

        if source_stock_ts is None:
            print(f"  Warning: no time series for stock '{source_stock.name}' in {match.outflow_org}")
            continue
        if inflow_ts is None:
            print(f"  Warning: no time series for flow '{match.inflow_name}' in {match.inflow_org}")
            continue

        lag = match.lag

        # Align with lag
        if lag > 0 and lag < len(source_stock_ts):
            x = source_stock_ts[:-lag]
            y = inflow_ts[lag:]
        else:
            x = source_stock_ts
            y = inflow_ts

        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        if n < 3:
            continue

        # Fit regression via MPC backend
        party_x = org_party_map.get(match.outflow_org, 0)
        party_y = org_party_map.get(match.inflow_org, 1)

        eq_id = eq_id_counter
        eq_id_counter += 1

        if persist:
            intercept, slope = mpc_backend.secure_regression_persist(
                x, y, party_x, party_y, eq_id
            )
        else:
            intercept, slope = mpc_backend.secure_regression(x, y, party_x, party_y)

        # Compute R²
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        eq = BoundaryEquation(
            match=match,
            source_stock_name=source_stock.name,
            intercept=intercept,
            slope=slope,
            lag=lag,
            r_squared=r_squared,
            eq_id=eq_id if persist else None,
        )
        equations.append(eq)

    return equations


def print_boundary_equations(equations: List[BoundaryEquation]) -> None:
    """Print a summary of boundary equations."""
    print(f"\n{'=' * 60}")
    print(f"Boundary Equations ({len(equations)} total)")
    print(f"{'=' * 60}")
    for eq in equations:
        print(
            f"  {eq.match.outflow_org}/{eq.source_stock_name} "
            f"→ {eq.match.inflow_org}/{eq.match.inflow_name}  "
            f"lag={eq.lag}"
        )
        print(
            f"    f[t] = {eq.intercept:.4f} + {eq.slope:.4f} * "
            f"stock[t-{eq.lag}]  (R²={eq.r_squared:.4f})"
        )