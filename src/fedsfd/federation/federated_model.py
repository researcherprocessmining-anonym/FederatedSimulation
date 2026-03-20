"""
Stitch local SFDs into a federated SFD.

Replaces cloud-connected external flows with:
  - Boundary flows (matched cross-org flows)
  - Residual flows (unexplained external components)

The resulting SFD connects stocks across organizations and can be
exported to Vensim .mdl format.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from fedsfd.federation.boundary import BoundaryEquation
from fedsfd.federation.flow_matching import FlowMatch
from fedsfd.federation.residual import ResidualModel
from fedsfd.sfd.model import AuxVariable, Flow, InfoDependency, SFD, Stock


def build_federated_sfd(
    local_sfds: Dict[str, SFD],
    boundary_equations: List[BoundaryEquation],
    residuals: List[ResidualModel],
) -> SFD:
    """Stitch local SFDs into a federated SFD.

    Parameters
    ----------
    local_sfds : dict of {org_name: SFD}
        Per-organization SFDs.
    boundary_equations : list of BoundaryEquation
        Learned cross-org flow equations.
    residuals : list of ResidualModel
        Residual models for unmatched external flows.

    Returns
    -------
    SFD
        A federated SFD connecting all organizations.
    """
    fed_sfd = SFD(name="Federated_SFD", org="federated")

    # --- Collect all stocks (prefixed with org name for uniqueness) ---
    stock_lookup = {}  # (org, original_name) -> new Stock
    for org_name, sfd in local_sfds.items():
        for stock in sfd.stocks:
            new_name = f"{org_name.lower()}_{stock.name}"
            new_stock = Stock(
                name=new_name,
                org=org_name,
                initial_value=stock.initial_value,
            )
            fed_sfd.stocks.append(new_stock)
            stock_lookup[(org_name, stock.name)] = new_stock

    # --- Collect all auxiliaries (prefixed with org name) ---
    aux_lookup = {}  # (org, original_name) -> new AuxVariable
    for org_name, sfd in local_sfds.items():
        for aux in sfd.auxiliaries:
            new_name = f"{org_name.lower()}_{aux.name}"
            new_aux = AuxVariable(
                name=new_name,
                org=org_name,
                time_series=aux.time_series,
                equation=aux.equation,
                equation_params=aux.equation_params,
            )
            fed_sfd.auxiliaries.append(new_aux)
            aux_lookup[(org_name, aux.name)] = new_aux

    # --- Process internal flows (keep as-is, with updated stock references) ---
    for org_name, sfd in local_sfds.items():
        for flow in sfd.get_internal_flows():
            source = stock_lookup.get((org_name, flow.source.name))
            sink = stock_lookup.get((org_name, flow.sink.name))
            if source is None or sink is None:
                continue
            new_flow = Flow(
                name=f"{org_name.lower()}_{flow.name}",
                org=org_name,
                source=source,
                sink=sink,
                equation=flow.equation,
                equation_params=flow.equation_params,
                time_series=flow.time_series,
            )
            fed_sfd.flows.append(new_flow)

    # --- Create boundary flows from equations ---
    boundary_flow_targets = set()  # Track which (org, flow_name) are handled by boundary
    boundary_flow_lookup = {}  # (org, flow_name) -> boundary Flow object
    renamed_flow_lookup = {}  # (org, flow_name) -> renamed Flow (residual/external)
    for eq in boundary_equations:
        source_stock = stock_lookup.get((eq.match.outflow_org, eq.source_stock_name))
        # Find the sink stock: the stock the inflow feeds into
        in_sfd = local_sfds.get(eq.match.inflow_org)
        in_flow = in_sfd.get_flow(eq.match.inflow_name) if in_sfd else None

        sink_stock = None
        if in_flow and in_flow.sink and not in_flow.sink.is_cloud:
            sink_stock = stock_lookup.get((eq.match.inflow_org, in_flow.sink.name))

        if source_stock is None or sink_stock is None:
            # Fallback: connect to first stock of receiving org
            if in_sfd and in_sfd.stocks:
                sink_stock = stock_lookup.get((eq.match.inflow_org, in_sfd.stocks[0].name))

        if source_stock is None or sink_stock is None:
            continue

        # Build boundary equation callable
        _intercept = eq.intercept
        _slope = eq.slope
        _src_name = source_stock.name
        _lag = eq.lag

        def make_boundary_eq(intercept, slope, src_name, lag):
            def boundary_eq(state, params, t):
                key = f"{src_name}__lag{lag}" if lag > 0 else src_name
                stock_val = state.get(key, state.get(src_name, 0.0))
                return max(intercept + slope * stock_val, 0.0)
            return boundary_eq

        boundary_flow = Flow(
            name=f"boundary_{eq.match.outflow_org}_{eq.match.outflow_name}_to_{eq.match.inflow_org}",
            org="federated",
            source=source_stock,
            sink=sink_stock,
            equation=make_boundary_eq(_intercept, _slope, source_stock.name, _lag),
            equation_params={
                "type": "boundary",
                "intercept": _intercept,
                "slope": _slope,
                "source_stock": source_stock.name,
                "lag": _lag,
            },
        )
        fed_sfd.flows.append(boundary_flow)
        boundary_flow_targets.add((eq.match.inflow_org, eq.match.inflow_name))
        boundary_flow_lookup[(eq.match.inflow_org, eq.match.inflow_name)] = boundary_flow
        boundary_flow_lookup[(eq.match.outflow_org, eq.match.outflow_name)] = boundary_flow

        # Add info dependency
        fed_sfd.dependencies.append(InfoDependency(
            source=source_stock,
            target=boundary_flow,
            lag=_lag,
            correlation=eq.match.correlation,
        ))

    # --- Handle external inflows: residual flows for unexplained portion ---
    for org_name, sfd in local_sfds.items():
        for flow in sfd.get_external_inflows():
            # Find matching residual
            residual = None
            for r in residuals:
                if r.org == org_name and r.flow_name == flow.name:
                    residual = r
                    break

            is_boundary = (org_name, flow.name) in boundary_flow_targets

            # For boundary-matched inflows, skip if residual is negligible
            if is_boundary and residual is not None:
                if abs(residual.mean) < 1e-6 and abs(residual.trend_slope) < 1e-6:
                    continue

            # Sink stock
            sink_stock = None
            if flow.sink and not flow.sink.is_cloud:
                sink_stock = stock_lookup.get((org_name, flow.sink.name))

            if sink_stock is None:
                continue

            # For boundary-matched inflows, the residual captures the
            # unexplained portion (actual - boundary_predicted).
            # For unmatched inflows, it captures the full flow value.
            _mean = residual.mean if residual else 0.0
            _trend = residual.trend_slope if residual else 0.0
            _model_type = residual.model_type if residual else "constant"

            def make_residual_eq(mean, trend, model_type):
                def residual_eq(state, params, t):
                    if model_type == "linear_trend":
                        return max(mean + trend * t, 0.0)
                    return max(mean, 0.0)
                return residual_eq

            residual_flow = Flow(
                name=f"residual_{org_name.lower()}_{flow.name}",
                org=org_name,
                source=fed_sfd.cloud,
                sink=sink_stock,
                equation=make_residual_eq(_mean, _trend, _model_type),
                equation_params={
                    "type": "residual",
                    "mean": _mean,
                    "trend_slope": _trend,
                    "model_type": _model_type,
                },
                time_series=flow.time_series,
            )
            fed_sfd.flows.append(residual_flow)
            renamed_flow_lookup[(org_name, flow.name)] = residual_flow

        for flow in sfd.get_external_outflows():
            # External outflows go to cloud in the federated model
            source_stock = None
            if flow.source and not flow.source.is_cloud:
                source_stock = stock_lookup.get((org_name, flow.source.name))

            if source_stock is None:
                continue

            ext_out_flow = Flow(
                name=f"external_{org_name.lower()}_{flow.name}",
                org=org_name,
                source=source_stock,
                sink=fed_sfd.cloud,
                equation=flow.equation,
                equation_params=flow.equation_params,
                time_series=flow.time_series,
            )
            fed_sfd.flows.append(ext_out_flow)
            renamed_flow_lookup[(org_name, flow.name)] = ext_out_flow

    # --- Copy information dependencies from local SFDs ---
    for org_name, sfd in local_sfds.items():
        for dep in sfd.dependencies:
            # Map source and target to federated names
            new_source = _map_var(dep.source, org_name, stock_lookup, aux_lookup, fed_sfd, boundary_flow_lookup, renamed_flow_lookup)
            new_target = _map_var(dep.target, org_name, stock_lookup, aux_lookup, fed_sfd, boundary_flow_lookup, renamed_flow_lookup)
            if new_source is not None and new_target is not None:
                fed_sfd.dependencies.append(InfoDependency(
                    source=new_source,
                    target=new_target,
                    lag=dep.lag,
                    correlation=dep.correlation,
                ))

    return fed_sfd


def _map_var(var, org_name, stock_lookup, aux_lookup, fed_sfd, boundary_flow_lookup=None, renamed_flow_lookup=None):
    """Map a local SFD variable to its federated counterpart."""
    if isinstance(var, Stock):
        if var.is_cloud:
            return fed_sfd.cloud
        return stock_lookup.get((org_name, var.name))
    elif isinstance(var, Flow):
        # Check if this flow was replaced by a boundary flow
        if boundary_flow_lookup:
            boundary = boundary_flow_lookup.get((org_name, var.name))
            if boundary is not None:
                return boundary
        # Check if this flow was renamed (residual/external)
        if renamed_flow_lookup:
            renamed = renamed_flow_lookup.get((org_name, var.name))
            if renamed is not None:
                return renamed
        fed_name = f"{org_name.lower()}_{var.name}"
        return fed_sfd.get_flow(fed_name)
    elif isinstance(var, AuxVariable):
        return aux_lookup.get((org_name, var.name))
    return None