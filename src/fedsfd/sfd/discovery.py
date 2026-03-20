"""
SFD discovery from analyst-defined variables via the Pourbafrani pipeline.

Reimplements the algorithm from Pourbafrani & van der Aalst (2022),
"Discovering System Dynamics Simulation Models Using Process Mining."

The pipeline:
  1. Analyst defines variables with role hints and (optionally) material
     flow topology (from_stock / to_stock on flow definitions).
  2. CLD discovery: pairwise lagged correlations between all variables.
  3. Label CLD: use analyst role hints (stock / flow / auxiliary).
  4. SD constraint check: apply Table 4 rules.  Skip synthetic flow
     insertion when analyst flows already connect the stock pair
     (including transitive connections through chains).
  5. SFD construction: for flows with analyst-declared from_stock/to_stock,
     use those directly.  For others, use Definition 7 (mapf from M_set).
  5b. Prune spurious information dependencies.
  5c. Ensure every internal flow has a dependency on its source stock
      (structural SD guarantee for negative feedback / stability).
  6. Fit equations via linear regression.  Validate R² and fall back to:
     - draining fraction (flow = α * stock) for internal flows with poor fit,
     - mean-constant for external inflows.

The original code at mbafrani/PMSD has no license, so this is a clean
reimplementation from the paper's formal definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from fedsfd.sfd.model import (
    AuxVariable,
    Flow,
    InfoDependency,
    SFD,
    Stock,
)


# =========================================================================
# Utility
# =========================================================================

def _lagged_correlation(
    x: np.ndarray, y: np.ndarray, max_lag: int,
) -> Tuple[float, int]:
    """Find the lag maximising |Pearson correlation| between x and y."""
    n = len(x)
    best_corr = 0.0
    best_lag = 0
    for lag in range(min(max_lag + 1, n - 2)):
        if lag == 0:
            xi, yi = x, y
        else:
            xi = x[:-lag]
            yi = y[lag:]
        if len(xi) < 3:
            continue
        if np.std(xi) < 1e-12 or np.std(yi) < 1e-12:
            continue
        corr, _ = stats.pearsonr(xi, yi)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    return best_corr, best_lag


# =========================================================================
# Step 2: CLD discovery
# =========================================================================

@dataclass
class CLDRelation:
    """A directed relation in the Causal-Loop Diagram."""
    source: str
    target: str
    sign: str            # "+" or "-"
    correlation: float
    lag: int


@dataclass
class CLD:
    """Causal-Loop Diagram: variables + directed signed relations."""
    variables: List[str]
    relations: List[CLDRelation]


def discover_cld(
    var_ts: Dict[str, np.ndarray],
    correlation_threshold: float = 0.3,
    max_lag: int = 3,
) -> CLD:
    """Discover a CLD from a dict of time series."""
    names = sorted(var_ts.keys())
    relations = []
    for src in names:
        for tgt in names:
            if src == tgt:
                continue
            x = var_ts[src]
            y = var_ts[tgt]
            if len(x) != len(y) or len(x) < 4:
                continue
            corr, lag = _lagged_correlation(x, y, max_lag)
            if abs(corr) >= correlation_threshold:
                relations.append(CLDRelation(
                    source=src, target=tgt,
                    sign="+" if corr > 0 else "-",
                    correlation=corr, lag=lag,
                ))
    return CLD(variables=names, relations=relations)


# =========================================================================
# Step 3: Label CLD
# =========================================================================

def label_cld(
    cld: CLD,
    variable_roles: Dict[str, str],
) -> Tuple[set, set, set]:
    """Partition CLD variables into S, F, A based on semantic role."""
    S, F, A = set(), set(), set()
    for v in cld.variables:
        role = variable_roles.get(v, "auxiliary")
        if role == "stock":
            S.add(v)
        elif role == "flow":
            F.add(v)
        else:
            A.add(v)
    return S, F, A


# =========================================================================
# Step 4: SD constraint check
# =========================================================================

def _var_type(name: str, S: set, F: set, A: set) -> str:
    if name in S:
        return "stock"
    if name in F:
        return "flow"
    return "auxiliary"


def _find_existing_flow_connections(
    cld: CLD, S: set, F: set, A: set,
    analyst_topology: Dict[str, Dict[str, str]],
) -> Set[Tuple[str, str]]:
    """Find stock pairs already connected by analyst-declared flows.

    Uses from_stock/to_stock declarations when available, and falls back
    to CLD relation analysis.
    """
    connected = set()

    # From analyst topology (most reliable)
    for flow_name, topo in analyst_topology.items():
        src = topo.get("from_stock", "cloud")
        snk = topo.get("to_stock", "cloud")
        if src != "cloud" and snk != "cloud":
            connected.add((src, snk))
            connected.add((snk, src))  # bidirectional suppression
        elif src != "cloud":
            # Flow drains src — src is connected to this flow's path
            for flow_name2, topo2 in analyst_topology.items():
                snk2 = topo2.get("to_stock", "cloud")
                if snk2 != "cloud" and snk2 != src:
                    # Two flows sharing a chain: src → flow → ? and ? → flow2 → snk2
                    pass  # don't over-connect

    # Also check CLD relations for analyst-defined flows
    flow_stocks: Dict[str, List[str]] = {}
    for rel in cld.relations:
        src_type = _var_type(rel.source, S, F, A)
        tgt_type = _var_type(rel.target, S, F, A)
        if src_type == "flow" and tgt_type == "stock":
            flow_stocks.setdefault(rel.source, []).append(rel.target)
        if src_type == "stock" and tgt_type == "flow":
            flow_stocks.setdefault(rel.target, []).append(rel.source)

    for flow_name, stocks in flow_stocks.items():
        for i in range(len(stocks)):
            for j in range(len(stocks)):
                if i != j:
                    connected.add((stocks[i], stocks[j]))

    # Transitive closure: if A→B and B→C are connected (via any path),
    # mark A→C as connected too.  This prevents synthetic flow insertion
    # between non-adjacent stocks in a chain (e.g., weighing_queue and
    # loading_bay_queue when terminal_inventory sits between them).
    changed = True
    while changed:
        changed = False
        new_pairs = set()
        for (a, b) in connected:
            for (c, d) in connected:
                if b == c and (a, d) not in connected and a != d:
                    new_pairs.add((a, d))
        if new_pairs:
            connected.update(new_pairs)
            changed = True

    return connected


def sd_constraint_check(
    cld: CLD,
    S: set, F: set, A: set,
    analyst_topology: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[set, set, List[CLDRelation], set, set]:
    """Apply SD constraints from Table 4 of the paper.

    When analyst-defined flows already connect two stocks, skip
    inserting synthetic intermediate flows.
    """
    if analyst_topology is None:
        analyst_topology = {}

    existing_connections = _find_existing_flow_connections(
        cld, S, F, A, analyst_topology
    )

    new_relations = list(cld.relations)
    inserted_flows = set()

    inserts = []
    removals = []
    for rel in cld.relations:
        src_type = _var_type(rel.source, S, F, A)
        tgt_type = _var_type(rel.target, S, F, A)

        if src_type == "stock" and tgt_type == "stock":
            if (rel.source, rel.target) in existing_connections:
                # Already connected by analyst flow — keep as info dep
                continue

            flow_name = f"flow_{rel.source}_to_{rel.target}"
            if flow_name not in inserted_flows:
                inserted_flows.add(flow_name)
                F.add(flow_name)
                inserts.append(CLDRelation(
                    source=rel.source, target=flow_name,
                    sign=rel.sign, correlation=rel.correlation, lag=rel.lag,
                ))
                inserts.append(CLDRelation(
                    source=flow_name, target=rel.target,
                    sign="+", correlation=rel.correlation, lag=rel.lag,
                ))
                removals.append(rel)

    for r in removals:
        if r in new_relations:
            new_relations.remove(r)
    new_relations.extend(inserts)

    M_set = set()
    I_set = set()

    for rel in new_relations:
        src_type = _var_type(rel.source, S, F, A)
        tgt_type = _var_type(rel.target, S, F, A)

        if src_type == "flow" and tgt_type == "stock":
            M_set.add((rel.source, rel.target, rel.sign))
        elif tgt_type in ("auxiliary", "flow") and src_type in ("stock", "flow", "auxiliary"):
            I_set.add((rel.source, rel.target))

    return S, F, new_relations, I_set, M_set


# =========================================================================
# Step 5: SFD construction
# =========================================================================

def build_sfd_from_labeled_cld(
    org: str,
    S: set, F: set, A: set,
    M_set: set, I_set: set,
    relations: List[CLDRelation],
    var_ts: Dict[str, np.ndarray],
    analyst_topology: Optional[Dict[str, Dict[str, str]]] = None,
) -> SFD:
    """Build an SFD from a labeled, constraint-checked CLD.

    For flows with analyst-declared from_stock/to_stock, use those
    directly (bypassing the correlation-based mapf from Definition 7).
    For other flows, fall back to Definition 7.
    """
    if analyst_topology is None:
        analyst_topology = {}

    sfd = SFD(name=f"{org}_SFD", org=org)

    stock_objs = {}
    for s_name in sorted(S):
        ts = var_ts.get(s_name, np.array([]))
        init_val = float(ts[0]) if len(ts) > 0 else 0.0
        stock = Stock(name=s_name, org=org, initial_value=init_val)
        sfd.stocks.append(stock)
        stock_objs[s_name] = stock

    for a_name in sorted(A):
        ts = var_ts.get(a_name, np.array([]))
        aux = AuxVariable(name=a_name, org=org, time_series=ts)
        sfd.auxiliaries.append(aux)

    for f_name in sorted(F):
        ts = var_ts.get(f_name, np.array([]))

        # Check if analyst declared the topology for this flow
        topo = analyst_topology.get(f_name)
        if topo is not None:
            src_name = topo.get("from_stock", "cloud")
            snk_name = topo.get("to_stock", "cloud")

            source_stock = (
                stock_objs.get(src_name, sfd.cloud)
                if src_name != "cloud" else sfd.cloud
            )
            sink_stock = (
                stock_objs.get(snk_name, sfd.cloud)
                if snk_name != "cloud" else sfd.cloud
            )
        else:
            # Fall back to Definition 7: derive mapf from M_set
            m_rels = [(s, sign) for (f, s, sign) in M_set if f == f_name]
            source_stock = None
            sink_stock = None
            for s_name, sign in m_rels:
                stock = stock_objs.get(s_name)
                if stock is None:
                    continue
                if sign == "-":
                    source_stock = stock
                elif sign == "+":
                    sink_stock = stock
            if source_stock is None:
                source_stock = sfd.cloud
            if sink_stock is None:
                sink_stock = sfd.cloud

        flow_obj = Flow(
            name=f_name, org=org,
            source=source_stock, sink=sink_stock,
            time_series=ts,
        )
        sfd.flows.append(flow_obj)

    # Create information dependencies (only from I_set)
    all_objs = {}
    for s in sfd.stocks:
        all_objs[s.name] = s
    for f in sfd.flows:
        all_objs[f.name] = f
    for a in sfd.auxiliaries:
        all_objs[a.name] = a

    for (src_name, tgt_name) in I_set:
        src_obj = all_objs.get(src_name)
        tgt_obj = all_objs.get(tgt_name)
        if src_obj is None or tgt_obj is None:
            continue
        if not isinstance(tgt_obj, (Flow, AuxVariable)):
            continue
        corr, lag = 0.0, 0
        for rel in relations:
            if rel.source == src_name and rel.target == tgt_name:
                corr = rel.correlation
                lag = rel.lag
                break
        sfd.dependencies.append(InfoDependency(
            source=src_obj, target=tgt_obj,
            lag=lag, correlation=corr,
        ))

    return sfd


# =========================================================================
# Step 5b: Prune information dependencies
# =========================================================================

def prune_dependencies(sfd: SFD) -> None:
    """Remove information dependencies that would create unstable feedback.

    Pruning rules:

    1. A stock should only drive a flow if it is the flow's **source**
       (the stock the flow drains from).  The sink stock does NOT drive
       the flow — it merely receives material.  This is fundamental SD
       theory: a stock's level influences its outflow rate, not its
       inflow rate.  Allowing sink→flow dependencies creates positive
       feedback loops (more stock → more inflow → even more stock).

    2. A flow should not drive another flow (already handled by Table 4).

    3. Auxiliary → flow dependencies are kept (rate modifiers like
       utilization or congestion).

    4. Per flow, limit to at most 2 dependencies (source stock + one
       auxiliary), preferring higher |correlation|.
    """
    # Only source stocks can drive flows — NOT sink stocks
    source_stock_connections = set()  # (stock_name, flow_name)
    for flow in sfd.flows:
        if flow.source and not flow.source.is_cloud:
            source_stock_connections.add((flow.source.name, flow.name))

    # Group by target
    deps_by_target: Dict[str, List[InfoDependency]] = {}
    for dep in sfd.dependencies:
        tgt_name = dep.target.name
        deps_by_target.setdefault(tgt_name, []).append(dep)

    # External inflows (from cloud) should have NO internal dependencies.
    # Their rate is determined by the upstream org (boundary equation in
    # federated mode) or treated as exogenous input — not by internal
    # stocks or auxiliaries of the receiving org.
    external_inflow_names = set()
    for flow in sfd.flows:
        if flow.source and flow.source.is_cloud:
            external_inflow_names.add(flow.name)

    kept = []
    for tgt_name, deps in deps_by_target.items():
        target_obj = deps[0].target
        candidates = []

        # Drop all dependencies targeting external inflows
        if tgt_name in external_inflow_names:
            continue

        for dep in deps:
            src_obj = dep.source

            if isinstance(src_obj, Stock) and isinstance(target_obj, Flow):
                # Stock → flow: only if this stock is the flow's SOURCE
                if (src_obj.name, target_obj.name) in source_stock_connections:
                    candidates.append(dep)

            elif isinstance(src_obj, AuxVariable):
                # Auxiliary → flow/aux: keep
                candidates.append(dep)

            elif isinstance(src_obj, Stock) and isinstance(target_obj, AuxVariable):
                # Stock → aux: keep (stock level influences derived vars)
                candidates.append(dep)

            elif isinstance(src_obj, Flow):
                # Flow → aux/flow: keep (e.g. moving average of throughput)
                candidates.append(dep)

        # Limit to top 2 by |correlation|
        candidates.sort(key=lambda d: abs(d.correlation), reverse=True)
        kept.extend(candidates[:2])

    sfd.dependencies = kept


def _compute_pipeline_depth(sfd: SFD) -> Dict[str, int]:
    """Compute topological depth for stocks and flows from the material flow graph.

    The material flow chain defines a DAG:
        source_stock → flow → sink_stock → next_flow → ...

    Each variable gets a depth based on its position in this chain.
    Flows from cloud start at depth 0.  Variables not part of the
    material flow chain (auxiliaries) are excluded from the result.
    """
    # Build adjacency from the stock-flow chain
    successors: Dict[str, Set[str]] = {}
    predecessors: Dict[str, Set[str]] = {}
    pipeline_vars: Set[str] = set()

    for flow in sfd.flows:
        pipeline_vars.add(flow.name)
        src_name = flow.source.name if flow.source and not flow.source.is_cloud else None
        snk_name = flow.sink.name if flow.sink and not flow.sink.is_cloud else None

        if src_name:
            pipeline_vars.add(src_name)
            successors.setdefault(src_name, set()).add(flow.name)
            predecessors.setdefault(flow.name, set()).add(src_name)
        if snk_name:
            pipeline_vars.add(snk_name)
            successors.setdefault(flow.name, set()).add(snk_name)
            predecessors.setdefault(snk_name, set()).add(flow.name)

    for s in sfd.stocks:
        pipeline_vars.add(s.name)

    # BFS from roots (no material-flow predecessor)
    depth: Dict[str, int] = {}
    queue = []
    for name in pipeline_vars:
        if name not in predecessors or not predecessors[name]:
            depth[name] = 0
            queue.append(name)

    while queue:
        current = queue.pop(0)
        for succ in successors.get(current, set()):
            new_depth = depth[current] + 1
            if succ not in depth or new_depth > depth[succ]:
                depth[succ] = new_depth
                queue.append(succ)

    return depth


def prune_backward_dependencies(sfd: SFD) -> None:
    """Remove information dependencies that point backward in the material flow pipeline.

    In a stock-flow chain (cloud → flow → stock → flow → stock → ...),
    downstream variables are correlated with upstream ones simply because
    they process the same workload.  These backward correlations are
    spurious — a downstream flow does not causally influence an upstream
    flow or stock.

    Uses the topological depth from the material flow graph: if the
    source variable has depth >= the target variable, the dependency
    points backward (or sideways) and is removed.  Dependencies
    involving auxiliaries (not part of the material flow chain) are
    kept unconditionally.
    """
    depth = _compute_pipeline_depth(sfd)

    kept = []
    for dep in sfd.dependencies:
        src_depth = depth.get(dep.source.name)
        tgt_depth = depth.get(dep.target.name)

        # If either variable is not in the pipeline (e.g. auxiliary), keep
        if src_depth is None or tgt_depth is None:
            kept.append(dep)
            continue

        # Only keep if source is strictly upstream of target
        if src_depth < tgt_depth:
            kept.append(dep)

    sfd.dependencies = kept


def ensure_source_stock_dependencies(sfd: SFD, var_ts: Dict[str, np.ndarray]) -> None:
    """Guarantee every internal flow has an information dependency on its source stock.

    In SD theory, an outflow rate must depend on the stock it drains.
    Without this, the stock has no negative feedback and can grow without
    bound (constant outflow < variable inflow → monotonic accumulation).

    The CLD correlation discovery may miss this link when the processing
    rate is roughly constant (weak correlation with queue length).  This
    function adds the structural dependency unconditionally, using the
    empirical correlation as metadata.  The equation fitting step will
    then calibrate the coefficient from data or fall back to a draining
    fraction if the fit is poor.

    This only applies to flows whose source is a non-cloud stock.
    External inflows (from cloud) are unaffected.
    """
    # Build lookup of existing dependencies
    existing_deps = set()
    for dep in sfd.dependencies:
        existing_deps.add((dep.source.name, dep.target.name))

    for flow in sfd.flows:
        if flow.source is None or flow.source.is_cloud:
            continue

        source_stock = flow.source
        if (source_stock.name, flow.name) in existing_deps:
            continue  # already has this dependency

        # Compute correlation for metadata (informational, not gating)
        src_ts = var_ts.get(source_stock.name)
        flow_ts = var_ts.get(flow.name)
        corr = 0.0
        if (src_ts is not None and flow_ts is not None
                and len(src_ts) == len(flow_ts) and len(src_ts) >= 4):
            if np.std(src_ts) > 1e-12 and np.std(flow_ts) > 1e-12:
                corr, _ = stats.pearsonr(src_ts, flow_ts)

        sfd.dependencies.append(InfoDependency(
            source=source_stock,
            target=flow,
            lag=0,
            correlation=float(corr),
        ))


# =========================================================================
# Step 6: Equation fitting
# =========================================================================

MIN_R_SQUARED = 0.1  # minimum R² to keep a fitted equation


def fit_equations(sfd: SFD, var_ts: Dict[str, np.ndarray]) -> None:
    """Fit linear regression equations for all flows and auxiliaries.

    For each flow/auxiliary with information dependencies, fit:
      value[t] = β₀ + Σ βᵢ * dep_i[t - lag_i]

    If the R² is below MIN_R_SQUARED:
      - For flows with a non-cloud source stock: fall back to a
        **draining fraction** equation:
            flow[t] = α * source_stock[t]
        where α = mean(flow) / mean(source_stock).
        This ensures negative feedback: as the stock grows, the outflow
        grows proportionally, preventing unbounded accumulation.
      - For external inflows (from cloud) or auxiliaries: fall back to
        a mean-constant equation.

    This implements Pourbafrani et al.'s point 2 ("if the relationship
    cannot be fitted well, it will typically not be kept") while adding
    the SD structural guarantee that outflows respond to their stock.
    """
    for target_var in list(sfd.flows) + list(sfd.auxiliaries):
        deps = sfd.get_dependencies_for(target_var)

        y = var_ts.get(target_var.name)
        if y is None or len(y) < 4:
            _set_fallback_eq(target_var, sfd, var_ts)
            continue

        if not deps:
            _set_fallback_eq(target_var, sfd, var_ts)
            continue

        dep_names = []
        dep_lags = []
        X_cols = []

        for dep in deps:
            src_ts = var_ts.get(dep.source.name)
            if src_ts is None or len(src_ts) != len(y):
                continue
            lag = dep.lag
            if lag > 0:
                col = np.zeros(len(y))
                col[lag:] = src_ts[:-lag]
            else:
                col = src_ts.copy()
            dep_names.append(dep.source.name)
            dep_lags.append(lag)
            X_cols.append(col)

        if not X_cols:
            _set_fallback_eq(target_var, sfd, var_ts)
            continue

        X = np.column_stack(X_cols)
        reg = LinearRegression()
        reg.fit(X, y)

        # Evaluate fit quality
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if r_squared < MIN_R_SQUARED:
            _set_fallback_eq(target_var, sfd, var_ts)
            continue

        intercept = float(reg.intercept_)
        coefficients = [float(c) for c in reg.coef_]

        def make_eq(dn, dl, inter, coefs):
            def equation(state, params, t):
                val = params["intercept"]
                for name, lag, coef in zip(
                    params["dep_names"], params["dep_lags"],
                    params["coefficients"],
                ):
                    key = f"{name}__lag{lag}" if lag > 0 else name
                    val += coef * state.get(key, state.get(name, 0.0))
                return max(val, 0.0)
            return equation

        params = {
            "intercept": intercept,
            "dep_names": dep_names,
            "dep_lags": dep_lags,
            "coefficients": coefficients,
            "r_squared": r_squared,
        }
        target_var.equation = make_eq(dep_names, dep_lags, intercept, coefficients)
        target_var.equation_params = params


def fit_stock_equations(sfd: SFD) -> None:
    """Set accumulation equations on all stocks per Definition 5.

    For each stock v ∈ S:
        v[t] = v[t-1] + Σ η((f, v)) * f[t]

    where η = +1 for inflows (f feeds into v) and η = -1 for outflows
    (f drains from v).  M = material flow relations (R \\ I).
    """
    for stock in sfd.stocks:
        if stock.is_cloud:
            continue

        inflows = sfd.get_stock_inflows(stock)
        outflows = sfd.get_stock_outflows(stock)

        inflow_names = [f.name for f in inflows]
        outflow_names = [f.name for f in outflows]

        stock.equation_params = {
            "type": "accumulation",
            "inflows": inflow_names,
            "outflows": outflow_names,
            "initial_value": stock.initial_value,
        }

        def make_accum_eq(s_name):
            def equation(state, params, t):
                prev = state.get(s_name, params["initial_value"])
                net = (
                    sum(state.get(f, 0.0) for f in params["inflows"])
                    - sum(state.get(f, 0.0) for f in params["outflows"])
                )
                return max(prev + net, 0.0)
            return equation

        stock.equation = make_accum_eq(stock.name)


def _set_fallback_eq(target_var, sfd: SFD, var_ts: Dict[str, np.ndarray]) -> None:
    """Set a fallback equation on a flow or auxiliary.

    For internal flows (non-cloud source stock): use draining fraction
        flow[t] = α * source_stock[t],  α = mean(flow) / mean(stock)
    For external inflows or auxiliaries: use mean-constant.
    """
    # Check if this is an internal flow with a source stock
    source_stock = None
    if isinstance(target_var, Flow) and target_var.source and not target_var.source.is_cloud:
        source_stock = target_var.source

    if source_stock is not None:
        _set_draining_fraction_eq(target_var, source_stock, var_ts)
    else:
        _set_constant_eq(target_var, var_ts)


def _set_draining_fraction_eq(
    target_var, source_stock: Stock, var_ts: Dict[str, np.ndarray],
) -> None:
    """Set a draining-fraction equation: flow = α * source_stock.

    The fraction α = mean(flow) / mean(stock) represents the average
    rate at which the stock drains per time step.  This is the simplest
    stock-dependent equation that provides negative feedback:
    as the stock grows, the outflow grows proportionally.

    In SD terms, this is equivalent to a first-order exponential drain
    with residence time τ = 1/α.
    """
    flow_ts = var_ts.get(target_var.name)
    stock_ts = var_ts.get(source_stock.name)

    if (flow_ts is not None and stock_ts is not None
            and len(flow_ts) > 0 and len(stock_ts) > 0):
        mean_flow = float(np.mean(flow_ts))
        mean_stock = float(np.mean(stock_ts))
        if mean_stock > 1e-10:
            alpha = mean_flow / mean_stock
        else:
            alpha = mean_flow  # stock near zero; degenerate case
    else:
        alpha = 0.0

    dep_name = source_stock.name

    target_var.equation_params = {
        "type": "draining_fraction",
        "intercept": 0.0,
        "dep_names": [dep_name],
        "dep_lags": [0],
        "coefficients": [alpha],
        "alpha": alpha,
        "source_stock_mean": float(np.mean(stock_ts)) if stock_ts is not None else 0.0,
        "flow_mean": float(np.mean(flow_ts)) if flow_ts is not None else 0.0,
    }

    def make_drain_eq(dep, a):
        def equation(state, params, t):
            stock_val = state.get(dep, 0.0)
            return max(params["coefficients"][0] * stock_val, 0.0)
        return equation

    target_var.equation = make_drain_eq(dep_name, alpha)


def _set_constant_eq(target_var, var_ts: Dict[str, np.ndarray]) -> None:
    """Set a mean-constant equation on a flow or auxiliary."""
    ts = var_ts.get(target_var.name)
    if ts is not None and len(ts) > 0:
        mean_val = float(np.mean(ts))
    else:
        mean_val = 0.0

    target_var.equation_params = {
        "type": "constant",
        "intercept": mean_val,
        "dep_names": [],
        "dep_lags": [],
        "coefficients": [],
    }

    def make_const_eq(val):
        def equation(state, params, t):
            return max(val, 0.0)
        return equation

    target_var.equation = make_const_eq(mean_val)


# =========================================================================
# Main entry points
# =========================================================================

def _extract_analyst_topology(
    sfd_variables_config: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, str]]:
    """Extract from_stock/to_stock declarations from config.

    Returns {flow_name: {"from_stock": ..., "to_stock": ...}} for
    flows that have analyst-declared topology.
    """
    if sfd_variables_config is None:
        return {}
    topology = {}
    for var_name, var_def in sfd_variables_config.items():
        if var_def.get("role") != "flow":
            continue
        topo = {}
        if "from_stock" in var_def:
            topo["from_stock"] = var_def["from_stock"]
        if "to_stock" in var_def:
            topo["to_stock"] = var_def["to_stock"]
        if topo:
            topology[var_name] = topo
    return topology


def discover_sfd_from_variables(
    var_ts: Dict[str, np.ndarray],
    variable_roles: Dict[str, str],
    org: str,
    correlation_threshold: float = 0.3,
    max_lag: int = 3,
    sfd_variables_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> SFD:
    """Discover an SFD using analyst-defined variables via the Pourbafrani pipeline.

    Parameters
    ----------
    var_ts : dict of {variable_name: np.ndarray}
        Time series for each variable.
    variable_roles : dict of {variable_name: "stock"|"flow"|"auxiliary"}
        Semantic role hint from the analyst.
    org : str
        Organization name.
    correlation_threshold : float
    max_lag : int
    sfd_variables_config : dict or None
        The raw YAML config for this org's sfd_variables, used to
        extract from_stock/to_stock topology declarations.

    Returns
    -------
    SFD
    """
    analyst_topology = _extract_analyst_topology(sfd_variables_config)

    # Filter out constant / near-constant time series
    filtered_ts = {}
    for name, ts in var_ts.items():
        if len(ts) > 0 and np.std(ts) > 1e-10:
            filtered_ts[name] = ts
        elif name in variable_roles and variable_roles[name] == "stock":
            filtered_ts[name] = ts

    # Step 2: Discover CLD
    cld = discover_cld(filtered_ts, correlation_threshold, max_lag)

    # Step 3: Label CLD
    S, F, A = label_cld(cld, variable_roles)

    # Include config variables not in CLD
    cld_vars = set(cld.variables)
    for name, role in variable_roles.items():
        if name not in cld_vars and name in var_ts:
            cld.variables.append(name)
            if role == "stock":
                S.add(name)
            elif role == "flow":
                F.add(name)
            else:
                A.add(name)

    # Step 4: SD constraint check
    S, F, relations, I_set, M_set = sd_constraint_check(
        cld, S, F, A, analyst_topology
    )

    # Step 5: Build SFD
    sfd = build_sfd_from_labeled_cld(
        org, S, F, A, M_set, I_set, relations, var_ts, analyst_topology
    )

    # Step 5b: Prune dependencies
    prune_dependencies(sfd)

    # Step 5b2: Remove backward-pointing dependencies in the pipeline
    prune_backward_dependencies(sfd)

    # Step 5c: Ensure every internal flow has a source-stock dependency
    ensure_source_stock_dependencies(sfd, var_ts)

    # Step 6: Fit equations with R² validation
    fit_equations(sfd, var_ts)

    # Step 6b: Set stock accumulation equations (Definition 5)
    fit_stock_equations(sfd)

    return sfd


def discover_sfd(
    agg_df: pd.DataFrame,
    org: str,
    scopes_config: Dict[str, List[str]],
    correlation_threshold: float = 0.3,
    max_lag: int = 3,
) -> SFD:
    """Legacy entry point: discover an SFD from scope-based aggregation."""
    org_df = agg_df[agg_df["org"] == org].copy()
    scopes = sorted(org_df["scope"].unique())

    var_ts = {}
    variable_roles = {}

    for scope in scopes:
        scope_df = org_df[org_df["scope"] == scope]
        for metric in ["wip", "throughput", "arrival"]:
            name = f"{scope}_{metric}".lower().replace(" ", "_")
            series = (
                scope_df[scope_df["metric"] == metric]
                .sort_values("time_window_start")["value"]
                .values.astype(float)
            )
            if len(series) > 0:
                var_ts[name] = series
                if metric == "wip":
                    variable_roles[name] = "stock"
                else:
                    variable_roles[name] = "flow"

    return discover_sfd_from_variables(
        var_ts=var_ts,
        variable_roles=variable_roles,
        org=org,
        correlation_threshold=correlation_threshold,
        max_lag=max_lag,
    )


