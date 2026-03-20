"""
Actor-based federated simulation.

Each organization runs as an independent process, and the MPC platform
runs as a separate reactive process.  Communication happens via message
queues, mirroring the protocol described in the paper:

    At each simulation time step t, each organization:
      1. secretly shares its boundary stock values to the MPC platform,
      2. receives inter-organizational inflows from the MPC platform,
      3. computes internal stock evolution locally.

    Organizations control when simulation ends: after |T^sim| steps
    they simply stop sharing, and the MPC platform ceases as well.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fedsfd.federation.boundary import BoundaryEquation
from fedsfd.federation.residual import ResidualModel
from fedsfd.mpc.interface import MPCBackend
from fedsfd.sfd.model import SFD


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class ShareStock:
    """Org -> MPC: share a boundary stock value for a given timestep."""
    org: str
    stock_name: str
    value: float
    timestep: int


@dataclass
class FlowResult:
    """MPC -> Org: computed boundary flow contribution."""
    flow_name: str
    rate: float
    timestep: int
    # Which boundary equation produced this (for trajectory tracking)
    boundary_key: str


@dataclass
class OrgDone:
    """Org -> MPC: this organization has finished its simulation."""
    org: str


def _compute_training_bounds(
    var_ts_train: Dict[str, Dict[str, np.ndarray]],
    margin: float = 3.0,
) -> Dict[str, Dict[str, tuple]]:
    """Compute (min, max) bounds from training data for clamping.

    Uses mean ± margin * std to allow reasonable extrapolation while
    preventing runaway trajectories. Floor is always 0.

    Parameters
    ----------
    var_ts_train : dict of {org_name: {var_name: np.ndarray}}
    margin : float
        Number of standard deviations beyond training range to allow.

    Returns
    -------
    dict of {org_name: {var_name: (lower, upper)}}
    """
    bounds = {}
    for org_name, org_ts in var_ts_train.items():
        bounds[org_name] = {}
        for var_name, ts in org_ts.items():
            if len(ts) == 0:
                continue
            mean = float(np.mean(ts))
            std = float(np.std(ts))
            upper = max(float(np.max(ts)), mean + margin * std)
            bounds[org_name][var_name] = (0.0, upper)
    return bounds


# ---------------------------------------------------------------------------
# Organization process
# ---------------------------------------------------------------------------

def _org_process(
    org_name: str,
    sfd: SFD,
    boundary_stocks_to_share: List[str],
    expected_inflows: Dict[str, int],
    residual_map: Dict[str, ResidualModel],
    boundary_by_inflow: Dict[str, List[BoundaryEquation]],
    what_if_modifications: Dict[str, float],
    n_steps: int,
    dt: float,
    to_mpc: mp.Queue,
    from_mpc: mp.Queue,
    result_queue: mp.Queue,
    timing_queue: Optional[mp.Queue] = None,
) -> None:
    """Run one organization's simulation loop.

    Parameters
    ----------
    org_name : str
    sfd : SFD
    boundary_stocks_to_share : list of stock names to send to MPC each step
    expected_inflows : {flow_name: n_messages_expected_per_step}
    residual_map : {flow_name: ResidualModel}
    boundary_by_inflow : {flow_name: [BoundaryEquation, ...]}
    what_if_modifications : {flow_name: multiplier}
    n_steps : int
    dt : float
    to_mpc : Queue  -- outbox to MPC platform
    from_mpc : Queue  -- inbox from MPC platform
    result_queue : Queue  -- final trajectories sent back to coordinator
    """
    # --- Initialize local state ---
    state = {}
    for stock in sfd.stocks:
        state[stock.name] = stock.initial_value

    history = {}
    for stock in sfd.stocks:
        history[stock.name] = [stock.initial_value]
    for flow in sfd.flows:
        history[flow.name] = []

    stock_traj = {}
    flow_traj = {}
    for stock in sfd.stocks:
        stock_traj[f"{org_name.lower()}_{stock.name}"] = [stock.initial_value]
    for flow in sfd.flows:
        flow_traj[f"{org_name.lower()}_{flow.name}"] = []

    # Boundary flow trajectories this org contributes to (as receiver)
    boundary_traj = {}

    # Timing accumulators
    local_compute_sec = 0.0
    waiting_sec = 0.0

    # --- Simulation loop ---
    for t in range(n_steps):

        # Step 1: Share boundary stock values with MPC platform
        for stock_name in boundary_stocks_to_share:
            hist = history.get(stock_name, [])
            value = hist[-1] if hist else 0.0
            to_mpc.put(ShareStock(
                org=org_name,
                stock_name=stock_name,
                value=value,
                timestep=t,
            ))

        # Step 2: Receive boundary inflow rates from MPC platform
        boundary_flows = {}  # flow_name -> accumulated rate
        n_expected = sum(expected_inflows.values())
        t_wait_start = time.perf_counter()
        for _ in range(n_expected):
            msg: FlowResult = from_mpc.get()
            boundary_flows[msg.flow_name] = (
                boundary_flows.get(msg.flow_name, 0.0) + msg.rate
            )
            boundary_traj.setdefault(msg.boundary_key, []).append(msg.rate)
        waiting_sec += time.perf_counter() - t_wait_start

        # Step 3: Compute internal flows locally and update stocks
        t_local_start = time.perf_counter()

        # Build eval state
        eval_state = dict(state)
        for var_name, hist_vals in history.items():
            for lag_i in range(1, len(hist_vals) + 1):
                if lag_i <= len(hist_vals):
                    idx = len(hist_vals) - lag_i
                    eval_state[f"{var_name}__lag{lag_i}"] = hist_vals[idx]
            if hist_vals:
                eval_state[var_name] = hist_vals[-1]

        # Compute auxiliaries
        for aux in sfd.auxiliaries:
            if aux.equation is not None:
                aux_val = aux.equation(eval_state, aux.equation_params, t)
            elif aux.time_series is not None and t < len(aux.time_series):
                aux_val = float(aux.time_series[t])
            else:
                aux_val = 0.0
            eval_state[aux.name] = aux_val
            history.setdefault(aux.name, []).append(aux_val)
            for lag_i in range(1, len(history[aux.name]) + 1):
                idx = len(history[aux.name]) - lag_i
                eval_state[f"{aux.name}__lag{lag_i}"] = history[aux.name][idx]

        # Compute flow rates
        org_flow_rates = {}
        for flow in sfd.flows:
            has_boundary = flow.name in boundary_by_inflow

            if has_boundary:
                rate = boundary_flows.get(flow.name, 0.0)
                residual = residual_map.get(flow.name)
                if residual:
                    rate += residual.predict(t)
                if flow.name in what_if_modifications:
                    rate *= what_if_modifications[flow.name]
                elif f"{org_name.lower()}_{flow.name}" in what_if_modifications:
                    rate *= what_if_modifications[f"{org_name.lower()}_{flow.name}"]
            elif flow.name in what_if_modifications:
                rate = flow.compute(eval_state, t) * what_if_modifications[flow.name]
            elif f"{org_name.lower()}_{flow.name}" in what_if_modifications:
                rate = flow.compute(eval_state, t) * what_if_modifications[
                    f"{org_name.lower()}_{flow.name}"
                ]
            else:
                rate = flow.compute(eval_state, t)

            rate = max(rate, 0.0)
            org_flow_rates[flow.name] = rate
            history.setdefault(flow.name, []).append(rate)

            traj_key = f"{org_name.lower()}_{flow.name}"
            if traj_key in flow_traj:
                flow_traj[traj_key].append(rate)

        # Update stocks
        for stock in sfd.stocks:
            if stock.is_cloud:
                continue
            inflows = sfd.get_stock_inflows(stock)
            outflows = sfd.get_stock_outflows(stock)
            net = 0.0
            for f in inflows:
                net += org_flow_rates.get(f.name, 0.0)
            for f in outflows:
                net -= org_flow_rates.get(f.name, 0.0)
            new_val = max(state[stock.name] + dt * net, 0.0)
            state[stock.name] = new_val
            history[stock.name].append(new_val)
            traj_key = f"{org_name.lower()}_{stock.name}"
            stock_traj[traj_key].append(new_val)

        local_compute_sec += time.perf_counter() - t_local_start

    # --- Simulation complete: stop sharing ---
    to_mpc.put(OrgDone(org=org_name))

    # Send trajectories back
    all_traj = {}
    all_traj.update(stock_traj)
    all_traj.update(flow_traj)
    all_traj.update(boundary_traj)
    result_queue.put((org_name, all_traj))

    if timing_queue is not None:
        timing_queue.put({
            "actor": f"org_{org_name}",
            "local_compute_sec": local_compute_sec,
            "waiting_sec": waiting_sec,
        })


# ---------------------------------------------------------------------------
# MPC platform process
# ---------------------------------------------------------------------------

def _mpc_platform_process(
    boundary_equations: List[BoundaryEquation],
    org_party_map: Dict[str, int],
    mpc_backend: MPCBackend,
    inbox: mp.Queue,
    org_outboxes: Dict[str, mp.Queue],
    n_orgs: int,
    source_stock_lags: Dict[int, int],
    timing_queue: Optional[mp.Queue] = None,
) -> None:
    """Reactive MPC platform: listen for stock shares, compute, respond.

    The platform sits in an "alert state." When it receives a stock
    share for a boundary equation, it computes the flow and sends the
    result to the receiving organization. When all organizations signal
    they are done, the platform shuts down.
    """
    done_orgs = set()
    mpc_compute_sec = 0.0

    # For each boundary equation, we need the source stock value at
    # each timestep. Since each equation depends on exactly one source
    # stock (possibly lagged), we buffer received values.
    # Buffer: (source_org, source_stock) -> {timestep: value}
    stock_buffer: Dict[tuple, Dict[int, float]] = {}

    # Track which equations have been processed at each timestep
    # eq_index -> set of processed timesteps
    processed: Dict[int, set] = {i: set() for i in range(len(boundary_equations))}

    while len(done_orgs) < n_orgs:
        msg = inbox.get()

        if isinstance(msg, OrgDone):
            done_orgs.add(msg.org)
            continue

        if isinstance(msg, ShareStock):
            buf_key = (msg.org, msg.stock_name)
            stock_buffer.setdefault(buf_key, {})[msg.timestep] = msg.value

            # Check which boundary equations can now fire
            for eq_idx, eq in enumerate(boundary_equations):
                if msg.timestep in processed[eq_idx]:
                    continue

                source_org = eq.match.outflow_org
                source_stock = eq.source_stock_name
                lag = eq.lag

                # Determine the timestep we need from the source stock
                needed_t = msg.timestep
                buf = stock_buffer.get((source_org, source_stock), {})

                if needed_t not in buf:
                    continue

                # We can also handle lag via historical buffer
                if lag > 0 and (needed_t - lag) >= 0:
                    lagged_val = buf.get(needed_t - lag)
                    if lagged_val is None:
                        # Use current if lagged not available
                        source_val = buf[needed_t]
                    else:
                        source_val = lagged_val
                else:
                    source_val = buf[needed_t]

                # Compute via MPC backend
                source_party = org_party_map[source_org]
                sink_party = org_party_map[eq.match.inflow_org]

                t_mpc_start = time.perf_counter()
                if eq.is_persisted:
                    result = mpc_backend.secure_boundary_flow_from_persistence(
                        stock_values={source_party: source_val},
                        equation_params={
                            "source_party": source_party,
                            "sink_party": sink_party,
                            "eq_id": eq.eq_id,
                        },
                    )
                else:
                    result = mpc_backend.secure_boundary_flow(
                        stock_values={source_party: source_val},
                        equation_params={
                            "source_party": source_party,
                            "sink_party": sink_party,
                            "intercept": eq.intercept,
                            "slope": eq.slope,
                        },
                    )
                mpc_compute_sec += time.perf_counter() - t_mpc_start

                flow_rate = result.get(sink_party, 0.0)
                receiving_org = eq.match.inflow_org
                boundary_key = (
                    f"boundary_{eq.match.outflow_org}_{eq.match.outflow_name}"
                    f"_to_{eq.match.inflow_org}"
                )

                org_outboxes[receiving_org].put(FlowResult(
                    flow_name=eq.match.inflow_name,
                    rate=flow_rate,
                    timestep=msg.timestep,
                    boundary_key=boundary_key,
                ))

                processed[eq_idx].add(msg.timestep)

    if timing_queue is not None:
        timing_queue.put({
            "actor": "mpc_platform",
            "mpc_compute_sec": mpc_compute_sec,
        })


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def federated_simulate_actors(
    local_sfds: Dict[str, SFD],
    boundary_equations: List[BoundaryEquation],
    residuals: List[ResidualModel],
    mpc_backend: MPCBackend,
    org_party_map: Dict[str, int],
    n_steps: int,
    dt: float = 1.0,
    what_if_modifications: Optional[Dict[str, float]] = None,
    report_timing: bool = False,
) -> Dict[str, np.ndarray]:
    """Run federated simulation with each org as an independent process.

    Each organization runs as an independent process.
    When ``report_timing`` is True, returns a
    ``(trajectories, timing_info)`` tuple instead of just trajectories.
    """
    if what_if_modifications is None:
        what_if_modifications = {}

    # --- Pre-compute per-org boundary metadata ---
    # Which stocks each org must share (source stocks of boundary equations)
    org_boundary_stocks: Dict[str, set] = {org: set() for org in local_sfds}
    # Which inflows each org expects from MPC, and how many messages per step
    org_expected_inflows: Dict[str, Dict[str, int]] = {org: {} for org in local_sfds}
    # Boundary equations indexed by receiving org's inflow
    org_boundary_by_inflow: Dict[str, Dict[str, list]] = {org: {} for org in local_sfds}

    for eq in boundary_equations:
        source_org = eq.match.outflow_org
        source_stock = eq.source_stock_name
        org_boundary_stocks[source_org].add(source_stock)

        recv_org = eq.match.inflow_org
        inflow_name = eq.match.inflow_name
        org_expected_inflows[recv_org][inflow_name] = (
            org_expected_inflows[recv_org].get(inflow_name, 0) + 1
        )
        org_boundary_by_inflow[recv_org].setdefault(inflow_name, []).append(eq)

    # Residuals indexed by (org, flow_name)
    org_residual_map: Dict[str, Dict[str, ResidualModel]] = {
        org: {} for org in local_sfds
    }
    for r in residuals:
        org_residual_map[r.org][r.flow_name] = r

    # Source stock lag info for MPC platform
    source_stock_lags = {i: eq.lag for i, eq in enumerate(boundary_equations)}

    # --- Create communication queues ---
    mpc_inbox = mp.Queue()
    org_inboxes = {org: mp.Queue() for org in local_sfds}
    result_queue = mp.Queue()
    timing_queue = mp.Queue() if report_timing else None

    # --- Spawn MPC platform process ---
    mpc_proc = mp.Process(
        target=_mpc_platform_process,
        args=(
            boundary_equations,
            org_party_map,
            mpc_backend,
            mpc_inbox,
            org_inboxes,
            len(local_sfds),
            source_stock_lags,
            timing_queue,
        ),
        name="MPC-Platform",
    )
    mpc_proc.start()

    # --- Spawn organization processes ---
    org_procs = {}
    for org_name, sfd in local_sfds.items():
        proc = mp.Process(
            target=_org_process,
            args=(
                org_name,
                sfd,
                list(org_boundary_stocks[org_name]),
                org_expected_inflows[org_name],
                org_residual_map[org_name],
                org_boundary_by_inflow[org_name],
                what_if_modifications,
                n_steps,
                dt,
                mpc_inbox,
                org_inboxes[org_name],
                result_queue,
                timing_queue,
            ),
            name=f"Org-{org_name}",
        )
        proc.start()
        org_procs[org_name] = proc

    # --- Collect results ---
    all_trajectories = {}
    for _ in local_sfds:
        org_name, traj = result_queue.get()
        for key, values in traj.items():
            all_trajectories[key] = values

    # --- Wait for processes to finish ---
    for proc in org_procs.values():
        proc.join()
    mpc_proc.join()

    trajectories = {name: np.array(traj) for name, traj in all_trajectories.items()}

    if report_timing and timing_queue is not None:
        timing_info = []
        # Collect: 1 MPC report + n_orgs org reports
        while not timing_queue.empty():
            timing_info.append(timing_queue.get_nowait())
        return trajectories, timing_info

    return trajectories
