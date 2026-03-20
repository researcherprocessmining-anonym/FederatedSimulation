"""
Local SFD simulation using Euler integration.

Simulates forward in time by computing flow rates from their equations,
then updating stock levels:
    s[t+1] = s[t] + Δt * Σ sign(f, s) * f[t]

External flows use their fitted equations (or time series replay as fallback).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from fedsfd.sfd.model import Flow, SFD, Stock


def simulate_sfd(
    sfd: SFD,
    n_steps: int,
    dt: float = 1.0,
    external_inputs: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Simulate an SFD forward in time using Euler integration.

    Parameters
    ----------
    sfd : SFD
        The Stock-Flow Diagram to simulate.
    n_steps : int
        Number of time steps to simulate.
    dt : float
        Time step size (default 1.0, matching the aggregation Δt).
    external_inputs : dict or None
        Optional overrides for external flow time series:
        {flow_name: np.ndarray of shape (n_steps,)}.
        If provided for a flow, this takes precedence over the
        flow's equation.

    Returns
    -------
    dict of {variable_name: np.ndarray}
        Trajectories for all stocks and flows, each of shape (n_steps+1,)
        for stocks and (n_steps,) for flows.
    """
    if external_inputs is None:
        external_inputs = {}

    # Initialize state: stock_name → current_value
    state = {}
    for stock in sfd.stocks:
        state[stock.name] = stock.initial_value

    # Trajectories
    stock_trajectories = {s.name: [s.initial_value] for s in sfd.stocks}
    flow_trajectories = {f.name: [] for f in sfd.flows}

    # Build lookup for lagged values
    # We maintain a history buffer for each variable
    history = {name: [val] for name, val in state.items()}
    for f in sfd.flows:
        history[f.name] = []

    for t in range(n_steps):
        # Build state dict with lagged values for equation evaluation
        eval_state = dict(state)  # current stock values

        # Add lagged stock values
        for s in sfd.stocks:
            for lag in range(1, len(history[s.name]) + 1):
                if lag <= len(history[s.name]):
                    idx = len(history[s.name]) - lag
                    eval_state[f"{s.name}__lag{lag}"] = history[s.name][idx]

        # Add lagged flow values
        for f in sfd.flows:
            if history[f.name]:
                for lag in range(1, len(history[f.name]) + 1):
                    if lag <= len(history[f.name]):
                        idx = len(history[f.name]) - lag
                        eval_state[f"{f.name}__lag{lag}"] = history[f.name][idx]
                # Also store last flow value as current for dependencies
                eval_state[f.name] = history[f.name][-1] if history[f.name] else 0.0

        # --- Compute auxiliary variables ---
        for aux in sfd.auxiliaries:
            if aux.equation is not None:
                aux_val = aux.equation(eval_state, aux.equation_params, t)
            elif aux.time_series is not None and t < len(aux.time_series):
                aux_val = float(aux.time_series[t])
            else:
                aux_val = 0.0
            eval_state[aux.name] = aux_val
            history.setdefault(aux.name, []).append(aux_val)
            # Add lagged auxiliary values
            for lag in range(1, len(history[aux.name]) + 1):
                if lag <= len(history[aux.name]):
                    idx = len(history[aux.name]) - lag
                    eval_state[f"{aux.name}__lag{lag}"] = history[aux.name][idx]

        # --- Compute flow rates ---
        flow_rates = {}
        for flow in sfd.flows:
            if flow.name in external_inputs:
                # Use external input override
                ext = external_inputs[flow.name]
                rate = float(ext[t]) if t < len(ext) else 0.0
            elif flow.equation is not None:
                rate = flow.compute(eval_state, t)
            elif flow.time_series is not None and t < len(flow.time_series):
                rate = float(flow.time_series[t])
            else:
                rate = 0.0

            flow_rates[flow.name] = max(rate, 0.0)  # non-negative rates
            flow_trajectories[flow.name].append(flow_rates[flow.name])
            history[flow.name].append(flow_rates[flow.name])

        # --- Update stocks (Definition 5) ---
        # Add flow rates to eval_state so stock equations can reference them
        eval_state.update(flow_rates)

        for stock in sfd.stocks:
            if stock.is_cloud:
                continue

            if stock.equation is not None:
                # Use the stock's accumulation equation (Definition 5)
                new_value = stock.compute(eval_state, t)
            else:
                # Fallback: manual Euler integration
                inflows = sfd.get_stock_inflows(stock)
                outflows = sfd.get_stock_outflows(stock)

                net_flow = (
                    sum(flow_rates.get(f.name, 0.0) for f in inflows)
                    - sum(flow_rates.get(f.name, 0.0) for f in outflows)
                )

                new_value = state[stock.name] + dt * net_flow
                new_value = max(new_value, 0.0)

            state[stock.name] = new_value
            stock_trajectories[stock.name].append(new_value)
            history[stock.name].append(new_value)

    # Convert to numpy arrays
    result = {}
    for name, traj in stock_trajectories.items():
        result[name] = np.array(traj)
    for name, traj in flow_trajectories.items():
        result[name] = np.array(traj)

    return result


def simulate_with_what_if(
    sfd: SFD,
    n_steps: int,
    dt: float = 1.0,
    what_if_modifications: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Simulate an SFD with what-if modifications to external flows.

    Parameters
    ----------
    sfd : SFD
        The Stock-Flow Diagram.
    n_steps : int
        Number of simulation steps.
    dt : float
        Time step size.
    what_if_modifications : dict or None
        {flow_name: multiplier} — multiply the flow's rate by this factor.
        E.g., {"order_registration_arrival": 1.3} for 30% increase.

    Returns
    -------
    dict of {variable_name: np.ndarray}
        Trajectories with modifications applied.
    """
    if what_if_modifications is None:
        return simulate_sfd(sfd, n_steps, dt)

    # Build external input overrides with multiplied time series
    external_inputs = {}
    for flow_name, multiplier in what_if_modifications.items():
        flow = sfd.get_flow(flow_name)
        if flow is None:
            continue
        if flow.time_series is not None:
            modified_ts = flow.time_series.copy() * multiplier
            external_inputs[flow_name] = modified_ts

    return simulate_sfd(sfd, n_steps, dt, external_inputs=external_inputs)