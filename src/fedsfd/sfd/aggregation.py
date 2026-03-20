"""
Temporal aggregation for SFD variable time series.

Two modes:

1. **Analyst-defined variables** (new, preferred): computes time series
   for each variable defined in the YAML config's ``sfd_variables``
   section.  Supports backlog (stock), rate (flow), ratio (auxiliary),
   and constant computation types.

2. **Legacy scope-based** (backward compatible): computes generic
   WIP / throughput / arrival per (organization, scope) pair.

Both modes share the same time-window generation and parse_time_delta.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_naive_dt64(ts):
    """Convert a (possibly tz-aware) Timestamp to a tz-naive np.datetime64."""
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        return np.datetime64(ts.tz_localize(None))
    return np.datetime64(ts)


def _strip_tz_array(arr):
    """Strip timezone from a datetime array to avoid np.datetime64 warnings."""
    if hasattr(arr, 'dtype') and hasattr(arr.dtype, 'tz'):
        return arr.astype('datetime64[ns]')
    return arr


# =========================================================================
# Shared utilities
# =========================================================================

def parse_time_delta(delta_str: str) -> pd.Timedelta:
    """Parse a time window string like '3d', '1d', '12h' into a Timedelta."""
    return pd.Timedelta(delta_str)


def generate_time_windows(
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    delta: pd.Timedelta,
) -> List[pd.Timestamp]:
    """Generate a list of time window start points from t_min to t_max."""
    windows = []
    t = t_min
    while t < t_max:
        windows.append(t)
        t += delta
    return windows


# Keep the old private name as an alias so nothing breaks internally
_generate_time_windows = generate_time_windows


# =========================================================================
# =====================  NEW: Analyst-defined variables  ===================
# =========================================================================

def compute_backlog_ts(
    events_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    object_type: str,
    entry_activities: List[str],
    exit_activities: List[str],
    windows: List[pd.Timestamp],
    delta: pd.Timedelta,
) -> np.ndarray:
    """Compute a backlog time series: objects that have entered but not exited.

    An object is "in the backlog" during window [w, w+Δt) if:
      - it has had at least one entry_activity event with timestamp < w+Δt
      - it has NOT yet had any exit_activity event with timestamp <= w
        (i.e., the object hasn't exited by the start of the window)

    Parameters
    ----------
    events_df : pd.DataFrame
        All events (columns: event_id, activity, timestamp).
    relations_df : pd.DataFrame
        Event-object relations (columns: event_id, object_id, object_type, …).
    object_type : str
        OCEL object type to track (e.g., "Customer Order", "Container").
    entry_activities : list of str
        Activities that mark an object entering the backlog.
    exit_activities : list of str
        Activities that mark an object leaving the backlog.
        If empty, objects never exit (terminal accumulation).
    windows : list of pd.Timestamp
        Time window start points.
    delta : pd.Timedelta
        Window size.

    Returns
    -------
    np.ndarray
        Backlog count at each time window, shape (len(windows),).
    """
    all_activities = entry_activities + exit_activities
    relevant_events = events_df[events_df["activity"].isin(all_activities)].copy()

    if len(relevant_events) == 0:
        return np.zeros(len(windows))

    type_rels = relations_df[relations_df["object_type"] == object_type]
    merged = relevant_events.merge(
        type_rels[["event_id", "object_id"]],
        on="event_id",
        how="inner",
    )

    if len(merged) == 0:
        return np.zeros(len(windows))

    # Per-object: earliest entry time and earliest exit time
    entry_mask = merged["activity"].isin(entry_activities)
    exit_mask = merged["activity"].isin(exit_activities)

    entry_times = (
        merged[entry_mask]
        .groupby("object_id")["timestamp"]
        .min()
        .rename("entry_time")
    )

    if exit_activities:
        exit_times = (
            merged[exit_mask]
            .groupby("object_id")["timestamp"]
            .min()
            .rename("exit_time")
        )
    else:
        exit_times = pd.Series(dtype="datetime64[ns]", name="exit_time")

    obj_df = pd.DataFrame(entry_times)
    if len(exit_times) > 0:
        obj_df = obj_df.join(exit_times, how="left")
    else:
        obj_df["exit_time"] = pd.NaT

    # Vectorised backlog computation
    values = np.zeros(len(windows))
    entry_arr = obj_df["entry_time"].values
    exit_arr = obj_df["exit_time"].values

    entry_arr = _strip_tz_array(entry_arr)
    exit_arr = _strip_tz_array(exit_arr)

    for i, w_start in enumerate(windows):
        w_end = w_start + delta
        entered = entry_arr < _to_naive_dt64(w_end)
        not_exited = np.isnat(exit_arr) | (exit_arr > _to_naive_dt64(w_start))
        values[i] = np.sum(entered & not_exited)

    return values


def compute_rate_ts(
    events_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    activities: List[str],
    windows: List[pd.Timestamp],
    delta: pd.Timedelta,
    object_type: Optional[str] = None,
) -> np.ndarray:
    """Compute a rate time series: count of events (or objects) per window.

    Parameters
    ----------
    events_df : pd.DataFrame
        All events.
    relations_df : pd.DataFrame
        Event-object relations.
    activities : list of str
        Activities to count.
    windows : list of pd.Timestamp
        Time window starts.
    delta : pd.Timedelta
        Window size.
    object_type : str or None
        If given, count events involving this object type (via relations).
        Otherwise count raw events.

    Returns
    -------
    np.ndarray
        Count per window, shape (len(windows),).
    """
    relevant = events_df[events_df["activity"].isin(activities)].copy()

    if len(relevant) == 0:
        return np.zeros(len(windows))

    if object_type is not None:
        type_rels = relations_df[relations_df["object_type"] == object_type]
        merged = relevant.merge(
            type_rels[["event_id", "object_id"]],
            on="event_id",
            how="inner",
        )
        if len(merged) == 0:
            return np.zeros(len(windows))
        timestamps = merged["timestamp"].values
    else:
        timestamps = relevant["timestamp"].values

    ts_arr = np.array(timestamps, dtype="datetime64[ns]")
    values = np.zeros(len(windows))

    for i, w_start in enumerate(windows):
        w_end = w_start + delta
        mask = (ts_arr >= _to_naive_dt64(w_start)) & (ts_arr < _to_naive_dt64(w_end))
        values[i] = np.sum(mask)

    return values


def compute_mean_duration_ts(
    events_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    object_type: str,
    entry_activities: List[str],
    exit_activities: List[str],
    windows: List[pd.Timestamp],
    delta: pd.Timedelta,
    unit: str = "hours",
) -> np.ndarray:
    """Compute mean case duration per time window.

    For each window [w, w+Δt), find objects whose *exit* event falls in that
    window, compute the elapsed time from their earliest entry event to that
    exit event, and return the mean duration.

    This produces a true SD auxiliary variable: a snapshot measurement of
    processing speed within the window, with no memory across windows.

    Parameters
    ----------
    events_df : pd.DataFrame
        All events (columns: event_id, activity, timestamp).
    relations_df : pd.DataFrame
        Event-object relations (columns: event_id, object_id, object_type, …).
    object_type : str
        OCEL object type to track (e.g., "Customer Order", "Container").
    entry_activities : list of str
        Activities marking the start of the phase being measured.
    exit_activities : list of str
        Activities marking the end of the phase being measured.
    windows : list of pd.Timestamp
        Time window start points.
    delta : pd.Timedelta
        Window size.
    unit : str
        Time unit for the result: "hours" (default), "days", or "seconds".

    Returns
    -------
    np.ndarray
        Mean duration per window in the chosen unit, shape (len(windows),).
        Windows with no completing objects get value 0.0.
    """
    all_activities = entry_activities + exit_activities
    relevant_events = events_df[events_df["activity"].isin(all_activities)].copy()

    if len(relevant_events) == 0:
        return np.zeros(len(windows))

    type_rels = relations_df[relations_df["object_type"] == object_type]
    merged = relevant_events.merge(
        type_rels[["event_id", "object_id"]],
        on="event_id",
        how="inner",
    )

    if len(merged) == 0:
        return np.zeros(len(windows))

    entry_mask = merged["activity"].isin(entry_activities)
    exit_mask = merged["activity"].isin(exit_activities)

    # Per-object: earliest entry time
    entry_times = (
        merged[entry_mask]
        .groupby("object_id")["timestamp"]
        .min()
        .rename("entry_time")
    )
    # Per-object: earliest exit time
    exit_times = (
        merged[exit_mask]
        .groupby("object_id")["timestamp"]
        .min()
        .rename("exit_time")
    )

    obj_df = pd.DataFrame(entry_times).join(exit_times, how="inner")
    if len(obj_df) == 0:
        return np.zeros(len(windows))

    # Duration in the requested unit
    duration_td = obj_df["exit_time"] - obj_df["entry_time"]
    if unit == "days":
        divisor = pd.Timedelta("1D")
    elif unit == "seconds":
        divisor = pd.Timedelta("1s")
    else:  # hours
        divisor = pd.Timedelta("1h")
    obj_df["duration"] = duration_td / divisor

    # Clip negative durations (shouldn't happen, but be safe)
    obj_df["duration"] = obj_df["duration"].clip(lower=0.0)

    exit_arr = _strip_tz_array(obj_df["exit_time"].values)
    dur_arr = obj_df["duration"].values

    values = np.zeros(len(windows))
    for i, w_start in enumerate(windows):
        w_end = w_start + delta
        mask = (exit_arr >= _to_naive_dt64(w_start)) & (exit_arr < _to_naive_dt64(w_end))
        if np.any(mask):
            values[i] = float(np.mean(dur_arr[mask]))

    return values


def compute_workload_per_object_ts(
    events_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    object_type: str,
    activities: List[str],
    windows: List[pd.Timestamp],
    delta: pd.Timedelta,
) -> np.ndarray:
    """Compute mean event count per active object per time window.

    For each window [w, w+Δt), counts the total events involving the
    specified object type and activities, then divides by the number of
    unique objects active in that window.  The result is the average
    workload (jobs) per resource unit.

    This is a true SD auxiliary: a snapshot of resource utilisation per
    window, computed directly from event data.  For resource-like objects
    with a small, near-constant active population (e.g., 6 trucks, 3
    forklifts), the raw active-object count has little variance.  But the
    workload per object varies with demand and captures capacity pressure:
    high workload → resources are saturated → flow rates are constrained.

    Parameters
    ----------
    events_df : pd.DataFrame
        All events (columns: event_id, activity, timestamp).
    relations_df : pd.DataFrame
        Event-object relations (columns: event_id, object_id, object_type, …).
    object_type : str
        OCEL object type to track (e.g., "Truck", "Forklift").
    activities : list of str
        Activities to consider.
    windows : list of pd.Timestamp
        Time window start points.
    delta : pd.Timedelta
        Window size.

    Returns
    -------
    np.ndarray
        Mean events per active object per window, shape (len(windows),).
        Windows with no events get value 0.0.
    """
    relevant_events = events_df[events_df["activity"].isin(activities)].copy()

    if len(relevant_events) == 0:
        return np.zeros(len(windows))

    type_rels = relations_df[relations_df["object_type"] == object_type]
    merged = relevant_events.merge(
        type_rels[["event_id", "object_id"]],
        on="event_id",
        how="inner",
    )

    if len(merged) == 0:
        return np.zeros(len(windows))

    ts_arr = merged["timestamp"].values.astype("datetime64[ns]")
    obj_arr = merged["object_id"].values

    values = np.zeros(len(windows))
    for i, w_start in enumerate(windows):
        w_end = w_start + delta
        mask = (ts_arr >= _to_naive_dt64(w_start)) & (ts_arr < _to_naive_dt64(w_end))
        if np.any(mask):
            n_events = int(np.sum(mask))
            n_objects = len(set(obj_arr[mask]))
            values[i] = float(n_events) / float(n_objects)

    return values


def compute_sfd_variables(
    events_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    sfd_variables_config: Dict[str, Dict[str, Any]],
    windows: List[pd.Timestamp],
    delta: pd.Timedelta,
    org_name: str,
) -> Dict[str, np.ndarray]:
    """Compute time series for all analyst-defined SFD variables of one org.

    Two-pass approach:
      1. Compute all "backlog", "rate", and "constant" variables
      2. Compute "ratio" variables (which depend on pass-1 results)

    Returns
    -------
    dict of {variable_name: np.ndarray}
    """
    n = len(windows)
    result: Dict[str, np.ndarray] = {}

    # Pass 1: primitive variables
    for var_name, var_def in sfd_variables_config.items():
        compute = var_def.get("compute", {})
        ctype = compute.get("type", "")

        if ctype == "backlog":
            result[var_name] = compute_backlog_ts(
                events_df=events_df,
                relations_df=relations_df,
                object_type=compute["object_type"],
                entry_activities=compute["entry_activities"],
                exit_activities=compute.get("exit_activities", []),
                windows=windows,
                delta=delta,
            )

        elif ctype == "rate":
            result[var_name] = compute_rate_ts(
                events_df=events_df,
                relations_df=relations_df,
                activities=compute["activities"],
                windows=windows,
                delta=delta,
                object_type=compute.get("object_type"),
            )

        elif ctype == "mean_duration":
            result[var_name] = compute_mean_duration_ts(
                events_df=events_df,
                relations_df=relations_df,
                object_type=compute["object_type"],
                entry_activities=compute["entry_activities"],
                exit_activities=compute["exit_activities"],
                windows=windows,
                delta=delta,
                unit=compute.get("unit", "hours"),
            )

        elif ctype == "workload_per_object":
            result[var_name] = compute_workload_per_object_ts(
                events_df=events_df,
                relations_df=relations_df,
                object_type=compute["object_type"],
                activities=compute["activities"],
                windows=windows,
                delta=delta,
            )

        elif ctype == "constant":
            result[var_name] = np.full(n, float(compute["value"]))

        elif ctype == "ratio":
            pass  # handled in pass 2

        else:
            print(f"  Warning: unknown compute type '{ctype}' for {org_name}/{var_name}")
            result[var_name] = np.zeros(n)

    # Pass 2: ratio variables
    for var_name, var_def in sfd_variables_config.items():
        compute = var_def.get("compute", {})
        if compute.get("type") != "ratio":
            continue

        num_name = compute["numerator"]
        den = compute["denominator"]

        numerator = result.get(num_name, np.zeros(n))

        if isinstance(den, (int, float)):
            denominator = np.full(n, float(den))
        else:
            denominator = result.get(den, np.ones(n))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denominator != 0, numerator / denominator, 0.0)
        result[var_name] = ratio

    return result


def compute_all_sfd_variables(
    events_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    sfd_config: Dict[str, Dict[str, Dict[str, Any]]],
    windows: List[pd.Timestamp],
    delta: pd.Timedelta,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute time series for all variables across all organizations.

    Returns
    -------
    dict of {org_name: {variable_name: np.ndarray}}
    """
    all_ts = {}
    for org_name, org_vars in sfd_config.items():
        print(f"  Computing variables for {org_name} ({len(org_vars)} variables)...")
        all_ts[org_name] = compute_sfd_variables(
            events_df=events_df,
            relations_df=relations_df,
            sfd_variables_config=org_vars,
            windows=windows,
            delta=delta,
            org_name=org_name,
        )
    return all_ts


def variables_to_long_df(
    all_ts: Dict[str, Dict[str, np.ndarray]],
    sfd_config: Dict[str, Dict[str, Dict[str, Any]]],
    windows: List[pd.Timestamp],
) -> pd.DataFrame:
    """Convert variable time series to a long-form DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: org, variable, role, time_window_start, value
    """
    rows = []
    for org_name, org_ts in all_ts.items():
        org_vars = sfd_config.get(org_name, {})
        for var_name, ts in org_ts.items():
            role = org_vars.get(var_name, {}).get("role", "unknown")
            for i, w in enumerate(windows):
                rows.append({
                    "org": org_name,
                    "variable": var_name,
                    "role": role,
                    "time_window_start": w,
                    "value": float(ts[i]),
                })
    return pd.DataFrame(rows)


def print_variable_summary(
    all_ts: Dict[str, Dict[str, np.ndarray]],
    sfd_config: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Print summary statistics for all computed variables."""
    print("=" * 70)
    print("SFD Variable Summary")
    print("=" * 70)
    for org_name, org_ts in all_ts.items():
        org_vars = sfd_config.get(org_name, {})
        print(f"\n{org_name}:")
        for var_name, ts in sorted(org_ts.items()):
            role = org_vars.get(var_name, {}).get("role", "?")
            unit = org_vars.get(var_name, {}).get("unit", "")
            print(
                f"  {var_name:35s} [{role:5s}] "
                f"mean={np.mean(ts):8.2f}  std={np.std(ts):7.2f}  "
                f"min={np.min(ts):6.0f}  max={np.max(ts):6.0f}  {unit}"
            )


# =========================================================================
# ===================  LEGACY: Scope-based aggregation  ====================
# =========================================================================


def compute_object_scope_intervals(
    scope_events: pd.DataFrame,
    relations_df: pd.DataFrame,
    scope_activities: List[str],
) -> pd.DataFrame:
    """Compute per-object entry/exit times within a scope.

    An object enters a scope at its first event in the scope, and exits
    at its last event in the scope.

    Parameters
    ----------
    scope_events : pd.DataFrame
        Events in this scope (columns: event_id, activity, timestamp).
    relations_df : pd.DataFrame
        Event-object relations (columns: event_id, object_id, …).
    scope_activities : list of str
        Activities belonging to this scope.

    Returns
    -------
    pd.DataFrame
        Columns: object_id, entry_time, exit_time
    """
    if len(scope_events) == 0:
        return pd.DataFrame(columns=["object_id", "entry_time", "exit_time"])

    event_ids = set(scope_events["event_id"])
    scope_rels = relations_df[relations_df["event_id"].isin(event_ids)].copy()

    if len(scope_rels) == 0:
        return pd.DataFrame(columns=["object_id", "entry_time", "exit_time"])

    merged = scope_rels.merge(
        scope_events[["event_id", "timestamp"]],
        on="event_id",
        how="left",
        suffixes=("_rel", ""),
    )
    ts_col = "timestamp" if "timestamp" in merged.columns else "timestamp_rel"

    obj_intervals = (
        merged.groupby("object_id")[ts_col]
        .agg(entry_time="min", exit_time="max")
        .reset_index()
    )

    return obj_intervals


def aggregate_scope(
    scope_events: pd.DataFrame,
    relations_df: pd.DataFrame,
    scope_activities: List[str],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    delta: pd.Timedelta,
) -> pd.DataFrame:
    """Compute WIP, throughput, and arrival time series for one scope.

    Parameters
    ----------
    scope_events : pd.DataFrame
        Events in this scope.
    relations_df : pd.DataFrame
        Event-object relations.
    scope_activities : list of str
        Activities in this scope.
    t_min, t_max : pd.Timestamp
        Overall time horizon.
    delta : pd.Timedelta
        Time window size.

    Returns
    -------
    pd.DataFrame
        Columns: time_window_start, wip, throughput, arrival
    """
    windows = _generate_time_windows(t_min, t_max, delta)
    obj_intervals = compute_object_scope_intervals(
        scope_events, relations_df, scope_activities
    )

    wip_values = []
    throughput_values = []
    arrival_values = []

    for w_start in windows:
        w_end = w_start + delta

        if len(obj_intervals) == 0:
            wip_values.append(0)
            throughput_values.append(0)
            arrival_values.append(0)
            continue

        wip_mask = (
            (obj_intervals["entry_time"] < w_end)
            & (obj_intervals["exit_time"] >= w_start)
        )
        wip_values.append(int(wip_mask.sum()))

        tp_mask = (
            (obj_intervals["exit_time"] >= w_start)
            & (obj_intervals["exit_time"] < w_end)
        )
        throughput_values.append(int(tp_mask.sum()))

        arr_mask = (
            (obj_intervals["entry_time"] >= w_start)
            & (obj_intervals["entry_time"] < w_end)
        )
        arrival_values.append(int(arr_mask.sum()))

    return pd.DataFrame({
        "time_window_start": windows,
        "wip": wip_values,
        "throughput": throughput_values,
        "arrival": arrival_values,
    })


def aggregate_all_scopes(
    scoped_events: Dict[str, Dict[str, pd.DataFrame]],
    relations_df: pd.DataFrame,
    scopes_config: Dict[str, Dict[str, List[str]]],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    delta: pd.Timedelta,
) -> pd.DataFrame:
    """Compute aggregated time series for all scopes across all orgs.

    Parameters
    ----------
    scoped_events : dict of {org: {scope: DataFrame}}
        Scoped event DataFrames.
    relations_df : pd.DataFrame
        Full event-object relations table.
    scopes_config : dict of {org: {scope: [activities]}}
        Scope activity definitions.
    t_min, t_max : pd.Timestamp
        Overall time horizon.
    delta : pd.Timedelta
        Time window size.

    Returns
    -------
    pd.DataFrame
        Long-form table: org, scope, metric, time_window_start, value
    """
    rows = []

    for org_name, scope_map in scoped_events.items():
        org_event_ids = set()
        for scope_df in scope_map.values():
            org_event_ids.update(scope_df["event_id"])
        org_relations = relations_df[relations_df["event_id"].isin(org_event_ids)]

        for scope_name, scope_df in scope_map.items():
            scope_activities = scopes_config[org_name][scope_name]
            ts_df = aggregate_scope(
                scope_df, org_relations, scope_activities,
                t_min, t_max, delta,
            )

            for metric in ["wip", "throughput", "arrival"]:
                for _, row in ts_df.iterrows():
                    rows.append({
                        "org": org_name,
                        "scope": scope_name,
                        "metric": metric,
                        "time_window_start": row["time_window_start"],
                        "value": row[metric],
                    })

    return pd.DataFrame(rows)


def get_time_series(
    agg_df: pd.DataFrame,
    org: str,
    scope: str,
    metric: str,
) -> pd.DataFrame:
    """Extract a single time series from the aggregated DataFrame.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Output of aggregate_all_scopes.
    org, scope, metric : str
        Filter criteria.

    Returns
    -------
    pd.DataFrame
        Columns: time_window_start, value
    """
    mask = (
        (agg_df["org"] == org)
        & (agg_df["scope"] == scope)
        & (agg_df["metric"] == metric)
    )
    return (
        agg_df[mask][["time_window_start", "value"]]
        .sort_values("time_window_start")
        .reset_index(drop=True)
    )


def print_aggregation_summary(agg_df: pd.DataFrame) -> None:
    """Print summary statistics of aggregated time series."""
    print("=" * 60)
    print("Aggregation Summary")
    print("=" * 60)

    n_windows = agg_df["time_window_start"].nunique()
    print(f"Time windows: {n_windows}")
    print()

    for org in sorted(agg_df["org"].unique()):
        print(f"{org}:")
        org_df = agg_df[agg_df["org"] == org]
        for scope in sorted(org_df["scope"].unique()):
            scope_df = org_df[org_df["scope"] == scope]
            for metric in ["wip", "throughput", "arrival"]:
                m_df = scope_df[scope_df["metric"] == metric]
                if len(m_df) > 0:
                    vals = m_df["value"]
                    print(
                        f"  {scope:25s} {metric:12s} "
                        f"mean={vals.mean():7.1f}  std={vals.std():7.1f}  "
                        f"min={vals.min():5.0f}  max={vals.max():5.0f}"
                    )
        print()