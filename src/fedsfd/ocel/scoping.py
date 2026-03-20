"""
Group events into process scopes within each organization.

A scope is a cohesive set of activities that forms a meaningful unit for
aggregation (e.g., "Order Registration" = {register customer order,
create transport document}).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from fedsfd.utils.config import ExperimentConfig


def assign_scopes(
    org_events: Dict[str, pd.DataFrame],
    config: ExperimentConfig,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Assign events to scopes within each organization.

    Parameters
    ----------
    org_events : dict of {org_name: pd.DataFrame}
        Per-org events (must have 'activity' column).
    config : ExperimentConfig
        Must contain scopes mapping.

    Returns
    -------
    dict of {org_name: {scope_name: pd.DataFrame}}
        Events grouped by scope within each org.

    Raises
    ------
    ValueError
        If an activity in an org is not covered by any scope.
    """
    result = {}

    for org_name, ev_df in org_events.items():
        if org_name not in config.scopes:
            raise ValueError(f"No scopes defined for org '{org_name}'")

        scope_map = config.scopes[org_name]
        result[org_name] = {}

        for scope_name, activities in scope_map.items():
            mask = ev_df["activity"].isin(activities)
            scope_df = ev_df[mask].copy().reset_index(drop=True)
            result[org_name][scope_name] = scope_df

    return result


def get_scope_event_ids(
    scoped_events: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, Dict[str, List[str]]]:
    """Extract event IDs per scope.

    Parameters
    ----------
    scoped_events : dict of {org: {scope: DataFrame}}

    Returns
    -------
    dict of {org: {scope: [event_ids]}}
    """
    result = {}
    for org_name, scope_map in scoped_events.items():
        result[org_name] = {}
        for scope_name, df in scope_map.items():
            result[org_name][scope_name] = df["event_id"].tolist()
    return result


def print_scope_summary(
    scoped_events: Dict[str, Dict[str, pd.DataFrame]],
) -> None:
    """Print a summary of scoped events."""
    print("=" * 60)
    print("Scope Summary")
    print("=" * 60)
    for org_name, scope_map in scoped_events.items():
        print(f"\n{org_name}:")
        for scope_name, df in scope_map.items():
            acts = sorted(df["activity"].unique())
            print(f"  {scope_name:30s} {len(df):>6d} events  [{', '.join(acts)}]")