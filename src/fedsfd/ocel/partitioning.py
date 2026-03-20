"""
Partition an OCEL event log into organization slices by activity type.

Each event is assigned to exactly one organization based on its activity.
Objects may span multiple organizations (e.g., a Container appears in both
Trucking and Terminal events).
"""

from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd

from fedsfd.utils.config import ExperimentConfig


def partition_events(
    events_df: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict[str, pd.DataFrame]:
    """Partition events into per-organization DataFrames.

    Parameters
    ----------
    events_df : pd.DataFrame
        Flat events table with columns: event_id, activity, timestamp.
    config : ExperimentConfig
        Must contain organizations mapping (activity -> org).

    Returns
    -------
    dict of {org_name: pd.DataFrame}
        Each DataFrame has the same columns as events_df, filtered to
        that organization's activities.

    Raises
    ------
    ValueError
        If any event's activity is not mapped to an organization.
    """
    unmapped = set(events_df["activity"].unique()) - set(config.activity_to_org.keys())
    if unmapped:
        raise ValueError(
            f"Activities not mapped to any organization: {unmapped}"
        )

    org_events = {}
    for org_name, activities in config.organizations.items():
        mask = events_df["activity"].isin(activities)
        org_events[org_name] = events_df[mask].copy().reset_index(drop=True)

    return org_events


def partition_relations(
    relations_df: pd.DataFrame,
    org_events: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Partition event-object relations by organization.

    Parameters
    ----------
    relations_df : pd.DataFrame
        Relations table with columns: event_id, object_id, object_type, ...
    org_events : dict of {org_name: pd.DataFrame}
        Per-org events (output of partition_events).

    Returns
    -------
    dict of {org_name: pd.DataFrame}
        Per-org relations, filtered to events in that org.
    """
    org_relations = {}
    for org_name, ev_df in org_events.items():
        event_ids = set(ev_df["event_id"])
        mask = relations_df["event_id"].isin(event_ids)
        org_relations[org_name] = relations_df[mask].copy().reset_index(drop=True)
    return org_relations


def get_org_object_types(
    org_relations: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, int]]:
    """Compute object type distributions per organization.

    Parameters
    ----------
    org_relations : dict of {org_name: pd.DataFrame}
        Per-org relations (must have 'object_id' and 'object_type' columns).

    Returns
    -------
    dict of {org_name: {object_type: count_of_unique_objects}}
    """
    result = {}
    for org_name, rel_df in org_relations.items():
        if len(rel_df) == 0:
            result[org_name] = {}
            continue
        unique_objs = rel_df.drop_duplicates(subset=["object_id"])
        result[org_name] = unique_objs["object_type"].value_counts().to_dict()
    return result


def print_partition_summary(
    org_events: Dict[str, pd.DataFrame],
    org_relations: Dict[str, pd.DataFrame],
) -> None:
    """Print a summary of the partitioned data."""
    obj_types = get_org_object_types(org_relations)

    print("=" * 60)
    print("Partition Summary")
    print("=" * 60)
    for org_name, ev_df in org_events.items():
        print(f"\n{org_name}:")
        print(f"  Events: {len(ev_df)}")
        print(f"  Activities: {sorted(ev_df['activity'].unique())}")
        if org_name in obj_types:
            print(f"  Object types:")
            for ot, count in sorted(obj_types[org_name].items()):
                print(f"    {ot:30s} {count:>6d} unique objects")