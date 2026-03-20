"""
Load OCEL 2.0 Logistics SQLite files via pm4py.

Provides utilities to extract a flat events DataFrame with columns:
event_id, activity, timestamp, and associated object IDs by type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import pm4py


def load_ocel(path: str | Path) -> pm4py.objects.ocel.obj.OCEL:
    """Load an OCEL 2.0 SQLite file.

    Parameters
    ----------
    path : str or Path
        Path to the .sqlite file.

    Returns
    -------
    pm4py.objects.ocel.obj.OCEL
        The parsed OCEL log object.
    """
    path = str(path)
    ocel = pm4py.read.read_ocel2_sqlite(path)
    return ocel


def load_ocel_simple_sqlite(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a simplified OCEL SQLite (EVENTS / OBJECTS / RELATIONS tables).

    This handles CPN-generated SQLite files that use a flat 3-table layout
    instead of the full OCEL 2.0 schema expected by pm4py.

    Parameters
    ----------
    path : str or Path
        Path to the .sqlite file.

    Returns
    -------
    events_df : pd.DataFrame
        Columns: event_id, activity, timestamp
    relations_df : pd.DataFrame
        Columns: event_id, object_id, object_type, activity, timestamp, qualifier
    """
    import sqlite3

    conn = sqlite3.connect(str(path))

    events_df = pd.read_sql("SELECT * FROM EVENTS", conn)
    events_df = events_df.rename(columns={
        "ocel:eid": "event_id",
        "ocel:activity": "activity",
        "ocel:timestamp": "timestamp",
    })
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format="ISO8601")
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)

    relations_df = pd.read_sql("SELECT * FROM RELATIONS", conn)
    relations_df = relations_df.rename(columns={
        "ocel:eid": "event_id",
        "ocel:activity": "activity",
        "ocel:timestamp": "timestamp",
        "ocel:oid": "object_id",
        "ocel:type": "object_type",
        "ocel:qualifier": "qualifier",
    })
    relations_df["timestamp"] = pd.to_datetime(relations_df["timestamp"], format="ISO8601")

    conn.close()
    return events_df, relations_df


def extract_events_df(ocel: pm4py.objects.ocel.obj.OCEL) -> pd.DataFrame:
    """Extract a flat events DataFrame from an OCEL log.

    Each row is one event with columns:
        event_id, activity, timestamp

    Parameters
    ----------
    ocel : OCEL
        The OCEL log object.

    Returns
    -------
    pd.DataFrame
        Events table sorted by timestamp.
    """
    df = ocel.events.copy()
    df = df.rename(columns={
        "ocel:eid": "event_id",
        "ocel:activity": "activity",
        "ocel:timestamp": "timestamp",
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def extract_relations_df(ocel: pm4py.objects.ocel.obj.OCEL) -> pd.DataFrame:
    """Extract the event-object relations DataFrame.

    Columns: event_id, object_id, object_type, activity, timestamp, qualifier

    Parameters
    ----------
    ocel : OCEL
        The OCEL log object.

    Returns
    -------
    pd.DataFrame
        Relations table.
    """
    df = ocel.relations.copy()
    df = df.rename(columns={
        "ocel:eid": "event_id",
        "ocel:oid": "object_id",
        "ocel:type": "object_type",
        "ocel:activity": "activity",
        "ocel:timestamp": "timestamp",
        "ocel:qualifier": "qualifier",
    })
    return df


def extract_objects_df(ocel: pm4py.objects.ocel.obj.OCEL) -> pd.DataFrame:
    """Extract the objects DataFrame.

    Columns: object_id, object_type

    Parameters
    ----------
    ocel : OCEL
        The OCEL log object.

    Returns
    -------
    pd.DataFrame
        Objects table.
    """
    df = ocel.objects.copy()
    df = df.rename(columns={
        "ocel:oid": "object_id",
        "ocel:type": "object_type",
    })
    return df


def get_time_horizon(events_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (first_timestamp, last_timestamp) from the events DataFrame.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have a 'timestamp' column.

    Returns
    -------
    tuple of (pd.Timestamp, pd.Timestamp)
    """
    return events_df["timestamp"].min(), events_df["timestamp"].max()


def print_summary(ocel: pm4py.objects.ocel.obj.OCEL) -> None:
    """Print summary statistics of the OCEL log to stdout."""
    events_df = extract_events_df(ocel)
    objects_df = extract_objects_df(ocel)
    t_min, t_max = get_time_horizon(events_df)

    print("=" * 60)
    print("OCEL 2.0 Log Summary")
    print("=" * 60)
    print(f"Total events:       {len(events_df)}")
    print(f"Total objects:      {len(objects_df)}")
    print(f"Event types:        {events_df['activity'].nunique()}")
    print(f"Object types:       {objects_df['object_type'].nunique()}")
    print(f"Time range:         {t_min} → {t_max}")
    print(f"Duration:           {t_max - t_min}")
    print()
    print("Events per activity:")
    for act, count in events_df["activity"].value_counts().items():
        print(f"  {act:40s} {count:>6d}")
    print()
    print("Objects per type:")
    for ot, count in objects_df["object_type"].value_counts().items():
        print(f"  {ot:40s} {count:>6d}")