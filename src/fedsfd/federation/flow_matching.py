"""
Cross-organization flow matching for federation.

Two modes:
  A) Manual matches from config (domain knowledge)
  B) Correlation-based auto-discovery (fallback)

Both modes produce the same output: a list of FlowMatch objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from fedsfd.mpc.interface import MPCBackend
from fedsfd.sfd.model import SFD


@dataclass
class FlowMatch:
    """A matched pair of cross-organization flows.

    Attributes
    ----------
    outflow_org : str
        Organization owning the outflow.
    outflow_name : str
        Name of the outflow (throughput leaving the source org).
    inflow_org : str
        Organization owning the inflow.
    inflow_name : str
        Name of the inflow (arrival entering the sink org).
    lag : int
        Time lag in Δt steps (inflow lags outflow by this many steps).
    method : str
        How this match was determined: "manual" or "correlation".
    correlation : float
        Cross-correlation value (0.0 for manual matches).
    """
    outflow_org: str
    outflow_name: str
    inflow_org: str
    inflow_name: str
    lag: int = 0
    method: str = "manual"
    correlation: float = 0.0


def manual_flow_matching(
    flow_matches_config: List[dict],
) -> List[FlowMatch]:
    """Create flow matches from explicit config entries.

    Parameters
    ----------
    flow_matches_config : list of dict
        Each entry has keys: outflow (org, flow), inflow (org, flow), lag.

    Returns
    -------
    list of FlowMatch
    """
    matches = []
    for entry in flow_matches_config:
        outflow = entry["outflow"]
        inflow = entry["inflow"]
        matches.append(FlowMatch(
            outflow_org=outflow["org"],
            outflow_name=outflow["flow"],
            inflow_org=inflow["org"],
            inflow_name=inflow["flow"],
            lag=entry.get("lag", 0),
            method="manual",
            correlation=0.0,
        ))
    return matches


def _is_link_allowed(
    out_org: str,
    in_org: str,
    allowed_links: Optional[Dict[str, Dict[str, List[str]]]],
) -> bool:
    """Check whether an outflow→inflow link between two orgs is permitted.

    When *allowed_links* is ``None`` (unconstrained), every cross-org pair is
    allowed.  Otherwise, the outflow org must list the inflow org in its
    ``outflow_to`` set **and** the inflow org must list the outflow org in its
    ``inflow_from`` set (if those keys are declared for the respective org).
    An org that has no entry in *allowed_links* is treated as unconstrained.
    """
    if allowed_links is None:
        return True

    out_spec = allowed_links.get(out_org)
    if out_spec is not None:
        allowed_targets = out_spec.get("outflow_to")
        if allowed_targets is not None and in_org not in allowed_targets:
            return False

    in_spec = allowed_links.get(in_org)
    if in_spec is not None:
        allowed_sources = in_spec.get("inflow_from")
        if allowed_sources is not None and out_org not in allowed_sources:
            return False

    return True


def correlation_flow_matching(
    local_sfds: Dict[str, SFD],
    org_party_map: Dict[str, int],
    mpc_backend: MPCBackend,
    correlation_threshold: float = 0.5,
    max_lag: int = 3,
    allowed_links: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> List[FlowMatch]:
    """Discover cross-org flow matches via lagged correlation.

    For each pair of orgs (o_i, o_j), compute lagged cross-correlation
    between every external outflow of o_i and every external inflow of o_j.

    Parameters
    ----------
    local_sfds : dict of {org_name: SFD}
        Per-org discovered SFDs.
    org_party_map : dict of {org_name: party_index}
        Maps org names to MPC party indices.
    mpc_backend : MPCBackend
        MPC backend for secure correlation computation.
    correlation_threshold : float
        Minimum |correlation| for a match (default 0.5).
    max_lag : int
        Maximum lag to test (default 3).
    allowed_links : dict or None
        Per-org directional constraints.  Each org may declare
        ``outflow_to: [OrgB, ...]`` and/or ``inflow_from: [OrgA, ...]``.
        Pairs not permitted are skipped.  ``None`` means unconstrained.

    Returns
    -------
    list of FlowMatch
    """
    matches = []
    org_names = list(local_sfds.keys())

    # Collect all external outflows and inflows with their time series
    outflows = {}  # (org, flow_name) -> (flow_obj, time_series)
    inflows = {}

    for org_name, sfd in local_sfds.items():
        for f in sfd.get_external_outflows():
            if f.time_series is not None and len(f.time_series) > 0:
                outflows[(org_name, f.name)] = (f, f.time_series)
        for f in sfd.get_external_inflows():
            if f.time_series is not None and len(f.time_series) > 0:
                inflows[(org_name, f.name)] = (f, f.time_series)

    # Also consider unmatched throughputs and arrivals as potential cross-org flows.
    # In practice, even flows classified as "internal" by the within-org discovery
    # might be cross-org flows; the federation config names them explicitly.
    # Here we try all throughputs as potential outflows and all arrivals as inflows.
    for org_name, sfd in local_sfds.items():
        for f in sfd.flows:
            if f.time_series is not None and len(f.time_series) > 0:
                if "throughput" in f.name:
                    outflows[(org_name, f.name)] = (f, f.time_series)
                elif "arrival" in f.name:
                    inflows[(org_name, f.name)] = (f, f.time_series)

    # All-pairs thresholded matching: for each (outflow, inflow) pair across
    # different orgs, accept the match if max_τ |ρ| ≥ θ.  This allows
    # many-to-one (multiple outflows feeding the same inflow) and
    # one-to-many (one outflow feeding multiple inflows), matching the
    # theory that "an inflow can come from the stocks of multiple orgs."
    seen_pairs = set()

    for (out_org, out_name), (out_flow, out_ts) in outflows.items():
        for (in_org, in_name), (in_flow, in_ts) in inflows.items():
            if in_org == out_org:
                continue  # skip same org
            if not _is_link_allowed(out_org, in_org, allowed_links):
                continue

            pair_key = (out_org, out_name, in_org, in_name)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            party_out = org_party_map.get(out_org, 0)
            party_in = org_party_map.get(in_org, 1)

            corr, lag = mpc_backend.secure_correlation(
                out_ts, in_ts, party_out, party_in, max_lag
            )

            if abs(corr) >= correlation_threshold:
                matches.append(FlowMatch(
                    outflow_org=out_org,
                    outflow_name=out_name,
                    inflow_org=in_org,
                    inflow_name=in_name,
                    lag=lag,
                    method="correlation",
                    correlation=corr,
                ))

    return matches


def discover_flow_matches(
    local_sfds: Dict[str, SFD],
    config,
    mpc_backend: Optional[MPCBackend] = None,
) -> List[FlowMatch]:
    """Discover cross-org flow matches using either manual or auto mode.

    Parameters
    ----------
    local_sfds : dict of {org_name: SFD}
    config : ExperimentConfig
    mpc_backend : MPCBackend or None
        Required for correlation mode.

    Returns
    -------
    list of FlowMatch
    """
    # Mode A: manual matches (takes priority)
    if config.federation.flow_matches:
        matches = manual_flow_matching(config.federation.flow_matches)
        print(f"  Flow matching mode: MANUAL ({len(matches)} matches)")
        return matches

    # Mode B: correlation-based auto-discovery
    if mpc_backend is None:
        from fedsfd.mpc.local_mock import LocalMockBackend
        mpc_backend = LocalMockBackend()
        mpc_backend.setup(len(local_sfds), {})

    org_party_map = {name: i for i, name in enumerate(local_sfds.keys())}
    matches = correlation_flow_matching(
        local_sfds=local_sfds,
        org_party_map=org_party_map,
        mpc_backend=mpc_backend,
        correlation_threshold=config.federation.correlation_threshold,
        max_lag=config.federation.max_lag,
        allowed_links=config.federation.allowed_links,
    )
    print(f"  Flow matching mode: CORRELATION ({len(matches)} matches)")
    return matches


def print_flow_matches(matches: List[FlowMatch]) -> None:
    """Print a summary of discovered flow matches."""
    print(f"\n{'=' * 60}")
    print(f"Flow Matches ({len(matches)} total)")
    print(f"{'=' * 60}")
    for i, m in enumerate(matches):
        print(
            f"  {i+1}. {m.outflow_org}/{m.outflow_name} "
            f"→ {m.inflow_org}/{m.inflow_name}  "
            f"lag={m.lag}  method={m.method}  corr={m.correlation:.3f}"
        )