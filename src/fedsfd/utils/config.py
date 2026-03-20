"""
YAML configuration loader and validator for federated SFD experiments.

Supports both the legacy scopes-only config and the new analyst-defined
``sfd_variables`` config.  When ``sfd_variables`` is present, the
aggregation module uses analyst-defined variable definitions instead of
the generic WIP/throughput/arrival per scope.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DiscoveryConfig:
    """CLD / SFD discovery parameters."""
    correlation_threshold: float = 0.3
    max_lag: int = 3


@dataclass
class FederationConfig:
    """Federation-level settings."""
    flow_matches: Optional[List[Dict[str, Any]]] = None
    correlation_threshold: float = 0.5
    max_lag: int = 3
    model_type: str = "linear"
    allowed_links: Optional[Dict[str, Dict[str, List[str]]]] = None


@dataclass
class SimulationConfig:
    """Simulation settings."""
    horizon: Optional[int] = None
    what_if: List[Dict[str, Any]] = field(default_factory=list)
    baseline_data_path: Optional[str] = None
    whatif_data_path: Optional[str] = None
    warmup_days: int = 0


@dataclass
class MPCConfig:
    """MPC backend settings."""
    backend: str = "mock"
    mp_spdz_path: Optional[str] = None
    protocol: str = "semi2k"
    persist_shares: bool = False


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    data_path: str
    time_window_delta: str
    random_seed: int
    organizations: Dict[str, List[str]]  # org_name -> [activities]
    scopes: Dict[str, Dict[str, List[str]]]  # org -> {scope -> [activities]}
    federation: FederationConfig
    simulation: SimulationConfig
    mpc: MPCConfig
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    train_fraction: float = 0.7

    # Analyst-defined SFD variables (new): org -> {var_name -> var_def}
    sfd_variables: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None

    # Derived mappings (computed on init)
    activity_to_org: Dict[str, str] = field(default_factory=dict, repr=False)
    activity_to_scope: Dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def has_sfd_variables(self) -> bool:
        """True if analyst-defined SFD variables are present."""
        return self.sfd_variables is not None and len(self.sfd_variables) > 0

    def get_variable_roles(self, org: str) -> Dict[str, str]:
        """Get {variable_name: role} for an organization's SFD variables."""
        if not self.has_sfd_variables or org not in self.sfd_variables:
            return {}
        return {
            name: vdef.get("role", "auxiliary")
            for name, vdef in self.sfd_variables[org].items()
        }

    def __post_init__(self):
        """Build derived lookup mappings and validate."""
        # activity -> org
        for org_name, activities in self.organizations.items():
            for act in activities:
                if act in self.activity_to_org:
                    raise ValueError(
                        f"Activity '{act}' assigned to multiple orgs: "
                        f"'{self.activity_to_org[act]}' and '{org_name}'"
                    )
                self.activity_to_org[act] = org_name

        # activity -> scope
        for org_name, scope_map in self.scopes.items():
            for scope_name, activities in scope_map.items():
                for act in activities:
                    self.activity_to_scope[act] = scope_name


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    ExperimentConfig
        Validated configuration dataclass.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Parse organizations: {name: [activities]}
    organizations = {}
    for org_name, org_cfg in raw["organizations"].items():
        organizations[org_name] = org_cfg["activities"]

    # Parse scopes
    scopes = {}
    for org_name, scope_map in raw.get("scopes", {}).items():
        scopes[org_name] = {}
        for scope_name, activities in scope_map.items():
            scopes[org_name][scope_name] = activities

    # Parse federation
    fed_raw = raw.get("federation", {})
    federation = FederationConfig(
        flow_matches=fed_raw.get("flow_matches"),
        correlation_threshold=fed_raw.get("correlation_threshold", 0.5),
        max_lag=fed_raw.get("max_lag", 3),
        model_type=fed_raw.get("model_type", "linear"),
        allowed_links=fed_raw.get("allowed_links"),
    )

    # Parse simulation
    sim_raw = raw.get("simulation", {})
    simulation = SimulationConfig(
        horizon=sim_raw.get("horizon"),
        what_if=sim_raw.get("what_if", []),
        baseline_data_path=sim_raw.get("baseline_data_path"),
        whatif_data_path=sim_raw.get("whatif_data_path"),
        warmup_days=sim_raw.get("warmup_days", 0),
    )

    # Parse MPC
    mpc_raw = raw.get("mpc", {})
    mpc = MPCConfig(
        backend=mpc_raw.get("backend", "mock"),
        mp_spdz_path=mpc_raw.get("mp_spdz_path"),
        protocol=mpc_raw.get("protocol", "semi2k"),
        persist_shares=mpc_raw.get("persist_shares", False),
    )

    # Parse discovery parameters
    disc_raw = raw.get("discovery", {})
    discovery = DiscoveryConfig(
        correlation_threshold=disc_raw.get("correlation_threshold", 0.3),
        max_lag=disc_raw.get("max_lag", 3),
    )

    # Parse sfd_variables (new, optional)
    sfd_variables = raw.get("sfd_variables")

    config = ExperimentConfig(
        data_path=raw["data"]["path"],
        time_window_delta=raw["time_window"]["delta"],
        random_seed=raw.get("random_seed", 42),
        organizations=organizations,
        scopes=scopes,
        federation=federation,
        simulation=simulation,
        mpc=mpc,
        discovery=discovery,
        train_fraction=raw.get("evaluation", {}).get("train_fraction", 0.7),
        sfd_variables=sfd_variables,
    )

    # Validate: every activity in scopes belongs to the right org
    for org_name, scope_map in config.scopes.items():
        if org_name not in config.organizations:
            raise ValueError(f"Scope org '{org_name}' not in organizations")
        org_activities = set(config.organizations[org_name])
        for scope_name, activities in scope_map.items():
            for act in activities:
                if act not in org_activities:
                    raise ValueError(
                        f"Activity '{act}' in scope '{scope_name}' "
                        f"not in org '{org_name}' activities"
                    )

    # Validate: every org activity appears in exactly one scope
    for org_name, org_activities in config.organizations.items():
        if org_name in config.scopes:
            scoped = set()
            for scope_name, acts in config.scopes[org_name].items():
                scoped.update(acts)
            missing = set(org_activities) - scoped
            if missing:
                raise ValueError(
                    f"Activities {missing} in org '{org_name}' "
                    f"not covered by any scope"
                )

    return config