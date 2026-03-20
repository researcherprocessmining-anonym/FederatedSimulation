"""
Stock-Flow Diagram (SFD) data structures.

An SFD consists of:
  - **Stocks**: accumulate over time via integration
  - **Flows**: rates that feed into/out of stocks
  - **Auxiliary variables**: intermediate computed values
  - **Information dependencies**: causal links between variables
  - **Cloud stock**: represents the external boundary (outside world)

Reference: Pourbafrani & van der Aalst (2022), "Discovering system dynamics
simulation models using process mining."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Stock:
    """A stock (level) variable that accumulates over time.

    Attributes
    ----------
    name : str
        Unique name within the SFD (e.g., "order_registration_wip").
    org : str
        Organization this stock belongs to.
    initial_value : float
        Value at t=0.
    is_cloud : bool
        True for the external "cloud" stock representing the outside world.
    equation : callable or None
        Accumulation function f(state_dict, params, t) → new stock value.
        Per Definition 5: stock[t] = stock[t-1] + Σ η(flow) * flow[t].
    equation_params : dict or None
        Parameters for the accumulation equation (inflows, outflows).
    """
    name: str
    org: str
    initial_value: float = 0.0
    is_cloud: bool = False
    equation: Optional[Callable] = field(default=None, repr=False)
    equation_params: Optional[Dict[str, Any]] = None

    def __hash__(self):
        return hash((self.name, self.org))

    def __eq__(self, other):
        if not isinstance(other, Stock):
            return NotImplemented
        return self.name == other.name and self.org == other.org

    def compute(self, state: Dict[str, float], t: int = 0) -> float:
        """Compute stock value given current state.

        Parameters
        ----------
        state : dict
            Mapping of variable_name → current_value.  Must include
            this stock's previous value and all connected flow rates.
        t : int
            Current time step index.

        Returns
        -------
        float
            Stock value at current time step.
        """
        if self.equation is not None:
            return self.equation(state, self.equation_params, t)
        return state.get(self.name, self.initial_value)


@dataclass
class Flow:
    """A flow (rate) variable connecting two stocks.

    Attributes
    ----------
    name : str
        Unique name (e.g., "order_registration_throughput").
    org : str
        Organization this flow belongs to.
    source : Stock
        Stock from which this flow drains.
    sink : Stock
        Stock into which this flow feeds.
    equation : callable or None
        Function f(state_dict, params) → flow_rate.
    equation_params : dict or None
        Parameters for the equation (e.g., regression coefficients).
    time_series : np.ndarray or None
        Original observed time series (for equation fitting).
    """
    name: str
    org: str
    source: Stock = None
    sink: Stock = None
    equation: Optional[Callable] = field(default=None, repr=False)
    equation_params: Optional[Dict[str, Any]] = None
    time_series: Optional[np.ndarray] = field(default=None, repr=False)

    def __hash__(self):
        return hash((self.name, self.org))

    def __eq__(self, other):
        if not isinstance(other, Flow):
            return NotImplemented
        return self.name == other.name and self.org == other.org

    def compute(self, state: Dict[str, float], t: int = 0) -> float:
        """Compute flow rate given current state.

        Parameters
        ----------
        state : dict
            Mapping of variable_name → current_value.
        t : int
            Current time step index.

        Returns
        -------
        float
            Flow rate at current time step.
        """
        if self.equation is not None:
            return self.equation(state, self.equation_params, t)
        # Fallback: if we have a time series, replay it
        if self.time_series is not None and 0 <= t < len(self.time_series):
            return float(self.time_series[t])
        return 0.0


@dataclass
class AuxVariable:
    """An auxiliary (computed) variable.

    Attributes
    ----------
    name : str
        Unique name.
    org : str
        Organization.
    equation : callable or None
        Function f(state_dict, params) → value.
    equation_params : dict or None
        Parameters for the equation.
    time_series : np.ndarray or None
        Original observed time series.
    """
    name: str
    org: str
    equation: Optional[Callable] = field(default=None, repr=False)
    equation_params: Optional[Dict[str, Any]] = None
    time_series: Optional[np.ndarray] = field(default=None, repr=False)

    def __hash__(self):
        return hash((self.name, self.org))

    def __eq__(self, other):
        if not isinstance(other, AuxVariable):
            return NotImplemented
        return self.name == other.name and self.org == other.org


@dataclass
class InfoDependency:
    """An information link (causal dependency) between two SFD variables.

    Attributes
    ----------
    source : Stock, Flow, or AuxVariable
        The influencing variable.
    target : Flow or AuxVariable
        The influenced variable.
    lag : int
        Time lag (in number of Δt steps). 0 = contemporaneous.
    correlation : float
        Pearson correlation between source and target (for provenance).
    """
    source: Union[Stock, Flow, AuxVariable]
    target: Union[Flow, AuxVariable]
    lag: int = 0
    correlation: float = 0.0


@dataclass
class SFD:
    """A complete Stock-Flow Diagram for one organization (or the whole system).

    Attributes
    ----------
    name : str
        Descriptive name (e.g., "Company_SFD").
    org : str
        Organization name.
    stocks : list of Stock
        All stock variables (excluding cloud).
    flows : list of Flow
        All flow variables.
    auxiliaries : list of AuxVariable
        All auxiliary variables.
    dependencies : list of InfoDependency
        All information dependencies.
    cloud : Stock
        The external boundary stock.
    """
    name: str
    org: str
    stocks: List[Stock] = field(default_factory=list)
    flows: List[Flow] = field(default_factory=list)
    auxiliaries: List[AuxVariable] = field(default_factory=list)
    dependencies: List[InfoDependency] = field(default_factory=list)
    cloud: Stock = None

    def __post_init__(self):
        if self.cloud is None:
            self.cloud = Stock(
                name=f"{self.org}_cloud",
                org=self.org,
                initial_value=0.0,
                is_cloud=True,
            )

    def get_stock(self, name: str) -> Optional[Stock]:
        """Look up a stock by name."""
        for s in self.stocks:
            if s.name == name:
                return s
        if self.cloud.name == name:
            return self.cloud
        return None

    def get_flow(self, name: str) -> Optional[Flow]:
        """Look up a flow by name."""
        for f in self.flows:
            if f.name == name:
                return f
        return None

    def get_external_inflows(self) -> List[Flow]:
        """Flows coming from the cloud into internal stocks."""
        return [f for f in self.flows if f.source == self.cloud and f.sink != self.cloud]

    def get_external_outflows(self) -> List[Flow]:
        """Flows going from internal stocks to the cloud."""
        return [f for f in self.flows if f.sink == self.cloud and f.source != self.cloud]

    def get_internal_flows(self) -> List[Flow]:
        """Flows between internal stocks (neither endpoint is cloud)."""
        return [
            f for f in self.flows
            if f.source != self.cloud and f.sink != self.cloud
        ]

    def get_stock_inflows(self, stock: Stock) -> List[Flow]:
        """All flows feeding into a stock."""
        return [f for f in self.flows if f.sink == stock]

    def get_stock_outflows(self, stock: Stock) -> List[Flow]:
        """All flows draining from a stock."""
        return [f for f in self.flows if f.source == stock]

    def get_dependencies_for(self, target: Union[Flow, AuxVariable]) -> List[InfoDependency]:
        """All information dependencies targeting a given variable."""
        return [d for d in self.dependencies if d.target == target]

    def print_summary(self) -> None:
        """Print a structured summary of the SFD."""
        print(f"\n{'=' * 60}")
        print(f"SFD: {self.name}  (org: {self.org})")
        print(f"{'=' * 60}")
        print(f"  Stocks:       {len(self.stocks)} + 1 cloud")
        print(f"  Flows:        {len(self.flows)}")
        print(f"    Internal:   {len(self.get_internal_flows())}")
        print(f"    Ext. in:    {len(self.get_external_inflows())}")
        print(f"    Ext. out:   {len(self.get_external_outflows())}")
        print(f"  Auxiliaries:  {len(self.auxiliaries)}")
        print(f"  Dependencies: {len(self.dependencies)}")

        print("\n  Stocks:")
        for s in self.stocks:
            params_str = ""
            if s.equation_params:
                ins = s.equation_params.get("inflows", [])
                outs = s.equation_params.get("outflows", [])
                parts = [f"+{f}" for f in ins] + [f"-{f}" for f in outs]
                params_str = f"  eq: s[t-1] {' '.join(parts)}" if parts else ""
            print(f"    {s.name:40s} init={s.initial_value:.1f}{params_str}")

        print("\n  Flows:")
        for f in self.flows:
            src = f.source.name if f.source else "?"
            snk = f.sink.name if f.sink else "?"
            params_str = ""
            if f.equation_params:
                params_str = f"  params={f.equation_params}"
            print(f"    {f.name:40s} {src} → {snk}{params_str}")

        if self.dependencies:
            print("\n  Dependencies:")
            for d in self.dependencies:
                print(
                    f"    {d.source.name:30s} → {d.target.name:30s} "
                    f"lag={d.lag}  corr={d.correlation:.3f}"
                )