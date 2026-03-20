"""
Abstract MPC (Multi-Party Computation) backend interface.

Defines the operations needed for federated SFD discovery and simulation:
  - secure_correlation: cross-party lagged Pearson correlation
  - secure_regression: cross-party linear regression
  - secure_boundary_flow: compute boundary flow rates from shared stocks

Implementations: LocalMockBackend (plaintext), MP-SPDZBackend (real MPC).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class MPCBackend(ABC):
    """Abstract interface for MPC operations in federated SFD."""

    @abstractmethod
    def setup(self, n_parties: int, config: dict) -> None:
        """Initialize the MPC backend.

        Parameters
        ----------
        n_parties : int
            Number of participating organizations.
        config : dict
            Backend-specific configuration.
        """

    @abstractmethod
    def secure_correlation(
        self,
        ts_a: np.ndarray,
        ts_b: np.ndarray,
        party_a: int,
        party_b: int,
        max_lag: int,
    ) -> Tuple[float, int]:
        """Compute lagged Pearson correlation between two private time series.

        Parameters
        ----------
        ts_a, ts_b : np.ndarray
            Time series held by party_a and party_b respectively.
        party_a, party_b : int
            Party indices (0-based).
        max_lag : int
            Maximum lag to test.

        Returns
        -------
        (max_correlation, optimal_lag) : (float, int)
            The signed correlation and lag that maximize |corr|.
        """

    @abstractmethod
    def secure_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
        party_x: int,
        party_y: int,
    ) -> Tuple[float, float]:
        """Fit linear regression y = α + β·x across two parties.

        Parameters
        ----------
        x : np.ndarray
            Predictor held by party_x.
        y : np.ndarray
            Response held by party_y.
        party_x, party_y : int
            Party indices.

        Returns
        -------
        (intercept, slope) : (float, float)
        """

    @abstractmethod
    def secure_boundary_flow(
        self,
        stock_values: Dict[int, float],
        equation_params: dict,
    ) -> Dict[int, float]:
        """Compute boundary flow rates from shared stock values.

        Parameters
        ----------
        stock_values : dict of {party_id: stock_value}
            Each party contributes its boundary stock value.
        equation_params : dict
            Boundary equation parameters (intercept, slope, source_party, sink_party).

        Returns
        -------
        dict of {party_id: incoming_flow_rate}
            Flow rates delivered to each receiving party.
        """

    # ------------------------------------------------------------------
    # Persistent secret shares
    # ------------------------------------------------------------------

    @abstractmethod
    def secure_regression_persist(
        self,
        x: np.ndarray,
        y: np.ndarray,
        party_x: int,
        party_y: int,
        eq_id: int,
    ) -> Tuple[float, float]:
        """Fit regression AND persist (α, β) as secret shares.

        The learned parameters are written to protocol-specific persistence
        files (one share per party). They can later be read back by
        secure_boundary_flow_from_persistence() without ever leaving the
        secret-shared domain.

        Parameters
        ----------
        x : np.ndarray
            Predictor held by party_x.
        y : np.ndarray
            Response held by party_y.
        party_x, party_y : int
            Party indices.
        eq_id : int
            Unique identifier for this boundary equation. Used as the
            persistence channel — must match the eq_id passed to
            secure_boundary_flow_from_persistence().

        Returns
        -------
        (intercept, slope) : (float, float)
            Revealed for diagnostic purposes (e.g. R² computation).
            In a production deployment these can be ignored.
        """

    @abstractmethod
    def secure_boundary_flow_from_persistence(
        self,
        stock_values: Dict[int, float],
        equation_params: dict,
    ) -> Dict[int, float]:
        """Compute boundary flow using persisted (α, β) shares.

        Reads the equation parameters from persistence files written by
        secure_regression_persist(). The parameters never leave the
        secret-shared domain — only the resulting flow rate is revealed.

        Parameters
        ----------
        stock_values : dict of {party_id: stock_value}
        equation_params : dict with keys:
            source_party, sink_party, eq_id.
            Note: intercept/slope are NOT needed — they are read from
            persistence.

        Returns
        -------
        dict of {party_id: incoming_flow_rate}
        """

    @abstractmethod
    def clear_persisted_params(self, eq_ids: Optional[list] = None) -> None:
        """Delete persisted boundary equation parameters.

        Call this after a federation session completes to clean up
        protocol-specific persistence files.

        Parameters
        ----------
        eq_ids : list of int, optional
            Specific equation IDs to clear. If None, clears ALL
            persisted parameters.
        """