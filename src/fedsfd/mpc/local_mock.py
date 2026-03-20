"""
Plaintext mock MPC backend.

Implements all MPC operations using numpy/scipy directly, producing
identical results to what the real MP-SPDZ backend would compute
(up to fixed-point rounding differences).

This is the primary development and testing path.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from fedsfd.mpc.interface import MPCBackend

logger = logging.getLogger(__name__)


class LocalMockBackend(MPCBackend):
    """Plaintext mock that simulates MPC operations locally.

    For persistence, stores parameters as JSON files in a temporary
    directory (simulating MP-SPDZ's Persistence/ directory).
    """

    def __init__(self):
        self.n_parties = 0
        self.config = {}
        self._persist_dir: Optional[Path] = None

    def setup(self, n_parties: int, config: dict) -> None:
        self.n_parties = n_parties
        self.config = config

    def _get_persist_dir(self) -> Path:
        """Get or create the persistence directory for mock shares."""
        if self._persist_dir is None:
            self._persist_dir = Path(tempfile.mkdtemp(prefix="fedsfd_mock_persist_"))
            logger.debug("Mock persistence dir: %s", self._persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        return self._persist_dir

    def secure_correlation(
        self,
        ts_a: np.ndarray,
        ts_b: np.ndarray,
        party_a: int,
        party_b: int,
        max_lag: int,
    ) -> Tuple[float, int]:
        n = len(ts_a)
        best_corr = 0.0
        best_lag = 0

        for lag in range(min(max_lag + 1, n - 2)):
            if lag == 0:
                xa, xb = ts_a, ts_b
            else:
                xa = ts_a[:-lag]
                xb = ts_b[lag:]

            if len(xa) < 3:
                continue
            if np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
                continue

            corr, _ = stats.pearsonr(xa, xb)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_corr, best_lag

    def secure_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
        party_x: int,
        party_y: int,
    ) -> Tuple[float, float]:
        if len(x) < 2 or np.std(x) < 1e-12:
            return float(np.mean(y)), 0.0

        reg = LinearRegression()
        reg.fit(x.reshape(-1, 1), y)
        return float(reg.intercept_), float(reg.coef_[0])

    def secure_boundary_flow(
        self,
        stock_values: Dict[int, float],
        equation_params: dict,
    ) -> Dict[int, float]:
        source_party = equation_params["source_party"]
        sink_party = equation_params["sink_party"]
        intercept = equation_params["intercept"]
        slope = equation_params["slope"]

        source_val = stock_values.get(source_party, 0.0)
        flow_rate = max(intercept + slope * source_val, 0.0)

        result = {party: 0.0 for party in stock_values}
        result[sink_party] = flow_rate
        return result

    # ------------------------------------------------------------------
    # Persistent secret shares (mock implementation)
    # ------------------------------------------------------------------

    def secure_regression_persist(
        self,
        x: np.ndarray,
        y: np.ndarray,
        party_x: int,
        party_y: int,
        eq_id: int,
    ) -> Tuple[float, float]:
        intercept, slope = self.secure_regression(x, y, party_x, party_y)

        # Persist to file (simulates MP-SPDZ writing shares)
        persist_dir = self._get_persist_dir()
        params_file = persist_dir / f"eq_{eq_id}.json"
        params_file.write_text(json.dumps({
            "intercept": intercept,
            "slope": slope,
            "eq_id": eq_id,
        }))
        logger.debug("Mock persisted eq_%d: intercept=%.6f, slope=%.6f",
                      eq_id, intercept, slope)

        return intercept, slope

    def secure_boundary_flow_from_persistence(
        self,
        stock_values: Dict[int, float],
        equation_params: dict,
    ) -> Dict[int, float]:
        source_party = equation_params["source_party"]
        sink_party = equation_params["sink_party"]
        eq_id = equation_params["eq_id"]

        # Read persisted params
        persist_dir = self._get_persist_dir()
        params_file = persist_dir / f"eq_{eq_id}.json"
        if not params_file.exists():
            raise FileNotFoundError(
                f"No persisted parameters for eq_id={eq_id}. "
                f"Run secure_regression_persist() first."
            )
        params = json.loads(params_file.read_text())
        intercept = params["intercept"]
        slope = params["slope"]

        source_val = stock_values.get(source_party, 0.0)
        flow_rate = max(intercept + slope * source_val, 0.0)

        result = {party: 0.0 for party in stock_values}
        result[sink_party] = flow_rate
        return result

    def clear_persisted_params(self, eq_ids: Optional[list] = None) -> None:
        if self._persist_dir is None or not self._persist_dir.exists():
            return

        if eq_ids is None:
            # Clear all
            for f in self._persist_dir.glob("eq_*.json"):
                f.unlink()
            logger.info("Cleared all mock persisted parameters")
        else:
            for eq_id in eq_ids:
                f = self._persist_dir / f"eq_{eq_id}.json"
                if f.exists():
                    f.unlink()
            logger.info("Cleared mock persisted parameters for eq_ids=%s", eq_ids)