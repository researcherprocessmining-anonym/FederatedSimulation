"""
MP-SPDZ backend for secure multi-party computation.

Implements the MPCBackend interface using real MP-SPDZ execution.
For each operation, the backend:
  1. Generates (or reuses) an .mpc program
  2. Writes party inputs to Player-Data/Input-P{i}-0
  3. Compiles the program with ./compile.py
  4. Runs the protocol (e.g., semi2k) with all parties on localhost
  5. Parses the output from stdout

Requirements:
  - MP-SPDZ installed and compiled (semi-party.x available)
  - Path to MP-SPDZ root set in config (mpc.mp_spdz_path)
  - The MP-SPDZ environment activated (LD_LIBRARY_PATH etc.) before running

Protocol: semi2k (semi-honest, arithmetic mod 2^k) by default.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from fedsfd.mpc.interface import MPCBackend

logger = logging.getLogger(__name__)


class MPSPDZBackend(MPCBackend):
    """Real MP-SPDZ backend for secure multi-party computation.

    Parameters
    ----------
    mp_spdz_path : str or Path
        Path to the MP-SPDZ installation root directory.
    protocol : str
        MPC protocol to use (default: "semi2k").
    n_parties : int
        Number of parties (set during setup).
    """

    def __init__(
        self,
        mp_spdz_path: str | Path = None,
        protocol: str = "semi2k",
    ):
        self.mp_spdz_path: Optional[Path] = Path(mp_spdz_path) if mp_spdz_path else None
        self.protocol = protocol
        self.n_parties = 2
        self.config: dict = {}
        self._programs_dir: Optional[Path] = None
        self._compile_cache: set = set()  # track already-compiled programs

    def setup(self, n_parties: int, config: dict) -> None:
        """Initialize the MP-SPDZ backend.

        Parameters
        ----------
        n_parties : int
            Number of parties.
        config : dict
            Must include 'mp_spdz_path'. Optional: 'protocol', 'activate_script'.
        """
        self.n_parties = n_parties
        self.config = config

        if "mp_spdz_path" in config and config["mp_spdz_path"]:
            self.mp_spdz_path = Path(config["mp_spdz_path"])
        if "protocol" in config and config["protocol"]:
            self.protocol = config["protocol"]

        if self.mp_spdz_path is None:
            raise ValueError(
                "MP-SPDZ path not configured. Set mpc.mp_spdz_path in YAML config "
                "or pass mp_spdz_path to MPSPDZBackend constructor."
            )

        if not self.mp_spdz_path.exists():
            raise FileNotFoundError(f"MP-SPDZ directory not found: {self.mp_spdz_path}")

        # Verify the protocol binary exists
        binary = self._get_binary_name()
        binary_path = self.mp_spdz_path / binary
        if not binary_path.exists():
            raise FileNotFoundError(
                f"Protocol binary not found: {binary_path}\n"
                f"Run 'make -j$(nproc) {binary}' in {self.mp_spdz_path}"
            )

        # Set up programs directory
        self._programs_dir = self.mp_spdz_path / "Programs" / "Source"
        self._programs_dir.mkdir(parents=True, exist_ok=True)

        # Ensure Player-Data directory exists
        player_data = self.mp_spdz_path / "Player-Data"
        player_data.mkdir(exist_ok=True)

        logger.info(
            "MP-SPDZ backend initialized: path=%s, protocol=%s, n_parties=%d",
            self.mp_spdz_path, self.protocol, self.n_parties,
        )

    def _get_binary_name(self) -> str:
        """Get the MP-SPDZ binary name for the configured protocol."""
        protocol_binaries = {
            "semi2k": "semi-party.x",
            "semi": "semi-party.x",
            "mascot": "mascot-party.x",
            "spdz2k": "spdz2k-party.x",
            "shamir": "shamir-party.x",
            "mal-shamir": "mal-shamir-party.x",
            "replicated": "replicated-ring-party.x",
            "ps-rep-ring": "ps-rep-ring-party.x",
        }
        return protocol_binaries.get(self.protocol, f"{self.protocol}-party.x")

    def _install_program(self, program_name: str, program_source: str) -> None:
        """Write an .mpc program to MP-SPDZ's Programs/Source directory.

        Parameters
        ----------
        program_name : str
            Name of the program (without .mpc extension).
        program_source : str
            The .mpc program source code.
        """
        target = self._programs_dir / f"{program_name}.mpc"
        target.write_text(program_source)
        logger.debug("Installed program: %s", target)

    def _write_inputs(self, party_inputs: Dict[int, List[float]]) -> None:
        """Write input files for each party.

        Parameters
        ----------
        party_inputs : dict of {party_id: list of float values}
            Values are written as whitespace-separated text.
        """
        player_data = self.mp_spdz_path / "Player-Data"
        player_data.mkdir(exist_ok=True)

        for party_id, values in party_inputs.items():
            input_file = player_data / f"Input-P{party_id}-0"
            with open(input_file, "w") as f:
                for v in values:
                    f.write(f"{v}\n")
            logger.debug(
                "Wrote %d values to %s", len(values), input_file
            )

    def _compile_program(self, program_name: str, args: List[str] = None) -> None:
        """Compile an .mpc program.

        Parameters
        ----------
        program_name : str
            Name of the program (without .mpc).
        args : list of str, optional
            Additional compile-time arguments passed to the program.
        """
        # Build cache key from program name + args
        cache_key = (program_name, tuple(args or []))
        if cache_key in self._compile_cache:
            logger.debug("Program already compiled: %s %s", program_name, args)
            return

        # The compiled bytecode name includes the args (MP-SPDZ convention)
        if args:
            bytecode_name = f"{program_name}-{'-'.join(args)}"
        else:
            bytecode_name = program_name

        # Clear any stale on-disk bytecode from previous compilations
        # (e.g., compiled without -R when we now need -R 64, or vice versa).
        # MP-SPDZ stores bytecode in Programs/Bytecode/ and schedules in
        # Programs/Schedules/.
        for subdir in ("Bytecode", "Schedules"):
            target_dir = self.mp_spdz_path / "Programs" / subdir
            if target_dir.exists():
                import glob
                for stale in target_dir.glob(f"{bytecode_name}*"):
                    logger.debug("Removing stale bytecode: %s", stale)
                    stale.unlink()

        cmd = ["python3", str(self.mp_spdz_path / "compile.py")]
        # Note: do NOT pass -R here. The compilation mode (ring vs field)
        # must match the binary. semi-party.x compiled from source with
        # default settings is a field-mode binary and rejects ring bytecode.
        # If you have a ring-mode binary (e.g., built with USE_GF2N_LONG=0),
        # add -R <bitlength> here.
        cmd.append(program_name)
        if args:
            cmd.extend(args)

        logger.info("Compiling: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            cwd=str(self.mp_spdz_path),
            capture_output=True,
            text=True,
            timeout=120,
            env=self._get_env(),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"MP-SPDZ compilation failed for '{program_name}':\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        self._compile_cache.add(cache_key)
        logger.debug("Compilation output: %s", result.stdout[-500:] if result.stdout else "")

    def _get_env(self) -> dict:
        """Build environment variables for MP-SPDZ subprocesses.

        Inherits the current process environment (which includes any
        variables set by the MP-SPDZ activate.sh script, such as
        LD_LIBRARY_PATH for libsodium).

        If the config contains an 'activate_script' path, the user is
        responsible for sourcing it before running the Python pipeline.
        """
        return dict(os.environ)

    def _run_program(
        self,
        program_name: str,
        args: List[str] = None,
        timeout: int = 300,
    ) -> str:
        """Run a compiled program with all parties on localhost.

        Launches n_parties instances of the protocol binary directly,
        rather than using convenience shell scripts (which may reference
        different binary names than what's actually compiled).

        Parameters
        ----------
        program_name : str
            Compiled program name (without compile-time args).
        args : list of str, optional
            Compile-time args that become part of the bytecode name.
        timeout : int
            Maximum execution time in seconds.

        Returns
        -------
        str
            Stdout from party 0 (which produces output by default in MP-SPDZ).
        """
        # The compiled bytecode name includes the args
        if args:
            bytecode_name = f"{program_name}-{'-'.join(args)}"
        else:
            bytecode_name = program_name

        binary = self._get_binary_name()
        binary_path = self.mp_spdz_path / binary
        env = self._get_env()

        # Pick a random-ish port to reduce collisions between concurrent runs
        base_port = 14000 + random.randint(0, 1000)

        processes = []
        try:
            for party in range(self.n_parties):
                cmd = [
                    str(binary_path),
                    "-N", str(self.n_parties),
                    "-p", str(party),
                    "-pn", str(base_port),
                    "-h", "localhost",
                    bytecode_name,
                ]
                logger.info("Starting party %d: %s", party, " ".join(cmd))

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.mp_spdz_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                processes.append(proc)

            # Wait for all processes to complete
            outputs = []
            errors = []
            return_codes = []
            for i, proc in enumerate(processes):
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    logger.error("Party %d timed out after %ds", i, timeout)

                outputs.append(stdout)
                errors.append(stderr)
                return_codes.append(proc.returncode)

            # Check for failures — party 0 is the primary output source
            if return_codes[0] != 0:
                raise RuntimeError(
                    f"MP-SPDZ execution failed for '{bytecode_name}':\n"
                    f"Party 0 returned {return_codes[0]}\n"
                    f"stdout: {outputs[0][:2000]}\n"
                    f"stderr: {errors[0][:2000]}"
                )

            # Log warnings for other party failures (non-fatal)
            for i in range(1, self.n_parties):
                if return_codes[i] != 0:
                    logger.warning(
                        "Party %d returned %d: %s",
                        i, return_codes[i], errors[i][:500],
                    )

            return outputs[0]

        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()

    # -----------------------------------------------------------------------
    # MPCBackend interface implementation
    # -----------------------------------------------------------------------

    def secure_correlation(
        self,
        ts_a: np.ndarray,
        ts_b: np.ndarray,
        party_a: int,
        party_b: int,
        max_lag: int,
    ) -> Tuple[float, int]:
        """Compute lagged Pearson correlation via MP-SPDZ.

        The MPC program reveals only the summary statistics (covariance,
        variances) at each lag. The final correlation is computed locally.

        Note: The .mpc program always reads ts_a from party 0 and ts_b
        from party 1. The party_a/party_b arguments identify the logical
        owners but data is mapped to physical parties 0/1 for the MPC
        program.
        """
        n = min(len(ts_a), len(ts_b))
        if n < 3:
            return 0.0, 0

        max_lag = min(max_lag, n - 3)

        # Install the program (idempotent)
        program_source = _load_bundled_program("fedsfd_correlation")
        self._install_program("fedsfd_correlation", program_source)

        # Write inputs: always map to parties 0 and 1 since the .mpc
        # program hardcodes get_input_from(0) and get_input_from(1).
        inputs = {
            0: list(ts_a[:n].astype(float)),
            1: list(ts_b[:n].astype(float)),
        }
        # Dummy inputs for any remaining parties
        for p in range(self.n_parties):
            if p not in inputs:
                inputs[p] = [0.0]
        self._write_inputs(inputs)

        # Compile with dimensions
        compile_args = [str(n), str(max_lag)]
        self._compile_program("fedsfd_correlation", compile_args)

        # Run
        output = self._run_program("fedsfd_correlation", compile_args)

        # Parse output
        return self._parse_correlation_output(output)

    def _parse_correlation_output(self, output: str) -> Tuple[float, int]:
        """Parse correlation program output to find best lag.

        Output format:
          LAG <lag> COV <cov> VARA <var_a> VARB <var_b> N <eff_n>
          ...
          DONE
        """
        best_corr = 0.0
        best_lag = 0

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line.startswith("LAG"):
                continue

            m = re.match(
                r"LAG\s+(\d+)\s+COV\s+([-\d.e+]+)\s+VARA\s+([-\d.e+]+)"
                r"\s+VARB\s+([-\d.e+]+)\s+N\s+(\d+)",
                line,
            )
            if not m:
                logger.warning("Could not parse correlation line: %s", line)
                continue

            lag = int(m.group(1))
            cov = float(m.group(2))
            var_a = float(m.group(3))
            var_b = float(m.group(4))
            eff_n = int(m.group(5))

            if eff_n < 3 or var_a < 1e-12 or var_b < 1e-12:
                continue

            denom = math.sqrt(var_a * var_b)
            if denom < 1e-12:
                continue

            corr = cov / denom
            corr = max(-1.0, min(1.0, corr))  # clamp to [-1, 1]

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
        """Fit linear regression y = α + β·x via MP-SPDZ.

        Note: The .mpc program always reads x from party 0 and y from
        party 1. The party_x/party_y arguments identify the logical
        owners but data is mapped to physical parties 0/1 for the MPC
        program.
        """
        n = min(len(x), len(y))
        if n < 2:
            return float(np.mean(y)), 0.0
        if np.std(x) < 1e-12:
            return float(np.mean(y)), 0.0

        # Install program
        program_source = _load_bundled_program("fedsfd_regression")
        self._install_program("fedsfd_regression", program_source)

        # Write inputs: always map to parties 0 and 1 since the .mpc
        # program hardcodes get_input_from(0) and get_input_from(1).
        inputs = {
            0: list(x[:n].astype(float)),
            1: list(y[:n].astype(float)),
        }
        for p in range(self.n_parties):
            if p not in inputs:
                inputs[p] = [0.0]
        self._write_inputs(inputs)

        # Compile and run
        compile_args = [str(n)]
        self._compile_program("fedsfd_regression", compile_args)
        output = self._run_program("fedsfd_regression", compile_args)

        return self._parse_regression_output(output, y)

    def _parse_regression_output(
        self, output: str, y_fallback: np.ndarray
    ) -> Tuple[float, float]:
        """Parse regression program output.

        Output format:
          RESULT INTERCEPT <alpha> SLOPE <beta>
          DONE
        """
        for line in output.strip().split("\n"):
            line = line.strip()
            m = re.match(
                r"RESULT\s+INTERCEPT\s+([-\d.e+]+)\s+SLOPE\s+([-\d.e+]+)",
                line,
            )
            if m:
                intercept = float(m.group(1))
                slope = float(m.group(2))
                return intercept, slope

        logger.warning(
            "Could not parse regression output, falling back to mean:\n%s",
            output[-500:],
        )
        return float(np.mean(y_fallback)), 0.0

    def secure_boundary_flow(
        self,
        stock_values: Dict[int, float],
        equation_params: dict,
    ) -> Dict[int, float]:
        """Compute boundary flow rate via MP-SPDZ.

        The equation parameters (intercept, slope) are public and passed as
        inputs from party 0. The source party provides its stock value as
        a secret input. The program computes:
            flow = max(intercept + slope * stock_value, 0)

        PERFORMANCE NOTE: This method is called once per boundary equation
        per simulation time step. Each call involves writing input files
        and running the MP-SPDZ binary. For long simulations, consider
        using the mock backend for simulation and MP-SPDZ only for
        discovery (correlation + regression), since the boundary equation
        parameters are already public after discovery.

        Parameters
        ----------
        stock_values : dict of {party_id: stock_value}
        equation_params : dict with keys:
            source_party, sink_party, intercept, slope
        """
        source_party = equation_params["source_party"]
        sink_party = equation_params["sink_party"]
        intercept = equation_params["intercept"]
        slope = equation_params["slope"]
        source_val = stock_values.get(source_party, 0.0)

        # Install program
        program_source = _load_bundled_program("fedsfd_boundary_flow")
        self._install_program("fedsfd_boundary_flow", program_source)

        # Write inputs:
        # Party 0 always provides: intercept, slope
        # Source party provides: stock_value
        # If source_party == 0, party 0 provides all three: intercept, slope, stock_value
        inputs = {}
        if source_party == 0:
            inputs[0] = [intercept, slope, source_val]
        else:
            inputs[0] = [intercept, slope]
            inputs[source_party] = [source_val]
        # Dummy inputs for remaining parties
        for p in range(self.n_parties):
            if p not in inputs:
                inputs[p] = [0.0]
        self._write_inputs(inputs)

        # Compile with source_party as the only compile-time arg
        compile_args = [str(source_party)]
        self._compile_program("fedsfd_boundary_flow", compile_args)
        output = self._run_program("fedsfd_boundary_flow", compile_args)

        # Parse output
        flow_rate = self._parse_boundary_flow_output(output)

        # Build result dict
        result = {party: 0.0 for party in stock_values}
        result[sink_party] = flow_rate
        return result

    def _parse_boundary_flow_output(self, output: str) -> float:
        """Parse boundary flow program output.

        Output format:
          RESULT FLOW <flow_rate>
          DONE
        """
        for line in output.strip().split("\n"):
            line = line.strip()
            m = re.match(r"RESULT\s+FLOW\s+([-\d.e+]+)", line)
            if m:
                return max(float(m.group(1)), 0.0)

        logger.warning("Could not parse boundary flow output:\n%s", output[-500:])
        return 0.0

    # ------------------------------------------------------------------
    # Persistent secret shares
    # ------------------------------------------------------------------

    def secure_regression_persist(
        self,
        x: np.ndarray,
        y: np.ndarray,
        party_x: int,
        party_y: int,
        eq_id: int,
    ) -> Tuple[float, float]:
        """Fit regression AND persist (alpha, beta) as secret shares.

        The MPC program computes alpha and beta, writes them to
        Player-Data/Persistence/ as protocol-specific secret shares
        (one file per party, channel = eq_id), and also reveals them
        for diagnostic R-squared computation on the Python side.
        """
        n = min(len(x), len(y))
        if n < 2:
            return float(np.mean(y)), 0.0
        if np.std(x) < 1e-12:
            return float(np.mean(y)), 0.0

        # Ensure Persistence directory exists
        persist_dir = self.mp_spdz_path / "Player-Data" / "Persistence"
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Clear any stale persistence for this eq_id before writing
        self._clear_persistence_channel(eq_id)

        # Install program
        program_source = _load_bundled_program("fedsfd_regression_persist")
        self._install_program("fedsfd_regression_persist", program_source)

        # Write inputs (same as secure_regression)
        inputs = {
            0: list(x[:n].astype(float)),
            1: list(y[:n].astype(float)),
        }
        for p in range(self.n_parties):
            if p not in inputs:
                inputs[p] = [0.0]
        self._write_inputs(inputs)

        # Compile with n and eq_id
        compile_args = [str(n), str(eq_id)]
        self._compile_program("fedsfd_regression_persist", compile_args)
        output = self._run_program("fedsfd_regression_persist", compile_args)

        # Verify persistence was written
        for line in output.strip().split("\n"):
            if f"PERSIST EQID {eq_id} OK" in line:
                logger.info(
                    "Persisted boundary equation %d as secret shares", eq_id
                )
                break
        else:
            logger.warning(
                "Persistence confirmation not found in output for eq_id=%d",
                eq_id,
            )

        return self._parse_regression_output(output, y)

    def secure_boundary_flow_from_persistence(
        self,
        stock_values: Dict[int, float],
        equation_params: dict,
    ) -> Dict[int, float]:
        """Compute boundary flow using persisted (alpha, beta) shares.

        Reads alpha, beta from MP-SPDZ persistence files (written by
        secure_regression_persist). Only the stock value is provided as
        a runtime secret input. The equation parameters never leave the
        secret-shared domain.
        """
        source_party = equation_params["source_party"]
        sink_party = equation_params["sink_party"]
        eq_id = equation_params["eq_id"]
        source_val = stock_values.get(source_party, 0.0)

        # Install program
        program_source = _load_bundled_program("fedsfd_boundary_flow_persist")
        self._install_program("fedsfd_boundary_flow_persist", program_source)

        # Write inputs: only the stock value from the source party
        inputs = {source_party: [source_val]}
        for p in range(self.n_parties):
            if p not in inputs:
                inputs[p] = [0.0]
        self._write_inputs(inputs)

        # Compile with source_party and eq_id
        compile_args = [str(source_party), str(eq_id)]
        self._compile_program("fedsfd_boundary_flow_persist", compile_args)
        output = self._run_program("fedsfd_boundary_flow_persist", compile_args)

        flow_rate = self._parse_boundary_flow_output(output)

        result = {party: 0.0 for party in stock_values}
        result[sink_party] = flow_rate
        return result

    def _clear_persistence_channel(self, eq_id: int) -> None:
        """Clear persistence files for a single channel/eq_id."""
        persist_dir = self.mp_spdz_path / "Player-Data" / "Persistence"
        if not persist_dir.exists():
            return
        for f in persist_dir.glob(f"Transactions-P*-{eq_id}"):
            f.unlink()
            logger.debug("Removed persistence file: %s", f)

    def clear_persisted_params(self, eq_ids: Optional[list] = None) -> None:
        """Delete persisted boundary equation parameters.

        Removes protocol-specific persistence files from
        Player-Data/Persistence/. Call after a federation session
        completes to clean up state.
        """
        persist_dir = self.mp_spdz_path / "Player-Data" / "Persistence"
        if not persist_dir.exists():
            logger.debug("No Persistence directory to clean")
            return

        if eq_ids is None:
            # Clear all persistence files
            count = 0
            for f in persist_dir.glob("Transactions-P*"):
                f.unlink()
                count += 1
            if count:
                logger.info("Cleared %d persistence files", count)
        else:
            for eq_id in eq_ids:
                self._clear_persistence_channel(eq_id)
            logger.info(
                "Cleared persistence for eq_ids=%s", eq_ids
            )


# ---------------------------------------------------------------------------
# Bundled .mpc programs
# ---------------------------------------------------------------------------

# The .mpc programs are stored as Python strings so they can be
# distributed alongside the package without requiring a separate
# mp_spdz/programs/ directory.  They are also saved as standalone
# .mpc files in the repo for reviewer inspection.

_BUNDLED_PROGRAMS: Dict[str, str] = {}


def _load_bundled_program(name: str) -> str:
    """Load a bundled .mpc program by name.

    Always uses the hardcoded source strings embedded in this module,
    which are guaranteed to match the Python input/output protocol.
    The standalone .mpc files in mp_spdz/programs/ are for reviewer
    inspection only — they are NOT loaded at runtime to avoid
    stale-file bugs.
    """
    if name in _BUNDLED_PROGRAMS:
        return _BUNDLED_PROGRAMS[name]

    # Use hardcoded programs (these are the authoritative versions)
    if name == "fedsfd_correlation":
        source = _CORRELATION_MPC
    elif name == "fedsfd_regression":
        source = _REGRESSION_MPC
    elif name == "fedsfd_boundary_flow":
        source = _BOUNDARY_FLOW_MPC
    elif name == "fedsfd_regression_persist":
        source = _REGRESSION_PERSIST_MPC
    elif name == "fedsfd_boundary_flow_persist":
        source = _BOUNDARY_FLOW_PERSIST_MPC
    else:
        raise FileNotFoundError(f"No bundled MPC program named '{name}'")

    _BUNDLED_PROGRAMS[name] = source
    return source


# Hardcoded program sources (identical to the .mpc files in mp_spdz/programs/)

_CORRELATION_MPC = '''\
"""
fedsfd_correlation.mpc — Secure lagged Pearson correlation
"""
sfix.set_precision(16, 31)
print_float_precision(8)

n = int(program.args[1])
max_lag = int(program.args[2])

ts_a = sfix.Array(n)
ts_b = sfix.Array(n)

for i in range(n):
    ts_a[i] = sfix.get_input_from(0)

for i in range(n):
    ts_b[i] = sfix.get_input_from(1)

for lag in range(max_lag + 1):
    eff_n = n - lag
    if eff_n < 3:
        print_ln('LAG %s COV 0 VARA 0 VARB 0 N %s', lag, eff_n)
        continue

    sum_a = sfix(0)
    sum_b = sfix(0)
    for i in range(eff_n):
        sum_a = sum_a + ts_a[i]
        sum_b = sum_b + ts_b[i + lag]

    mean_a = sum_a / eff_n
    mean_b = sum_b / eff_n

    cov = sfix(0)
    var_a = sfix(0)
    var_b = sfix(0)
    for i in range(eff_n):
        da = ts_a[i] - mean_a
        db = ts_b[i + lag] - mean_b
        cov = cov + da * db
        var_a = var_a + da * da
        var_b = var_b + db * db

    print_ln('LAG %s COV %s VARA %s VARB %s N %s',
             lag, cov.reveal(), var_a.reveal(), var_b.reveal(), eff_n)

print_ln('DONE')
'''

_REGRESSION_MPC = '''\
"""
fedsfd_regression.mpc — Secure linear regression (mean-centered)

Uses the mean-centered formulation to avoid fixed-point overflow:
  beta  = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
  alpha = mean_y - beta * mean_x

This keeps intermediate sums proportional to variance (not raw values^2 * n),
avoiding overflow in sfix(16,31) which can only represent ~[-16384, 16384).
"""
sfix.set_precision(16, 31)
print_float_precision(8)

n = int(program.args[1])

x = sfix.Array(n)
y = sfix.Array(n)

for i in range(n):
    x[i] = sfix.get_input_from(0)

for i in range(n):
    y[i] = sfix.get_input_from(1)

# Step 1: compute means
sum_x = sfix(0)
sum_y = sfix(0)
for i in range(n):
    sum_x = sum_x + x[i]
    sum_y = sum_y + y[i]

mean_x = sum_x / n
mean_y = sum_y / n

# Step 2: compute slope via mean-centered sums
cov_xy = sfix(0)
var_x = sfix(0)
for i in range(n):
    dx = x[i] - mean_x
    dy = y[i] - mean_y
    cov_xy = cov_xy + dx * dy
    var_x = var_x + dx * dx

beta = cov_xy / var_x
alpha = mean_y - beta * mean_x

print_ln('RESULT INTERCEPT %s SLOPE %s', alpha.reveal(), beta.reveal())
print_ln('DONE')
'''

_BOUNDARY_FLOW_MPC = '''\
"""
fedsfd_boundary_flow.mpc — Secure boundary flow computation

Party 0 always provides the equation parameters (intercept, slope).
The source party provides the stock value.
If source_party == 0, party 0 provides all three values.

Compile: ./compile.py fedsfd_boundary_flow <source_party>
"""
sfix.set_precision(16, 31)
print_float_precision(8)

source_party = int(program.args[1])

# Party 0 always provides intercept and slope
intercept = sfix.get_input_from(0)
slope = sfix.get_input_from(0)

# Stock value comes from source party
# If source_party == 0, it's the 3rd value from party 0
# If source_party != 0, it's the 1st value from that party
stock_val = sfix.get_input_from(source_party)

flow_rate = intercept + slope * stock_val

is_positive = flow_rate > sfix(0)
flow_rate = is_positive.if_else(flow_rate, sfix(0))

print_ln('RESULT FLOW %s', flow_rate.reveal())
print_ln('DONE')
'''

# --- Persistence variants ---

_REGRESSION_PERSIST_MPC = '''\
"""
fedsfd_regression_persist.mpc — Secure linear regression with persistence

Same regression as fedsfd_regression.mpc, but writes (alpha, beta) as
secret shares to MP-SPDZ persistence files. The parameters never exist
in plaintext — they stay in the secret-shared domain and can be read
back by fedsfd_boundary_flow_persist.mpc.

Compile: ./compile.py fedsfd_regression_persist <n> <eq_id>
"""
sfix.set_precision(16, 31)
print_float_precision(8)

n = int(program.args[1])
eq_id = int(program.args[2])

x = sfix.Array(n)
y = sfix.Array(n)

for i in range(n):
    x[i] = sfix.get_input_from(0)

for i in range(n):
    y[i] = sfix.get_input_from(1)

# Step 1: compute means
sum_x = sfix(0)
sum_y = sfix(0)
for i in range(n):
    sum_x = sum_x + x[i]
    sum_y = sum_y + y[i]

mean_x = sum_x / n
mean_y = sum_y / n

# Step 2: compute slope via mean-centered sums
cov_xy = sfix(0)
var_x = sfix(0)
for i in range(n):
    dx = x[i] - mean_x
    dy = y[i] - mean_y
    cov_xy = cov_xy + dx * dy
    var_x = var_x + dx * dx

beta = cov_xy / var_x
alpha = mean_y - beta * mean_x

# Step 3: persist as secret shares
# Each party writes its share to Player-Data/Persistence/Transactions-P{i}-{eq_id}
alpha.write_to_file(eq_id)
beta.write_to_file(eq_id)

print_ln('PERSIST EQID %s OK', eq_id)

# Step 4: reveal for diagnostics (R-squared on Python side)
print_ln('RESULT INTERCEPT %s SLOPE %s', alpha.reveal(), beta.reveal())
print_ln('DONE')
'''

_BOUNDARY_FLOW_PERSIST_MPC = '''\
"""
fedsfd_boundary_flow_persist.mpc — Boundary flow from persisted parameters

Reads (alpha, beta) from MP-SPDZ persistence files written by
fedsfd_regression_persist.mpc. The parameters never leave the
secret-shared domain — only the resulting flow rate is revealed.

Compile: ./compile.py fedsfd_boundary_flow_persist <source_party> <eq_id>
"""
sfix.set_precision(16, 31)
print_float_precision(8)

source_party = int(program.args[1])
eq_id = int(program.args[2])

# Read persisted equation parameters (secret shares)
alpha = sfix.read_from_file(eq_id)
beta = sfix.read_from_file(eq_id)

# Read stock value from source party (secret input)
stock_val = sfix.get_input_from(source_party)

# Compute boundary flow in the secret domain
flow_rate = alpha + beta * stock_val

# Clamp to non-negative
is_positive = flow_rate > sfix(0)
flow_rate = is_positive.if_else(flow_rate, sfix(0))

# Only the flow rate is revealed
print_ln('RESULT FLOW %s', flow_rate.reveal())
print_ln('DONE')
'''