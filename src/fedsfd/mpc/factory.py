"""
MPC backend factory.

Creates the appropriate MPCBackend implementation based on configuration.
"""

from __future__ import annotations

from typing import Optional

from fedsfd.mpc.interface import MPCBackend


def create_mpc_backend(
    backend_type: str = "mock",
    mp_spdz_path: Optional[str] = None,
    protocol: str = "semi2k",
    n_parties: int = 2,
    config: Optional[dict] = None,
) -> MPCBackend:
    """Create and initialize an MPC backend.

    Parameters
    ----------
    backend_type : str
        "mock" for plaintext local backend, "mp_spdz" for real MP-SPDZ.
    mp_spdz_path : str, optional
        Path to MP-SPDZ installation (required for mp_spdz backend).
    protocol : str
        MPC protocol (default: "semi2k").
    n_parties : int
        Number of parties.
    config : dict, optional
        Additional backend-specific config.

    Returns
    -------
    MPCBackend
        Initialized backend instance.
    """
    if config is None:
        config = {}

    if backend_type == "mock":
        from fedsfd.mpc.local_mock import LocalMockBackend

        backend = LocalMockBackend()
        backend.setup(n_parties, config)
        return backend

    elif backend_type == "mp_spdz":
        from fedsfd.mpc.mp_spdz import MPSPDZBackend

        backend = MPSPDZBackend(
            mp_spdz_path=mp_spdz_path,
            protocol=protocol,
        )
        setup_config = dict(config)
        setup_config["mp_spdz_path"] = mp_spdz_path
        setup_config["protocol"] = protocol
        backend.setup(n_parties, setup_config)
        return backend

    else:
        raise ValueError(
            f"Unknown MPC backend type: '{backend_type}'. "
            f"Supported: 'mock', 'mp_spdz'"
        )


def create_backend_from_experiment_config(experiment_config) -> MPCBackend:
    """Create an MPC backend from an ExperimentConfig.

    Parameters
    ----------
    experiment_config : ExperimentConfig
        The experiment configuration (from config.py).

    Returns
    -------
    MPCBackend
    """
    return create_mpc_backend(
        backend_type=experiment_config.mpc.backend,
        mp_spdz_path=experiment_config.mpc.mp_spdz_path,
        protocol=experiment_config.mpc.protocol,
        n_parties=len(experiment_config.organizations),
    )