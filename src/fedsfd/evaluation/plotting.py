"""
Trajectory comparison plots for federated SFD evaluation.

Generates one figure per stock variable showing actual vs simulated
trajectories over the holdout period, with R² annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from fedsfd.sfd.model import SFD


def plot_trajectory_comparisons(
    actual_ts: Dict[str, Dict[str, np.ndarray]],
    simulated_trajectories: Dict[str, Dict[str, np.ndarray]],
    local_sfds: Dict[str, SFD],
    output_dir: Path,
    filename_prefix: str = "eval_trajectory",
) -> List[Path]:
    """Generate per-stock trajectory comparison plots.

    Parameters
    ----------
    actual_ts : dict of {org_name: {var_name: np.ndarray}}
        Actual holdout time series.
    simulated_trajectories : dict of {label: {traj_key: np.ndarray}}
        Named simulation trajectories to plot.
        E.g., {"Federated": fed_traj, "Local-only": local_traj, "One-step-ahead": osa_traj}
    local_sfds : dict of {org_name: SFD}
        Per-org SFDs (to identify stock variables).
    output_dir : Path
        Directory to save plots.
    filename_prefix : str
        Prefix for output filenames.

    Returns
    -------
    list of Path
        Paths to saved plot files.
    """
    if not HAS_MPL:
        print("  WARNING: matplotlib not installed, skipping trajectory plots")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for org_name, sfd in local_sfds.items():
        for stock in sfd.stocks:
            stock_key = f"{org_name.lower()}_{stock.name}"
            org_ts = actual_ts.get(org_name, {})
            actual = org_ts.get(stock.name)
            if actual is None:
                continue

            n = len(actual)
            t_axis = np.arange(n)

            fig, ax = plt.subplots(figsize=(10, 4))

            # Plot actual
            ax.plot(t_axis, actual, color="black", linewidth=2,
                    label="Actual", zorder=10)

            # Plot each simulation
            colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]
            linestyles = ["-", "--", ":", "-.", "-"]

            for i, (label, traj_dict) in enumerate(simulated_trajectories.items()):
                sim = traj_dict.get(stock_key)
                if sim is None:
                    continue

                # Align: stock trajectories have n+1 entries (including initial)
                sim_aligned = sim[1:n + 1] if len(sim) > n else sim[:n]
                m = min(len(sim_aligned), n)

                # Compute R²
                errors = sim_aligned[:m] - actual[:m]
                ss_res = np.sum(errors ** 2)
                ss_tot = np.sum((actual[:m] - np.mean(actual[:m])) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
                r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"

                color = colors[i % len(colors)]
                ls = linestyles[i % len(linestyles)]
                ax.plot(t_axis[:m], sim_aligned[:m], color=color,
                        linewidth=1.5, linestyle=ls,
                        label=f"{label}") # (R²={r2_str})

            ax.set_xlabel("Holdout time step (days)")
            ax.set_ylabel("Value")
            ax.set_title(f"{stock_key}", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            fig.tight_layout()
            fname = f"{filename_prefix}_{stock_key}.pdf"
            fpath = output_dir / fname
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(fpath)

    return saved
