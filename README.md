# Federated SFD

Federated process simulation using Stock-Flow Diagrams (SFDs) discovered from object-centric event logs.

## Overview

Multiple organizations each build a local SFD from their slice of an OCEL 2.0 event log, discover inter-organizational flows via secure multi-party computation (MPC), and run federated what-if simulations.

## Dataset

OCEL 2.0 Logistics event log.

**To use the logistics dataset:**
```bash
bash data/download.sh
```
Downloads the Container Logistics OCEL 2.0 log from [Zenodo (DOI: 10.5281/zenodo.8428084)](https://zenodo.org/records/8428084).


## Setup

```bash
pip install -e ".[dev]"
```

#### MPC Backend

The federated protocol uses secure multi-party computation (MPC) for cross-organizational flow matching and boundary equation discovery. Two backends are available, configured via the `mpc.backend` field in the config YAML:

- **`mock`** (default) — a local mock that emulates MPC semantics without cryptographic overhead. Suitable for development and reproducing results without installing MP-SPDZ.
- **`mp_spdz`** — uses [MP-SPDZ](https://github.com/data61/MP-SPDZ) for real cryptographic MPC. To use it:
  1. Clone and build MP-SPDZ following the [MP-SPDZ README](https://github.com/data61/MP-SPDZ#readme).
  2. Set `mpc.backend: "mp_spdz"` and `mpc.mp_spdz_path` to your MP-SPDZ installation directory in the config file.

## System Requirements

- **Python** 3.10+
- **OS:** Linux or macOS (tested on macOS 14 and Ubuntu 22.04)
- **RAM:** ~2 GB (dataset + pandas operations)
- **GPU:** Not required — all computation is CPU-based
- **Estimated runtime:** ~2–5 minutes for the full reproduction pipeline (mock MPC backend, single machine). The MP-SPDZ backend adds network and cryptographic overhead depending on the deployment.

## Reproducing Paper Results

A single script runs all experiments:

```bash
python experiments/reproduce.py

# Or via Make
make reproduce
```

**Step-by-step:**

1. Install dependencies: `pip install -e ".[dev]"`
2. Download the dataset: `bash data/download.sh`
3. Run all experiments: `make reproduce`
4. Outputs are written to `results/`

### Experiments and Output-to-Paper Mapping

The reproduction script runs three experiments sequentially:

#### Experiment 1 — What-If Simulation

Runs federated simulation under a baseline scenario and a what-if scenario (+43% demand surge), comparing SFD-simulated trajectories against CPN ground truth.

| Output file | Description | 
|---|---|
| `fed_trajectories_real_baseline.csv` | CPN ground-truth stock trajectories (baseline) |
| `fed_trajectories_sim_baseline.csv` | SFD-simulated stock trajectories (baseline) |
| `fed_trajectories_real_whatif.csv` | CPN "ground-truth" stock trajectories (what-if) |
| `fed_trajectories_sim_whatif.csv` | SFD-simulated stock trajectories (what-if) |
| `fed_four_way_comparison.csv` | Combined 4-way stock comparison (all scenarios) | 
| `fed_four_way_comparison.pdf` | 4-way comparison plot | 

#### Experiment 2 — Some metrics comparing Federated vs Local vs CPN 

Compares prediction accuracy of federated SFD simulation, local-only simulation (no cross-org flows), and one-step-ahead baseline, all evaluated against CPN ground truth.

| Output file | Description |
| `eval_metrics_federated.csv` | Per-stock RMSE, MAE, R² (federated) | 
| `eval_metrics_local.csv` | Per-stock RMSE, MAE, R² (local-only) |
| `eval_trajectory_comparison.csv` | Time-step-level trajectories for all methods | 
| `eval_equation_quality.csv` | Per-flow R² of discovered equations |
| `results/plots/*.pdf` | Per-stock trajectory comparison plots |

#### Experiment 3 — Scalability / MPC Overhead (Paper Section 5.3 / Table 5)

Measures wall-clock time of federated simulation, decomposed into MPC boundary-flow computation vs local internal-flow computation, across varying simulation horizons.

| Output file | Description | 
|---|---|
| `eval_scalability.csv` | Timing breakdown per simulation horizon | 

### other artefacts

The other experimental files produce these files, useful for inspection:

| Output file | Description |
|---|---|
| `events_*.csv` | Partitioned event logs per organization |
| `all_events.csv` | Full event log export |
| `variables_*.csv` | Aggregated SFD time-series per organization |
| `sfd_variables.csv` | All SFD variables across organizations |
| `sfd_*.mdl` | Vensim-compatible SFD model files (per org + federated) |
| `sfd_*_params.txt` | Human-readable SFD equation parameters |
| `trajectories_*.csv` | Simulated trajectories per organization |

- **`.mdl` files**: Vensim-compatible model definitions. These can be opened in [Vensim](https://vensim.com/) or any compatible system dynamics tool to inspect the discovered stock-flow structure.

## Configuration

All experiment parameters are in `configs/logistics_3org.yaml`. The 3-organization partition splits the logistics process into:

1. **Company** — order registration, transport planning
2. **Trucking** — container pickup, transport to terminal
3. **Terminal** — receiving, storage, loading, departures

## Data

The `data/whatIF/` directory contains pre-generated CPN simulation outputs (baseline and what-if scenarios) used as ground truth for evaluation. These were produced using a CPN Tools model of the same logistics process, with a 43% demand increase applied in the what-if scenario.

The raw CPN data is provided as CSV files. To convert them to the SQLite format required by the pipeline, run the `data/whatIF/csv_to_sql.ipynb` notebook.

## Project Structure

```
configs/          YAML experiment configurations
data/             Dataset (downloaded or synthetic)
  whatIF/          CPN ground-truth simulation data (baseline + what-if)
src/fedsfd/       Source code
  ocel/           OCEL loading, partitioning, scoping
  sfd/            SFD aggregation, discovery, simulation
  federation/     Cross-org flow matching, boundary equations
  mpc/            MPC backends (mock + MP-SPDZ)
  evaluation/     Metrics and export
  utils/          Config loader
experiments/      Experiment scripts
  reproduce.py    Reproduce all paper experiments
results/          Output CSVs, plots, and .mdl files
tests/            Unit tests
mp_spdz/          MP-SPDZ integration programs and documentation
```

