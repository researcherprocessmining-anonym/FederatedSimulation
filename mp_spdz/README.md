# MP-SPDZ Integration

This directory contains the MP-SPDZ programs used for secure multi-party
computation in the federated SFD pipeline.

## Setup

### 1. Install MP-SPDZ

Clone and build MP-SPDZ alongside the `federated-sfd/` directory:

```bash
cd ~/code/federatedSimulation  # or your parent directory
git clone https://github.com/data61/MP-SPDZ.git
cd MP-SPDZ
make -j$(nproc) semi-party.x
```

The `semi-party.x` binary implements the `semi2k` protocol (semi-honest,
arithmetic mod 2^k), which is the default for this project.

### 2. Configure the Path

In your experiment config YAML (e.g., `configs/logistics_3org.yaml`), set:

```yaml
mpc:
  backend: "mp_spdz"
  mp_spdz_path: "/path/to/MP-SPDZ"
  protocol: "semi2k"
```

### 3. Run with MP-SPDZ

```bash
python experiments/04_discover_federation.py configs/logistics_3org.yaml
python experiments/05_simulate_federated.py configs/logistics_3org.yaml
```

When `backend: "mp_spdz"` is set, the pipeline will:
1. Copy .mpc programs into `MP-SPDZ/Programs/Source/`
2. Write party inputs to `MP-SPDZ/Player-Data/Input-P{i}-0`
3. Compile programs with `./compile.py -R 64`
4. Run all parties on localhost via `Scripts/semi2k.sh`
5. Parse results from stdout

## Programs

### `fedsfd_correlation.mpc`

Computes lagged Pearson correlation between two secret time series.
Each party inputs its time series; the program reveals only the summary
statistics (covariance, variances) at each lag, not the raw data.

**Compile:** `./compile.py -R 64 fedsfd_correlation <n> <max_lag>`

### `fedsfd_regression.mpc`

Fits a linear regression `y = α + β·x` where x and y are held by
different parties. Uses the OLS normal equations computed over secret
values. Only the coefficients (α, β) are revealed.

**Compile:** `./compile.py -R 64 fedsfd_regression <n>`

### `fedsfd_boundary_flow.mpc`

Evaluates a boundary flow equation `flow = max(α + β·stock, 0)` where
the stock value is a secret input from the source party and the equation
parameters are public. Used during federated simulation at each time step.

**Compile:** `./compile.py -R 64 fedsfd_boundary_flow <source_party> <intercept> <slope>`

## Security Model

- **Protocol:** `semi2k` (semi-honest / honest-but-curious)
- **Threat model:** All parties follow the protocol correctly but may try
  to learn private information from the messages they observe.
- **What is revealed:**
  - Correlation coefficients (scalar per lag per flow pair)
  - Regression coefficients (two scalars per boundary equation)
  - Boundary flow rates (one scalar per equation per time step)
- **What stays private:**
  - Raw time series data (WIP, throughput, arrival rates)
  - Internal stock trajectories during simulation

## Fixed-Point Precision

All programs use `sfix` with 16 fractional bits and 31 total bits:
- Range: approximately [-16384, 16384)
- Precision: approximately 0.000015
- This is sufficient for the logistics dataset values (typically 0-500)

To adjust, modify `sfix.set_precision(16, 31)` in the .mpc programs.

## Troubleshooting

- **"Protocol binary not found"**: Run `make -j$(nproc) semi-party.x`
  in the MP-SPDZ directory.
- **Compilation errors**: Ensure you compile with `-R 64` (not `-F`),
  since `semi2k` operates modulo 2^k.
- **Incorrect results with sfix**: Check that input values are within the
  representable range ([-16384, 16384) with default precision).
- **Timeout**: For large time series (n > 100), compilation may take a
  few minutes. Increase the timeout in `mp_spdz.py` if needed.