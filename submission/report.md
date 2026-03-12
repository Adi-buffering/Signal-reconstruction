# Linear signal reconstruction submission

## Problem summary
This repository contains train/validation/test shards under `train/train`, `val/val`, `test/test`, plus the main task statement `Test_hsf_atlas.pdf` and supporting references (`fcurcio_EuCAIFCon.pdf`, `Oliveira_Gonçalves_2020_J._Phys.__Conf._Ser._1525_012092.pdf`, `tilecal-2005-001.pdf`).

The implemented goal is linear reconstruction of low-gain energy (`ene_lo`) from low-gain samples in each event window.

## Repository inspection findings
- Serialization format: each `.pt` shard is a torch zip-serialized dictionary with keys `X`, `y`, `y_OF`.
- Per-row structure found in the data:
  - `X[row]` has 2 gain channels, each with 7 samples.
  - `y[row]` has 2 targets (z-score normalized using `y_stats.npz`).
  - `y_OF[row]` has 2 optimal-filter-related values.
- `y_stats.npz` contains `mean.npy` and `std.npy` (shape `(1,2)`), used to denormalize `y`.

**Important caveat:** the provided tensors expose **7** low-gain samples per row, not 8. The code validates this explicitly and raises a clear error if shapes deviate.

## Method
- Extract low-gain input vector from each row: `x = X[row][1]`.
- Target: `ene_lo = y[row][1] * std[1] + mean[1]` (denormalized).
- Fit linear ridge model with optional bias:
  \[
  \hat{y} = b + \sum_{i=0}^{6} w_i x_i
  \]
- Hyperparameter selection on validation split only:
  - candidate alphas: `[0, 1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100]`
  - selection criterion: minimum validation RMS of residual ratio.

This is consistent with linear optimal-filter/Wiener-style reconstruction principles from the supporting papers, while remaining strictly linear in the raw input samples.

## Numerical stability policy
Metric residual ratio:
\[
 r = \frac{\hat{y} - y}{y}
\]
with signed epsilon guard for near-zero targets:
- if `|y| > eps`, use denominator `y`
- otherwise use `sign(y) * eps`

This prevents division-by-zero while keeping a signed residual convention.

## Model selected on validation
Best alpha: `0.1`

Final reconstruction formula used:
\[
\hat{ene}_{lo} = 0.0331313032
-233.2523620\,x_0
+1333.4958370\,x_1
-3284.1641944\,x_2
+6363.7804198\,x_3
-2547.0407158\,x_4
+955.7674570\,x_5
-276.8249020\,x_6
\]

## Results
Validation:
- mean residual ratio: `0.1120641143`
- RMS residual ratio: `9.9746320676`

Test:
- mean residual ratio: `0.0708084928`
- RMS residual ratio: `9.6883180264`

## Generated outputs
- `submission/metrics.json`
- `submission/figures/residual_distribution.svg`
- `submission/figures/residual_vs_target.svg`

## Limitations and next steps
- The repository data provides 7-sample windows. If an 8-sample definition is mandatory, raw waveform-level data with the extra index is required.
- Additional physically motivated linear constraints (e.g., unbiasedness constraints) could be evaluated as future work.
