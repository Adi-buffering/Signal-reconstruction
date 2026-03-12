# Reproducible linear reconstruction pipeline

## Requirements
- Python 3.10+
- No external Python dependencies (stdlib-only implementation)

## Run
From repository root:

```bash
python submission/src/train_linear.py --data-root . --out-dir submission
```

## Outputs
- Metrics: `submission/metrics.json`
- Figures:
  - `submission/figures/residual_distribution.svg`
  - `submission/figures/residual_vs_target.svg`
- Report: `submission/report.md`

## Runtime
On this environment, full run was ~70 seconds (data loading + fitting + plotting).

## Notes
- Script performs format checks and fails with explicit errors for unexpected tensor shapes/keys.
- The dataset exposes 7 low-gain samples per row in `X[row][1]`.
