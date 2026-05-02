# Experiment Results

I ran all four C-MAPSS subsets on RunPod using:

```bash
bash scripts/run_all_fd_experiment.sh
```

Each model was trained for 30 epochs with a window size of 30.

## Test RMSE

Lower is better.

| Subset | LSTM | TCN | DLinear-style |
|---|---:|---:|---:|
| FD001 | 15.78 | 14.73 | 21.76 |
| FD002 | 15.41 | 16.05 | 20.21 |
| FD003 | 13.27 | 15.30 | 16.02 |
| FD004 | 17.19 | 19.87 | 25.31 |

## Main observations

- TCN did best on FD001, the easiest subset.
- LSTM was strongest on FD002, FD003, and FD004.
- DLinear-style was much faster and much smaller, but its RMSE was worse on every subset.
- FD004 was harder than FD001, which makes sense because FD004 has multiple operating conditions and multiple fault modes.

## Output files

- `outputs_FD001/metrics.csv`
- `outputs_FD002/metrics.csv`
- `outputs_FD003/metrics.csv`
- `outputs_FD004/metrics.csv`

Each output folder also includes loss curves, prediction plots, checkpoints, and the sensor occlusion analysis figure.
