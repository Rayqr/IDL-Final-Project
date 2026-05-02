# IDL Final Project

This repo is for my 24-788 Introduction to Deep Learning mini-project. I use the NASA C-MAPSS turbofan engine degradation dataset. I start with FD001 as suggested in the project handout, and the code can also run FD001-FD004 for the full C-MAPSS comparison.

The task is to predict remaining useful life (RUL) from a window of engine sensor readings.

## What I am comparing

| Role | Model | Reason |
|---|---|---|
| Baseline | LSTM | Natural sequence baseline for sensor time series |
| Variant 1 | TCN | Uses dilated 1D convolutions instead of recurrence |
| Variant 2 | DLinear-style model | Uses trend/residual decomposition with a small linear model |

Main metric: RMSE on the FD001 test set. I also report the PHM asymmetric score, parameter count, and training time.

I also included a simple sensor occlusion analysis to see which sensor channels matter most for the trained LSTM.

## Current results

These are the test RMSE numbers from the RunPod run with 30 epochs:

| Subset | LSTM | TCN | DLinear-style |
|---|---:|---:|---:|
| FD001 | 15.78 | 14.73 | 21.76 |
| FD002 | 15.41 | 16.05 | 20.21 |
| FD003 | 13.27 | 15.30 | 16.02 |
| FD004 | 17.19 | 19.87 | 25.31 |

The full metrics, including PHM score, parameter count, and training time, are in `outputs/FD001/metrics.csv` through `outputs/FD004/metrics.csv`. Overall, the LSTM was the most stable across the harder subsets, while the TCN was best on FD001. The DLinear-style model was much smaller and faster, but it lost accuracy on every subset.

## Files

The main training code is in `src/train.py`, and the three model definitions are in `src/models.py`. Data loading and sliding-window preprocessing are in `src/data.py`. I used `src/analyze_outputs.py` after training to make the RMSE comparison plots and feature occlusion analysis.

For the experiments, I used:

- `scripts/run_all_fd_experiment.sh` for FD001-FD004

The report source is in `final_report.tex`, with citations in `references.bib`.

## Run on RunPod

```bash
git clone https://github.com/Rayqr/IDL-Final-Project.git
cd IDL-Final-Project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/run_full_experiment.sh
```

The full script currently runs FD001 for 30 epochs for all three models, then runs the extra analysis.

To run all four C-MAPSS subsets:

```bash
bash scripts/run_all_fd_experiment.sh
```

## Output folders

The final RunPod results are organized by subset:

```text
outputs/FD001/
outputs/FD002/
outputs/FD003/
outputs/FD004/
```

For each subset, I mainly used:

```text
metrics.csv
figures/loss_curves.png
figures/model_rmse_comparison.png
figures/feature_importance_occlusion.png
```

## Dataset

C-MAPSS has four subsets:

| Subset | Train trajectories | Test trajectories | Conditions | Fault modes |
|---|---:|---:|---:|---:|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 248 | 249 | 6 | 2 |

FD001 is the smallest/easiest subset. FD004 is the hardest because it has multiple operating conditions and multiple fault modes.

Each row has:

- unit id
- cycle index
- 3 operating settings
- 21 sensor measurements

The training trajectories run until failure. The test trajectories stop before failure, and the target RUL values are provided separately.

## References

- Saxena, A., Goebel, K., Simon, D., and Eklund, N. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." PHM, 2008.
- Bai, S., Kolter, J. Z., and Koltun, V. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." 2018.
- Zeng, A., Chen, M., Zhang, L., and Xu, Q. "Are Transformers Effective for Time Series Forecasting?" AAAI, 2023.
