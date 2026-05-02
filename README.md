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

The full metrics, including PHM score, parameter count, and training time, are in `outputs_FD001/metrics.csv` through `outputs_FD004/metrics.csv`.

## Repo layout

```text
data/raw/                  # C-MAPSS data files
src/data.py                # reading files, RUL labels, scaling, sliding windows
src/models.py              # LSTM, TCN, DLinear-style models
src/train.py               # training and evaluation
src/analyze_outputs.py     # model comparison plot and feature occlusion analysis
scripts/download_data.py   # downloads/unzips C-MAPSS if needed
scripts/run_full_experiment.sh      # FD001
scripts/run_all_fd_experiment.sh    # FD001-FD004
outputs/                   # metrics, figures, and checkpoints
RESULTS.md
final_report.tex
references.bib
requirements.txt
```

`final_report.tex` is the Overleaf/LaTeX report draft. Before submitting, replace the name, department, and Andrew email in the author block.

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

This writes separate folders:

```text
outputs_FD001/
outputs_FD002/
outputs_FD003/
outputs_FD004/
```

If I only want to test that the code works:

```bash
python src/train.py --epochs 3
python src/analyze_outputs.py --importance-model lstm
```

## Main commands

Train all models:

```bash
python src/train.py --fd FD001 --epochs 30 --window-size 30 --batch-size 128
```

Train only one or two models while debugging:

```bash
python src/train.py --epochs 3 --models dlinear
python src/train.py --epochs 3 --models lstm tcn
```

Run analysis after training:

```bash
python src/analyze_outputs.py --fd FD001 --importance-model lstm
```

## Output folders

`outputs/` is just my quick FD001 test run, so I did not use it for the final numbers.

The actual RunPod results are in:

```text
outputs_FD001/
outputs_FD002/
outputs_FD003/
outputs_FD004/
```

For each subset, I mainly used:

```text
metrics.csv
figures/loss_curves.png
figures/model_rmse_comparison.png
figures/feature_importance_occlusion.png
```

I left the checkpoints and prediction plots in the same folders too, mostly so I could double-check results later without rerunning the full experiment.

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
