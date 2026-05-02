# IDL Final Project: C-MAPSS RUL Prediction

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
requirements.txt
```

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

## Outputs used in the report

```text
outputs/metrics.csv
outputs/figures/loss_curves.png
outputs/figures/model_rmse_comparison.png
outputs/figures/feature_importance_occlusion.png
outputs/figures/lstm_test_predictions.png
outputs/figures/tcn_test_predictions.png
outputs/figures/dlinear_test_predictions.png
outputs/feature_importance_occlusion.csv
outputs/checkpoints/lstm_best.pt
outputs/checkpoints/tcn_best.pt
outputs/checkpoints/dlinear_best.pt
```

The `outputs/` folder is from a short FD001 sanity run. The `outputs_FD001/` through `outputs_FD004/` folders contain the full C-MAPSS run from RunPod. For the report, use the `outputs_FD00*/metrics.csv` files and the figures inside each corresponding `outputs_FD00*/figures/` folder.

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
