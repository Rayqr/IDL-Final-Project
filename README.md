# IDL Final Project: C-MAPSS RUL Prediction

This repo is for my 24-788 Introduction to Deep Learning mini-project. I use the NASA C-MAPSS turbofan engine degradation dataset, starting with FD001 as suggested in the project handout.

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
scripts/run_full_experiment.sh
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

The full script currently runs 30 epochs for all three models, then runs the extra analysis.

If I only want to test that the code works:

```bash
python src/train.py --epochs 3
python src/analyze_outputs.py --importance-model lstm
```

## Main commands

Train all models:

```bash
python src/train.py --epochs 30 --window-size 30 --batch-size 128
```

Train only one or two models while debugging:

```bash
python src/train.py --epochs 3 --models dlinear
python src/train.py --epochs 3 --models lstm tcn
```

Run analysis after training:

```bash
python src/analyze_outputs.py --importance-model lstm
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

The checked-in outputs are from a short 3-epoch sanity run. For final report numbers, rerun the full command on RunPod and use the regenerated files.

## Dataset

FD001 contains 100 training engine trajectories and 100 test trajectories. Each row has:

- unit id
- cycle index
- 3 operating settings
- 21 sensor measurements

The training trajectories run until failure. The test trajectories stop before failure, and the target RUL values are provided separately.

## References

- Saxena, A., Goebel, K., Simon, D., and Eklund, N. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." PHM, 2008.
- Bai, S., Kolter, J. Z., and Koltun, V. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." 2018.
- Zeng, A., Chen, M., Zhang, L., and Xu, Q. "Are Transformers Effective for Time Series Forecasting?" AAAI, 2023.
