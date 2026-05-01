# NASA C-MAPSS FD001 RUL Prediction

This is a small 24-788 mini-project experiment for one student:

- Dataset: NASA C-MAPSS FD001 turbofan degradation data
- Task: predict remaining useful life (RUL) from multivariate sensor time series
- Baseline: LSTM
- Variant: Temporal Convolutional Network (TCN)
- Metrics: RMSE and PHM asymmetric score

The official NASA page describes C-MAPSS as multivariate engine time series with train/test splits and true RUL labels for the test set. FD001 has 100 train trajectories, 100 test trajectories, one operating condition, and one fault mode.

## Files

```text
cmapss_rul_project/
  data/raw/                 # CMAPSSData.zip and extracted FD001 files
  scripts/download_data.py  # optional data downloader
  src/data.py               # parsing, scaling, windowing
  src/models.py             # LSTM and TCN models
  src/train.py              # training, evaluation, plots, checkpoints
  outputs/                  # generated metrics, figures, checkpoints
  requirements.txt
```

## Setup

Create an environment and install dependencies:

```bash
cd "/Users/raymondye/Documents/New project/cmapss_rul_project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the dataset files are missing, download them:

```bash
python scripts/download_data.py
```

The dataset has already been downloaded in this project folder:

```text
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

## Quick Smoke Run

Use 3 epochs to verify the pipeline quickly:

```bash
python src/train.py --epochs 3
```

## Report Run

Use 20-50 epochs for report-quality curves:

```bash
python src/train.py --epochs 30 --window-size 30 --batch-size 128
```

Generated outputs:

```text
outputs/metrics.csv
outputs/metrics.json
outputs/figures/loss_curves.png
outputs/figures/lstm_test_predictions.png
outputs/figures/tcn_test_predictions.png
outputs/checkpoints/lstm_best.pt
outputs/checkpoints/tcn_best.pt
```

Use `outputs/metrics.csv` for the main results table and `outputs/figures/loss_curves.png` as the required training curve figure.

## Suggested Report Claim

The LSTM baseline models temporal degradation recursively through hidden states, while the TCN variant uses dilated causal convolutions to capture temporal context with more parallel computation. The experiment tests whether convolutional temporal modeling can match or improve RUL prediction on FD001 while training efficiently.

## Citation Notes

Cite the NASA C-MAPSS dataset and the PHM 2008 paper referenced by NASA:

Saxena, A., Goebel, K., Simon, D., and Eklund, N. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." Proceedings of the 1st International Conference on Prognostics and Health Management, 2008.

For the TCN variant, cite:

Bai, S., Kolter, J. Z., and Koltun, V. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." 2018.
