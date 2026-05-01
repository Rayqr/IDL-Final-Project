# NASA C-MAPSS FD001 RUL Prediction

This is a small 24-788 mini-project experiment for one student:

- Dataset: NASA C-MAPSS FD001 turbofan degradation data
- Task: predict remaining useful life (RUL) from multivariate sensor time series
- Baseline: LSTM
- Variant 1: Temporal Convolutional Network (TCN)
- Variant 2: DLinear-style decomposed linear time-series model
- Metrics: RMSE and PHM asymmetric score
- Extra analysis: parameter count, training time, model comparison plot, and sensor occlusion importance

The official NASA page describes C-MAPSS as multivariate engine time series with train/test splits and true RUL labels for the test set. FD001 has 100 train trajectories, 100 test trajectories, one operating condition, and one fault mode.

## Files

```text
cmapss_rul_project/
  data/raw/                 # CMAPSSData.zip and extracted FD001 files
  scripts/download_data.py  # optional data downloader
  src/data.py               # parsing, scaling, windowing
  src/models.py             # LSTM, TCN, and DLinear-style models
  src/train.py              # training, evaluation, plots, checkpoints
  src/analyze_outputs.py    # extra analysis plots for the report
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

By default this trains the solo extra-credit scope: one baseline plus two model variants.
You can also train a subset while debugging:

```bash
python src/train.py --epochs 3 --models dlinear
python src/train.py --epochs 3 --models lstm tcn
```

Then run the extra analysis:

```bash
python src/analyze_outputs.py --importance-model lstm
```

Generated outputs:

```text
outputs/metrics.csv
outputs/metrics.json
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

Use `outputs/metrics.csv` for the main results table and `outputs/figures/loss_curves.png` as the required training curve figure. Use `outputs/figures/model_rmse_comparison.png` and `outputs/figures/feature_importance_occlusion.png` as additional analysis figures.

## Suggested Report Claim

The LSTM baseline models temporal degradation recursively through hidden states. The TCN variant uses dilated causal convolutions to capture temporal context with more parallel computation. The DLinear-style variant decomposes the sensor window into trend and residual components, then uses simple linear projections before a small regression head. The experiment tests whether convolutional temporal modeling or decomposition-based linear modeling can match or improve RUL prediction on FD001 while training efficiently.

## Citation Notes

Cite the NASA C-MAPSS dataset and the PHM 2008 paper referenced by NASA:

Saxena, A., Goebel, K., Simon, D., and Eklund, N. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." Proceedings of the 1st International Conference on Prognostics and Health Management, 2008.

For the TCN variant, cite:

Bai, S., Kolter, J. Z., and Koltun, V. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." 2018.

For the DLinear-style variant, cite:

Zeng, A., Chen, M., Zhang, L., and Xu, Q. "Are Transformers Effective for Time Series Forecasting?" Proceedings of the AAAI Conference on Artificial Intelligence, 2023.
