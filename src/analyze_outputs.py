from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import FEATURE_COLUMNS, load_cmapss_subset
from train import build_model, get_device, rmse


@torch.no_grad()
def predict(model: nn.Module, sequences: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(sequences, batch_size=128, shuffle=False)
    for x in loader:
        preds.append(model(x.to(device)).detach().cpu().numpy())
    return np.concatenate(preds).reshape(-1)


def plot_metric_bars(metrics: pd.DataFrame, out_path: Path, subset: str) -> None:
    ordered = metrics.sort_values("test_rmse")
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(ordered["model"], ordered["test_rmse"], color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Test RMSE")
    plt.title(f"{subset} model comparison")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_feature_occlusion(model_name: str, data_dir: Path, output_dir: Path, window_size: int, subset: str) -> None:
    device = get_device()
    datasets = load_cmapss_subset(data_dir, subset=subset, window_size=window_size)
    checkpoint_path = output_dir / "checkpoints" / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name, datasets.feature_dim, checkpoint.get("window_size", window_size)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    sequences = datasets.test_dataset.sequences
    targets = datasets.test_dataset.targets.numpy().reshape(-1)
    baseline_pred = predict(model, sequences, device)
    baseline_rmse = rmse(targets, baseline_pred)

    rows = []
    for i, feature_name in enumerate(FEATURE_COLUMNS):
        occluded = sequences.clone()
        occluded[:, :, i] = 0.0
        occluded_pred = predict(model, occluded, device)
        occluded_rmse = rmse(targets, occluded_pred)
        rows.append(
            {
                "model": model_name,
                "feature": feature_name,
                "baseline_rmse": baseline_rmse,
                "occluded_rmse": occluded_rmse,
                "rmse_increase": occluded_rmse - baseline_rmse,
            }
        )

    rows = sorted(rows, key=lambda row: row["rmse_increase"], reverse=True)
    csv_path = output_dir / "feature_importance_occlusion.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top = rows[:12]
    plt.figure(figsize=(8, 5))
    plt.barh([row["feature"] for row in reversed(top)], [row["rmse_increase"] for row in reversed(top)])
    plt.xlabel("RMSE increase after feature occlusion")
    plt.title(f"Top sensor/setting importance by occlusion ({model_name.upper()})")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "feature_importance_occlusion.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create model comparison and feature-importance analysis plots.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--fd", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--importance-model", default="lstm", choices=["lstm", "tcn", "dlinear"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.output_dir / "metrics.csv")
    plot_metric_bars(metrics, args.output_dir / "figures" / "model_rmse_comparison.png", args.fd)
    run_feature_occlusion(args.importance_model, args.data_dir, args.output_dir, args.window_size, args.fd)
    print(f"Saved analysis outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
