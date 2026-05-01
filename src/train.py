from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import load_fd001
from models import DLinearRegressor, LSTMRegressor, TCNRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    scores = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(scores))


def run_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    preds = []
    targets = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        losses.append(float(criterion(pred, y).detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(preds).reshape(-1)
    y_true = np.concatenate(targets).reshape(-1)
    return float(np.mean(losses)), y_true, y_pred


def build_model(name: str, input_dim: int, window_size: int) -> nn.Module:
    if name == "lstm":
        return LSTMRegressor(input_dim=input_dim, hidden_dim=64, num_layers=1)
    if name == "tcn":
        return TCNRegressor(input_dim=input_dim, channels=(32, 32, 64), kernel_size=3, dropout=0.1)
    if name == "dlinear":
        return DLinearRegressor(input_dim=input_dim, window_size=window_size, moving_avg=7)
    raise ValueError(f"Unknown model: {name}")


def plot_loss(histories: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.plot(history["train_loss"], label=f"{name} train")
        plt.plot(history["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("FD001 training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, out_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    lims = [0, max(float(y_true.max()), float(y_pred.max())) + 5]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"{model_name.upper()} test predictions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_one_model(name: str, datasets, args, device: torch.device):
    train_loader = DataLoader(datasets.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets.val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets.test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(name, datasets.feature_dim, args.window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val = float("inf")
    ckpt_path = args.output_dir / "checkpoints" / f"{name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val, p_val = evaluate(model, val_loader, criterion, device)
        val_rmse = rmse(y_val, p_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": name,
                    "input_dim": datasets.feature_dim,
                    "window_size": args.window_size,
                },
                ckpt_path,
            )
        print(f"{name} epoch {epoch:03d}: train_loss={train_loss:.3f} val_rmse={val_rmse:.3f}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, y_test, p_test = evaluate(model, test_loader, criterion, device)
    metrics = {
        "model": name,
        "best_val_rmse": best_val,
        "test_mse": test_loss,
        "test_rmse": rmse(y_test, p_test),
        "test_phm_score": phm_score(y_test, p_test),
    }
    plot_predictions(y_test, p_test, name, args.output_dir / "figures" / f"{name}_test_predictions.png")
    np.savez(args.output_dir / f"{name}_test_predictions.npz", y_true=y_test, y_pred=p_test)
    return history, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM baseline and variants on NASA C-MAPSS FD001.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "tcn", "dlinear"],
        choices=["lstm", "tcn", "dlinear"],
        help="Models to train. Default trains the baseline plus two variants.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    datasets = load_fd001(args.data_dir, window_size=args.window_size, seed=args.seed)

    histories = {}
    metrics = []
    for name in args.models:
        history, model_metrics = train_one_model(name, datasets, args, device)
        histories[name] = history
        metrics.append(model_metrics)

    plot_loss(histories, args.output_dir / "figures" / "loss_curves.png")

    metrics_csv = args.output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nFinal test metrics")
    for row in metrics:
        print(
            f"{row['model']}: RMSE={row['test_rmse']:.3f}, "
            f"PHM score={row['test_phm_score']:.1f}"
        )
    print(f"\nSaved metrics to {metrics_csv}")


if __name__ == "__main__":
    main()
