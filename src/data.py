from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


MAX_RUL = 125
FEATURE_COLUMNS = [f"setting_{i}" for i in range(1, 4)] + [
    f"sensor_{i}" for i in range(1, 22)
]
COLUMNS = ["unit", "cycle"] + FEATURE_COLUMNS


@dataclass
class CmapssData:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    feature_dim: int
    scaler: StandardScaler


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


def read_cmapss_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype("float32")
    return df


def add_train_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    out = df.copy()
    max_cycles = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycles - out["cycle"]).clip(upper=max_rul)
    return out


def split_units(units: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(units)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def make_sliding_windows(
    df: pd.DataFrame,
    units: np.ndarray,
    window_size: int,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    for unit in units:
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        values = unit_df[feature_columns].to_numpy(dtype=np.float32)
        ruls = unit_df["rul"].to_numpy(dtype=np.float32)
        if len(unit_df) < window_size:
            continue
        for end in range(window_size, len(unit_df) + 1):
            sequences.append(values[end - window_size : end])
            targets.append(ruls[end - 1])
    return np.stack(sequences), np.array(targets, dtype=np.float32)


def make_last_windows_for_test(
    test_df: pd.DataFrame,
    rul_path: Path,
    window_size: int,
    max_rul: int = MAX_RUL,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    for i, unit in enumerate(sorted(test_df["unit"].unique())):
        unit_df = test_df[test_df["unit"] == unit].sort_values("cycle")
        values = unit_df[feature_columns].to_numpy(dtype=np.float32)
        if len(values) >= window_size:
            window = values[-window_size:]
        else:
            pad = np.repeat(values[:1], window_size - len(values), axis=0)
            window = np.concatenate([pad, values], axis=0)
        sequences.append(window)
        targets.append(min(float(true_rul.iloc[i]["rul"]), max_rul))
    return np.stack(sequences), np.array(targets, dtype=np.float32)


def load_cmapss_subset(
    data_dir: str | Path,
    subset: str = "FD001",
    window_size: int = 30,
    val_fraction: float = 0.2,
    seed: int = 7,
    max_rul: int = MAX_RUL,
) -> CmapssData:
    data_dir = Path(data_dir)
    subset = subset.upper()
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"
    missing = [str(p) for p in [train_path, test_path, rul_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing C-MAPSS files: "
            + ", ".join(missing)
            + ". Run scripts/download_data.py or unzip CMAPSSData.zip into data/raw."
        )

    train_df = add_train_rul(read_cmapss_file(train_path), max_rul=max_rul)
    test_df = read_cmapss_file(test_path)
    train_units, val_units = split_units(train_df["unit"].unique(), val_fraction, seed)

    scaler = StandardScaler()
    scaler.fit(train_df[train_df["unit"].isin(train_units)][FEATURE_COLUMNS])
    train_df.loc[:, FEATURE_COLUMNS] = scaler.transform(train_df[FEATURE_COLUMNS])
    test_df.loc[:, FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    x_train, y_train = make_sliding_windows(train_df, train_units, window_size)
    x_val, y_val = make_sliding_windows(train_df, val_units, window_size)
    x_test, y_test = make_last_windows_for_test(test_df, rul_path, window_size, max_rul=max_rul)

    return CmapssData(
        train_dataset=SequenceDataset(x_train, y_train),
        val_dataset=SequenceDataset(x_val, y_val),
        test_dataset=SequenceDataset(x_test, y_test),
        feature_dim=len(FEATURE_COLUMNS),
        scaler=scaler,
    )


def load_fd001(
    data_dir: str | Path,
    window_size: int = 30,
    val_fraction: float = 0.2,
    seed: int = 7,
    max_rul: int = MAX_RUL,
) -> CmapssData:
    return load_cmapss_subset(
        data_dir=data_dir,
        subset="FD001",
        window_size=window_size,
        val_fraction=val_fraction,
        seed=seed,
        max_rul=max_rul,
    )
