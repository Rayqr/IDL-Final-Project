from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: tuple[int, ...] = (32, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_channels = input_dim
        for i, out_channels in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(),
            nn.Linear(channels[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.tcn(x)
        return self.head(out[:, :, -1])
