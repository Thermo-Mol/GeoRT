from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class Standardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std < 1e-12:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray | float) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "Standardizer":
        return cls(mean=float(data["mean"]), std=float(data["std"]))


class TargetTransform:
    def __init__(self, mode: str, standardizer: Standardizer):
        if mode not in {"standard", "log_standard"}:
            raise ValueError(f"Unsupported target transform: {mode}")
        self.mode = mode
        self.standardizer = standardizer

    @classmethod
    def fit(cls, values: np.ndarray, mode: str) -> "TargetTransform":
        values = np.asarray(values, dtype=np.float64)
        fit_values = np.log(np.clip(values, 1e-12, None)) if mode == "log_standard" else values
        return cls(mode=mode, standardizer=Standardizer.fit(fit_values))

    def transform(self, values: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if self.mode == "log_standard":
            arr = np.log(np.clip(arr, 1e-12, None))
        return self.standardizer.transform(arr)

    def inverse(self, values: np.ndarray | float) -> np.ndarray:
        arr = self.standardizer.inverse(values)
        if self.mode == "log_standard":
            arr = np.exp(arr)
        return arr

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "standardizer": self.standardizer.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetTransform":
        return cls(mode=data["mode"], standardizer=Standardizer.from_dict(data["standardizer"]))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-12, None)) * 100.0)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "R2": float(r2)}


class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.step_idx = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_idx += 1
        if self.step_idx <= self.warmup_steps:
            scale = self.step_idx / self.warmup_steps
        else:
            progress = (self.step_idx - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * scale
