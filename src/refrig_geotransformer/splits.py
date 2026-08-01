from __future__ import annotations

import random
from typing import Iterable

import pandas as pd


def _ratio_to_group_counts(n_groups: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    raw = {
        "train": n_groups * train_ratio,
        "val": n_groups * val_ratio,
        "test": n_groups * test_ratio,
    }
    counts = {name: int(value) for name, value in raw.items()}
    remaining = n_groups - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], raw[name]), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1

    for name in ("train", "val", "test"):
        if counts[name] == 0 and n_groups >= 3:
            donor = max(counts, key=counts.get)
            counts[donor] -= 1
            counts[name] += 1
    return counts


def make_group_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    forced_val_groups: Iterable[str] | None = None,
    forced_test_groups: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.")

    groups = sorted(str(g) for g in df[group_col].dropna().unique())
    group_set = set(groups)
    forced_val = set(forced_val_groups or [])
    forced_test = set(forced_test_groups or [])
    unknown = (forced_val | forced_test) - group_set
    if unknown:
        raise ValueError(f"Forced split groups not present in data: {sorted(unknown)}")
    overlap = forced_val & forced_test
    if overlap:
        raise ValueError(f"Groups cannot be forced into both val and test: {sorted(overlap)}")

    target_counts = _ratio_to_group_counts(len(groups), train_ratio, val_ratio, test_ratio)
    if len(forced_val) > target_counts["val"]:
        raise ValueError(f"forced_val_groups has {len(forced_val)} groups, larger than val quota {target_counts['val']}.")
    if len(forced_test) > target_counts["test"]:
        raise ValueError(f"forced_test_groups has {len(forced_test)} groups, larger than test quota {target_counts['test']}.")

    assigned: dict[str, list[str]] = {
        "train": [],
        "val": sorted(forced_val),
        "test": sorted(forced_test),
    }

    remaining = [g for g in groups if g not in forced_val and g not in forced_test]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    for name in ("val", "test", "train"):
        need = target_counts[name] - len(assigned[name])
        if need > 0:
            assigned[name].extend(remaining[:need])
            remaining = remaining[need:]
    assigned["train"].extend(remaining)

    result = {name: sorted(assigned[name]) for name in ("train", "val", "test")}
    seen = result["train"] + result["val"] + result["test"]
    if len(seen) != len(set(seen)):
        raise AssertionError("Group leakage detected inside split assignment.")
    if set(seen) != group_set:
        raise AssertionError("Some groups were lost during splitting.")
    return result


def apply_group_split(df: pd.DataFrame, group_col: str, split: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    out = {}
    all_used: list[str] = []
    for name, groups in split.items():
        all_used.extend(groups)
        out[name] = df[df[group_col].astype(str).isin(groups)].copy()
    if len(all_used) != len(set(all_used)):
        raise AssertionError("A refrigerant appears in more than one split.")
    return out
