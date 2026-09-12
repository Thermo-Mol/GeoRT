from __future__ import annotations

from pathlib import Path

import torch


MOF_PREFIXES = (
    "mofchem_",
    "localchem_",
    "ffchem_",
    "cifcharge_",
    "poreesp_",
    "geometry_",
)
MOF_PHYSICAL_COLUMNS = {
    "LCD",
    "PLD",
    "Density",
    "ASA_m2_cm3",
    "ASA_m2_g",
    "VF",
    "PV_cm3_g",
}


def is_mof_feature(column: str) -> bool:
    return column in MOF_PHYSICAL_COLUMNS or column.startswith(MOF_PREFIXES)


def split_mof_feature_indices(feature_cols: list[str]) -> tuple[list[int], list[int], list[str], list[str]]:
    mof_indices = [idx for idx, column in enumerate(feature_cols) if is_mof_feature(column)]
    mof_index_set = set(mof_indices)
    other_indices = [idx for idx, _ in enumerate(feature_cols) if idx not in mof_index_set]
    mof_cols = [feature_cols[idx] for idx in mof_indices]
    other_cols = [feature_cols[idx] for idx in other_indices]
    return mof_indices, other_indices, mof_cols, other_cols


def load_partial_mof_encoder(
    encoder: torch.nn.Module,
    checkpoint_path: str,
    current_feature_cols: list[str],
) -> dict[str, object]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"MOF encoder pretrain checkpoint not found: {path}")

    checkpoint = torch.load(str(path), map_location="cpu")
    pretrained_cols = list(checkpoint.get("feature_cols", []))
    encoder_state = checkpoint.get("encoder_state", checkpoint.get("model", checkpoint))
    current_state = encoder.state_dict()
    copied = {}

    matched_cols = []
    if "0.weight" in encoder_state and "0.weight" in current_state and pretrained_cols:
        source_weight = encoder_state["0.weight"]
        target_weight = current_state["0.weight"].clone()
        source_lookup = {column: idx for idx, column in enumerate(pretrained_cols)}
        for target_idx, column in enumerate(current_feature_cols):
            source_idx = source_lookup.get(column)
            if source_idx is None or source_idx >= source_weight.shape[1]:
                continue
            target_weight[:, target_idx] = source_weight[:, source_idx]
            matched_cols.append(column)
        copied["0.weight"] = target_weight

    for key, value in encoder_state.items():
        if key == "0.weight":
            continue
        if key in current_state and current_state[key].shape == value.shape:
            copied[key] = value

    current_state.update(copied)
    encoder.load_state_dict(current_state, strict=True)
    return {
        "path": str(path),
        "pretrained_feature_count": len(pretrained_cols),
        "current_feature_count": len(current_feature_cols),
        "matched_feature_count": len(matched_cols),
        "matched_features": matched_cols,
    }
