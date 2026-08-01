from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .chem import MoleculeGraph, find_molecule_file, parse_molecule_file
from .utils import Standardizer, TargetTransform


def get_condition_columns(config: dict[str, Any]) -> list[str]:
    data_cfg = config["data"]
    return list(data_cfg.get("condition_cols", [data_cfg["temperature_col"], data_cfg["pressure_col"]]))


def get_group_columns(config: dict[str, Any]) -> list[str]:
    return list(config["data"].get("group_cols", []))


def get_refrigerant_weight(config: dict[str, Any], refrigerant: str) -> float:
    weighting = config.get("sample_weighting", {})
    weights = weighting.get("refrigerant_weights", {})
    return float(weights.get(refrigerant, weighting.get("default_weight", 1.0)))


def resolve_condition_columns(config: dict[str, Any], df: pd.DataFrame) -> list[str]:
    data_cfg = config["data"]
    resolved = []
    for col in get_condition_columns(config):
        if col in df.columns:
            resolved.append(col)
        elif col == data_cfg.get("critical_temperature_col") and "critical_temperature" in df.columns:
            resolved.append("critical_temperature")
        elif col == data_cfg.get("critical_pressure_col") and "critical_pressure" in df.columns:
            resolved.append("critical_pressure")
        else:
            raise KeyError(f"Condition column '{col}' was not found after table parsing. Existing columns: {list(df.columns)}")
    return resolved


def read_thermal_table(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    df = pd.read_excel(data_cfg["excel_path"], sheet_name=data_cfg.get("sheet_name", 0))
    property_path = data_cfg.get("physical_property_path")
    if property_path:
        prop_df = pd.read_csv(property_path)
        prop_ref_col = data_cfg.get("physical_property_refrigerant_col", "refrigerant")
        if prop_ref_col not in prop_df.columns:
            raise KeyError(f"Physical property table is missing refrigerant column '{prop_ref_col}'. Existing columns: {list(prop_df.columns)}")
        prop_df = prop_df.rename(columns={prop_ref_col: data_cfg["refrigerant_col"]})
        prop_df[data_cfg["refrigerant_col"]] = prop_df[data_cfg["refrigerant_col"]].astype(str).str.strip()
        df[data_cfg["refrigerant_col"]] = df[data_cfg["refrigerant_col"]].astype(str).str.strip()
        overlap = [c for c in prop_df.columns if c != data_cfg["refrigerant_col"] and c in df.columns]
        if overlap:
            df = df.drop(columns=overlap)
        df = df.merge(prop_df, on=data_cfg["refrigerant_col"], how="left")
    aux_enabled = bool(config.get("auxiliary", {}).get("predict_critical_properties", False))
    condition_cols = get_condition_columns(config)
    group_cols = get_group_columns(config) if bool(config.get("model", {}).get("use_group_features", False)) else []
    keep_cols = [
        data_cfg["refrigerant_col"],
        data_cfg["target_col"],
    ]
    keep_cols.extend(condition_cols)
    keep_cols.extend(group_cols)
    if aux_enabled:
        keep_cols.extend([data_cfg["critical_temperature_col"], data_cfg["critical_pressure_col"]])
    keep_cols = list(dict.fromkeys(keep_cols))
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in Excel file: {missing}. Existing columns: {list(df.columns)}")

    df = df[keep_cols].copy()
    df = df.rename(
        columns={
            data_cfg["refrigerant_col"]: "refrigerant",
            data_cfg["target_col"]: "target",
            data_cfg.get("critical_temperature_col"): "critical_temperature",
            data_cfg.get("critical_pressure_col"): "critical_pressure",
        }
    )
    df["refrigerant"] = df["refrigerant"].astype(str).str.strip()
    renamed_condition_cols = resolve_condition_columns(config, df)
    for col in ["target", *renamed_condition_cols, *group_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.attrs["condition_cols"] = renamed_condition_cols
    df.attrs["group_cols"] = group_cols
    required = ["refrigerant", "target", *renamed_condition_cols, *group_cols]
    if aux_enabled:
        for col in ["critical_temperature", "critical_pressure"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        required.extend(["critical_temperature", "critical_pressure"])
    df = df.dropna(subset=required).reset_index(drop=True)
    if (df["target"] <= 0).any():
        raise ValueError("Target values must be positive for log-standard target transform.")
    return df


class MoleculeCache:
    def __init__(self, config: dict[str, Any]):
        self.mol_dir = Path(config["data"]["mol_dir"])
        self.file_map = config["data"].get("molecule_file_map", {})
        mol_cfg = config["molecule"]
        self.max_atoms = int(mol_cfg["max_atoms"])
        self.add_h = bool(mol_cfg.get("add_hydrogens_if_missing", False))
        self.cache: dict[str, MoleculeGraph] = {}
        self.paths: dict[str, str] = {}

    def get(self, refrigerant: str) -> MoleculeGraph:
        if refrigerant not in self.cache:
            path = find_molecule_file(self.mol_dir, refrigerant, self.file_map)
            self.paths[refrigerant] = str(path)
            self.cache[refrigerant] = parse_molecule_file(path, max_atoms=self.max_atoms, add_hydrogens_if_missing=self.add_h)
        return self.cache[refrigerant]


class ThermalConductivityDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
        condition_standardizers: dict[str, Standardizer],
        target_transform: TargetTransform,
        molecule_cache: MoleculeCache,
        auxiliary_standardizers: dict[str, Standardizer] | None = None,
        descriptor_standardizers: dict[str, Standardizer] | None = None,
        group_standardizers: dict[str, Standardizer] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.condition_standardizers = condition_standardizers
        self.condition_cols = list(condition_standardizers.keys())
        self.target_transform = target_transform
        self.molecule_cache = molecule_cache
        self.auxiliary_standardizers = auxiliary_standardizers or {}
        self.descriptor_standardizers = descriptor_standardizers or {}
        self.group_standardizers = group_standardizers or {}
        self.group_cols = list(self.group_standardizers.keys())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        refrigerant = str(row["refrigerant"])
        mol = self.molecule_cache.get(refrigerant)
        target = float(row["target"])
        cond = np.array(
            [self.condition_standardizers[col].transform(float(row[col])) for col in self.condition_cols],
            dtype=np.float32,
        )
        y = self.target_transform.transform(target).astype(np.float32)
        if self.auxiliary_standardizers:
            critical = np.array(
                [
                    self.auxiliary_standardizers["critical_temperature"].transform(float(row["critical_temperature"])),
                    self.auxiliary_standardizers["critical_pressure"].transform(float(row["critical_pressure"])),
                ],
                dtype=np.float32,
            )
            critical_mask = True
        else:
            critical = np.zeros(2, dtype=np.float32)
            critical_mask = False
        if self.descriptor_standardizers:
            descriptor = np.array(
                [
                    float(self.descriptor_standardizers[f"desc_{i}"].transform(float(value)))
                    for i, value in enumerate(mol.explicit_descriptor)
                ],
                dtype=np.float32,
            )
        else:
            descriptor = mol.explicit_descriptor.astype(np.float32)
        if self.group_standardizers:
            group_feature = np.array(
                [self.group_standardizers[col].transform(float(row[col])) for col in self.group_cols],
                dtype=np.float32,
            )
        else:
            group_feature = np.zeros(0, dtype=np.float32)

        return {
            "atom_z": torch.from_numpy(mol.atom_z),
            "degree": torch.from_numpy(mol.degree),
            "formal_charge": torch.from_numpy(mol.formal_charge),
            "hybridization": torch.from_numpy(mol.hybridization),
            "aromatic": torch.from_numpy(mol.aromatic),
            "in_ring": torch.from_numpy(mol.in_ring),
            "mass": torch.from_numpy(mol.mass),
            "distance": torch.from_numpy(mol.distance),
            "bond_type": torch.from_numpy(mol.bond_type),
            "atom_mask": torch.from_numpy(mol.atom_mask),
            "condition": torch.from_numpy(cond),
            "explicit_descriptor": torch.from_numpy(descriptor),
            "group_feature": torch.from_numpy(group_feature),
            "target": torch.tensor(y, dtype=torch.float32),
            "target_raw": torch.tensor(target, dtype=torch.float32),
            "critical_target": torch.from_numpy(critical),
            "critical_mask": torch.tensor(critical_mask, dtype=torch.bool),
            "weight": torch.tensor(get_refrigerant_weight(self.config, refrigerant), dtype=torch.float32),
            "refrigerant": refrigerant,
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "atom_z",
        "degree",
        "formal_charge",
        "hybridization",
        "aromatic",
        "in_ring",
        "mass",
        "distance",
        "bond_type",
        "atom_mask",
        "condition",
        "explicit_descriptor",
        "group_feature",
        "target",
        "target_raw",
        "critical_target",
        "critical_mask",
        "weight",
    ]
    batch = {key: torch.stack([item[key] for item in items], dim=0) for key in tensor_keys}
    batch["refrigerant"] = [item["refrigerant"] for item in items]
    return batch
