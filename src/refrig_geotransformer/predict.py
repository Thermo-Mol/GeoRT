from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .chem import parse_molecule_file
from .data import collate_batch, get_condition_columns
from .model import GeometryAwareStateTransformer
from .utils import Standardizer, TargetTransform


def make_item(
    mol,
    row,
    condition_standardizers: dict[str, Standardizer],
    descriptor_standardizers: dict[str, Standardizer] | None = None,
):
    cond = torch.tensor(
        [condition_standardizers[col].transform(float(row[col])) for col in condition_standardizers],
        dtype=torch.float32,
    )
    if descriptor_standardizers:
        descriptor = torch.tensor(
            [
                float(descriptor_standardizers[f"desc_{i}"].transform(float(value)))
                for i, value in enumerate(mol.explicit_descriptor)
            ],
            dtype=torch.float32,
        )
    else:
        descriptor = torch.from_numpy(mol.explicit_descriptor.astype("float32"))
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
        "condition": cond,
        "explicit_descriptor": descriptor,
        "group_feature": torch.zeros(0, dtype=torch.float32),
        "target": torch.tensor(0.0),
        "target_raw": torch.tensor(0.0),
        "critical_target": torch.zeros(2, dtype=torch.float32),
        "critical_mask": torch.tensor(False, dtype=torch.bool),
        "weight": torch.tensor(1.0),
        "refrigerant": "unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--molecule", required=True, help="Path to .mol/.mol2/.sdf file, e.g. R1216.mol")
    parser.add_argument("--conditions", required=True, help="CSV with columns T and P, or temperature and pressure.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    transforms = ckpt["transforms"]
    condition_standardizers = {
        key: Standardizer.from_dict(value)
        for key, value in transforms["condition"].items()
    }
    descriptor_standardizers = None
    if "explicit_descriptor" in transforms:
        descriptor_standardizers = {
            key: Standardizer.from_dict(value)
            for key, value in transforms["explicit_descriptor"]["standardizers"].items()
        }
    target_transform = TargetTransform.from_dict(transforms["target"])

    mol = parse_molecule_file(
        Path(args.molecule),
        max_atoms=int(config["molecule"]["max_atoms"]),
        add_hydrogens_if_missing=bool(config["molecule"].get("add_hydrogens_if_missing", False)),
    )
    cond_df = pd.read_csv(args.conditions)
    aliases = {
        config["data"].get("pressure_col", "p Mpa"): ["P", "pressure", "p Mpa"],
        "critical_temperature": ["Tc", "critical_temperature"],
        "critical_pressure": ["Pc", "critical_pressure"],
    }
    for col in condition_standardizers:
        if col in cond_df.columns:
            continue
        found = next((alias for alias in aliases.get(col, []) if alias in cond_df.columns), None)
        if found:
            cond_df[col] = cond_df[found]
        else:
            raise KeyError(f"Conditions CSV must contain required condition column: {col}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryAwareStateTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds = []
    with torch.no_grad():
        for _, row in cond_df.iterrows():
            item = make_item(mol, row, condition_standardizers, descriptor_standardizers)
            batch = collate_batch([item])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred_t = model(batch).cpu().numpy()
            preds.append(float(target_transform.inverse(pred_t)[0]))
    out = cond_df.copy()
    out["predicted_liquid_thermal_conductivity"] = preds
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
