from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refrig_geotransformer.data import MoleculeCache, read_thermal_table
from refrig_geotransformer.splits import make_group_split
from refrig_geotransformer.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "r1216_geometry_transformer.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = read_thermal_table(cfg)
    print(f"Rows: {len(df)}")
    print(f"Refrigerants: {df['refrigerant'].nunique()} -> {sorted(df['refrigerant'].unique().tolist())}")
    cache = MoleculeCache(cfg)
    for name in sorted(df["refrigerant"].unique()):
        mol = cache.get(name)
        print(f"{name:14s} atoms={int(mol.atom_mask.sum()):2d} file={cache.paths[name]}")
    split = make_group_split(
        df,
        "refrigerant",
        cfg["split"]["train_ratio"],
        cfg["split"]["val_ratio"],
        cfg["split"]["test_ratio"],
        cfg["seed"],
        cfg["split"].get("forced_val_groups", []),
        cfg["split"].get("forced_test_groups", []),
    )
    print("Leak-free group split:")
    for key, groups in split.items():
        print(f"  {key}: {groups}")


if __name__ == "__main__":
    main()
