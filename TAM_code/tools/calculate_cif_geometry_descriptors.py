import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

try:
    from .geometry_features import build_geometry_features, normalize_atom
except ImportError:
    from geometry_features import build_geometry_features, normalize_atom


GEOMETRY_COLUMNS = [f"geometry_{idx:02d}" for idx in range(18)]


def calculate_one(cif_path: Path) -> dict:
    structure = Structure.from_file(str(cif_path))
    frame = structure.as_dataframe()
    atoms = frame["Species"].astype(str).map(normalize_atom).to_numpy(dtype=object)
    coords = frame[["x", "y", "z"]].to_numpy()
    values = build_geometry_features(
        atoms=atoms,
        coords=coords,
        lattice_abc=pd.Series(structure.lattice.abc).to_numpy(),
        volume=structure.lattice.volume,
    )
    row = {"coreid": cif_path.stem}
    row.update({column: float(value) for column, value in zip(GEOMETRY_COLUMNS, values)})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate lightweight CIF geometry descriptors.")
    parser.add_argument("--cif-dir", required=True, help="Directory containing <coreid>.cif files.")
    parser.add_argument("--output", default="geometry_descriptors.csv", help="Output CSV path.")
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    rows = [calculate_one(path) for path in sorted(cif_dir.glob("*.cif"))]
    if not rows:
        raise SystemExit(f"No CIF files found in {cif_dir}")
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

