from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError as exc:  # pragma: no cover
    raise ImportError("RDKit is required. Install it with conda-forge or pip package 'rdkit'.") from exc


BOND_TYPE_TO_ID = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}

HYBRIDIZATION_TO_ID = {
    Chem.HybridizationType.SP: 1,
    Chem.HybridizationType.SP2: 2,
    Chem.HybridizationType.SP3: 3,
    Chem.HybridizationType.SP3D: 4,
    Chem.HybridizationType.SP3D2: 5,
}

EXPLICIT_DESCRIPTOR_NAMES = [
    "n_C",
    "n_H",
    "n_F",
    "n_Cl",
    "n_Br",
    "n_heavy_atoms",
    "n_halogens",
    "mol_wt",
    "f_per_c",
    "h_per_c",
    "halogen_per_c",
    "n_single_bonds",
    "n_double_bonds",
    "n_triple_bonds",
    "n_C_C_single",
    "n_C_C_double",
    "n_C_H",
    "n_C_F",
    "n_C_Cl",
    "n_C_Br",
    "n_CH3",
    "n_CH2",
    "n_CHF2",
    "n_CH2F",
    "n_CF3",
    "n_CHF",
    "n_CF2",
    "n_saturated_carbons",
    "n_unsaturated_carbons",
    "n_CH2_eq_CH",
    "n_CH_eq_CH",
    "n_CH2_eq_C",
    "n_CH_eq_C",
    "n_C_eq_C",
]


@dataclass(frozen=True)
class MoleculeGraph:
    atom_z: np.ndarray
    degree: np.ndarray
    formal_charge: np.ndarray
    hybridization: np.ndarray
    aromatic: np.ndarray
    in_ring: np.ndarray
    mass: np.ndarray
    coords: np.ndarray
    distance: np.ndarray
    bond_type: np.ndarray
    explicit_descriptor: np.ndarray
    atom_mask: np.ndarray


def _read_molecule(path: Path, sanitize: bool = True):
    suffix = path.suffix.lower()
    if suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=sanitize, removeHs=False)
    elif suffix in {".mol", ".sdf"}:
        mol = Chem.MolFromMolFile(str(path), sanitize=sanitize, removeHs=False)
    else:
        raise ValueError(f"Unsupported molecule file extension: {path.suffix}")
    if mol is None and sanitize:
        return _read_molecule(path, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse molecule file: {path}")
    return mol


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _bond_order(bond) -> str:
    btype = bond.GetBondType()
    if btype == Chem.BondType.SINGLE:
        return "single"
    if btype == Chem.BondType.DOUBLE:
        return "double"
    if btype == Chem.BondType.TRIPLE:
        return "triple"
    if btype == Chem.BondType.AROMATIC:
        return "aromatic"
    return "other"


def _atom_counts_for_carbon(atom) -> dict[str, int]:
    counts = {"H": int(atom.GetTotalNumHs()), "F": 0, "Cl": 0, "Br": 0, "C_single": 0, "C_double": 0}
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        z = other.GetAtomicNum()
        if z == 9:
            counts["F"] += 1
        elif z == 17:
            counts["Cl"] += 1
        elif z == 35:
            counts["Br"] += 1
        elif z == 6 and bond.GetBondType() == Chem.BondType.SINGLE:
            counts["C_single"] += 1
        elif z == 6 and bond.GetBondType() == Chem.BondType.DOUBLE:
            counts["C_double"] += 1
    return counts


def extract_explicit_descriptors(mol) -> np.ndarray:
    values = {name: 0.0 for name in EXPLICIT_DESCRIPTOR_NAMES}
    element_counts = {6: 0, 1: 0, 9: 0, 17: 0, 35: 0}
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in element_counts:
            element_counts[z] += 1
        if z == 6:
            element_counts[1] += int(atom.GetTotalNumHs())

    values["n_C"] = element_counts[6]
    values["n_H"] = element_counts[1]
    values["n_F"] = element_counts[9]
    values["n_Cl"] = element_counts[17]
    values["n_Br"] = element_counts[35]
    values["n_heavy_atoms"] = mol.GetNumHeavyAtoms()
    values["n_halogens"] = element_counts[9] + element_counts[17] + element_counts[35]
    values["mol_wt"] = float(Descriptors.MolWt(mol))
    values["f_per_c"] = _safe_ratio(values["n_F"], values["n_C"])
    values["h_per_c"] = _safe_ratio(values["n_H"], values["n_C"])
    values["halogen_per_c"] = _safe_ratio(values["n_halogens"], values["n_C"])

    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        za, zb = sorted([a.GetAtomicNum(), b.GetAtomicNum()])
        order = _bond_order(bond)
        if order == "single":
            values["n_single_bonds"] += 1
        elif order == "double":
            values["n_double_bonds"] += 1
        elif order == "triple":
            values["n_triple_bonds"] += 1

        if (za, zb) == (6, 6):
            if order == "single":
                values["n_C_C_single"] += 1
            elif order == "double":
                values["n_C_C_double"] += 1
                ca = _atom_counts_for_carbon(a)
                cb = _atom_counts_for_carbon(b)
                hs = sorted([ca["H"], cb["H"]], reverse=True)
                if hs == [2, 1]:
                    values["n_CH2_eq_CH"] += 1
                elif hs == [1, 1]:
                    values["n_CH_eq_CH"] += 1
                elif hs == [2, 0]:
                    values["n_CH2_eq_C"] += 1
                elif hs == [1, 0]:
                    values["n_CH_eq_C"] += 1
                elif hs == [0, 0]:
                    values["n_C_eq_C"] += 1
        elif (za, zb) == (1, 6):
            values["n_C_H"] += 1
        elif (za, zb) == (6, 9):
            values["n_C_F"] += 1
        elif (za, zb) == (6, 17):
            values["n_C_Cl"] += 1
        elif (za, zb) == (6, 35):
            values["n_C_Br"] += 1

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        c = _atom_counts_for_carbon(atom)
        h, f = c["H"], c["F"]
        if c["C_double"] > 0:
            values["n_unsaturated_carbons"] += 1
        else:
            values["n_saturated_carbons"] += 1
        if h == 3:
            values["n_CH3"] += 1
        if h == 2 and f == 0:
            values["n_CH2"] += 1
        if h == 1 and f == 2:
            values["n_CHF2"] += 1
        if h == 2 and f == 1:
            values["n_CH2F"] += 1
        if h == 0 and f == 3:
            values["n_CF3"] += 1
        if h == 1 and f == 1:
            values["n_CHF"] += 1
        if h == 0 and f == 2:
            values["n_CF2"] += 1

    return np.array([values[name] for name in EXPLICIT_DESCRIPTOR_NAMES], dtype=np.float32)


def parse_molecule_file(
    path: str | Path,
    max_atoms: int,
    add_hydrogens_if_missing: bool = False,
) -> MoleculeGraph:
    path = Path(path)
    mol = _read_molecule(path)
    if add_hydrogens_if_missing:
        mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers() == 0:
        raise ValueError(f"Molecule has no 3D coordinates: {path}")
    n_atoms = mol.GetNumAtoms()
    if n_atoms > max_atoms:
        raise ValueError(f"{path.name} has {n_atoms} atoms, larger than max_atoms={max_atoms}")

    conf = mol.GetConformer()
    atom_z = np.zeros(max_atoms, dtype=np.int64)
    degree = np.zeros(max_atoms, dtype=np.int64)
    formal_charge = np.zeros(max_atoms, dtype=np.int64)
    hybridization = np.zeros(max_atoms, dtype=np.int64)
    aromatic = np.zeros(max_atoms, dtype=np.int64)
    in_ring = np.zeros(max_atoms, dtype=np.int64)
    mass = np.zeros(max_atoms, dtype=np.float32)
    coords = np.zeros((max_atoms, 3), dtype=np.float32)
    atom_mask = np.zeros(max_atoms, dtype=bool)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_z[idx] = atom.GetAtomicNum()
        degree[idx] = min(atom.GetTotalDegree(), 8)
        formal_charge[idx] = int(np.clip(atom.GetFormalCharge(), -5, 5)) + 5
        hybridization[idx] = HYBRIDIZATION_TO_ID.get(atom.GetHybridization(), 0)
        aromatic[idx] = int(atom.GetIsAromatic())
        in_ring[idx] = int(atom.IsInRing())
        mass[idx] = atom.GetMass() / 200.0
        pos = conf.GetAtomPosition(idx)
        coords[idx] = [pos.x, pos.y, pos.z]
        atom_mask[idx] = True

    real_coords = coords[:n_atoms]
    diff = real_coords[:, None, :] - real_coords[None, :, :]
    real_distance = np.linalg.norm(diff, axis=-1).astype(np.float32)
    distance = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    distance[:n_atoms, :n_atoms] = real_distance

    bond_type = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b = BOND_TYPE_TO_ID.get(bond.GetBondType(), 0)
        bond_type[i, j] = b
        bond_type[j, i] = b

    return MoleculeGraph(
        atom_z=atom_z,
            degree=degree,
        formal_charge=formal_charge,
        hybridization=hybridization,
        aromatic=aromatic,
        in_ring=in_ring,
        mass=mass,
        coords=coords,
        distance=distance,
        bond_type=bond_type,
        explicit_descriptor=extract_explicit_descriptors(mol),
        atom_mask=atom_mask,
    )


def find_molecule_file(mol_dir: str | Path, refrigerant: str, file_map: dict[str, str] | None = None) -> Path:
    mol_dir = Path(mol_dir)
    file_map = file_map or {}
    if refrigerant in file_map:
        mapped = mol_dir / file_map[refrigerant]
        if mapped.exists():
            return mapped
        raise FileNotFoundError(f"Configured molecule file does not exist for {refrigerant}: {mapped}")

    candidates = []
    normalized = refrigerant.lower().replace("_", "").replace("-", "")
    for path in mol_dir.iterdir():
        if path.suffix.lower() not in {".mol", ".mol2", ".sdf"}:
            continue
        stem = path.stem.lower().replace("_", "").replace("-", "")
        if stem == normalized:
            return path
        candidates.append(path.name)
    raise FileNotFoundError(
        f"No molecule file found for {refrigerant} in {mol_dir}. "
        f"Add it to molecule_file_map. Available: {candidates}"
    )
