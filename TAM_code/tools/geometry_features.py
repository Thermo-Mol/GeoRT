import re
from pathlib import Path

import numpy as np
from pymatgen.core import Structure


def normalize_atom(atom: object) -> str:
    return re.sub(r"\d+", "", str(atom))


def build_geometry_features(
    atoms: np.ndarray,
    coords: np.ndarray,
    lattice_abc: np.ndarray,
    volume: np.float32,
) -> np.ndarray:
    """Build the same lightweight geometry summary used by the isobutane fusion baseline."""

    coords = coords.astype(np.float32)
    centered = coords - coords.mean(axis=0, keepdims=True)
    bbox = centered.max(axis=0) - centered.min(axis=0)
    element_counts = {}
    for atom in atoms.tolist():
        element_counts[atom] = element_counts.get(atom, 0) + 1
    counts = np.array(sorted(element_counts.values(), reverse=True), dtype=np.float32)
    top_counts = np.zeros(3, dtype=np.float32)
    top_counts[: min(3, len(counts))] = counts[:3]
    top_fracs = top_counts / max(len(atoms), 1)

    if len(coords) >= 2:
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)
        upper = dist[np.triu_indices(len(coords), k=1)]
        nearest = np.partition(dist + np.eye(len(coords), dtype=np.float32) * 1e6, 1, axis=1)[:, 1]
        distance_stats = np.array(
            [
                upper.mean(),
                upper.std(),
                np.quantile(upper, 0.25),
                np.quantile(upper, 0.5),
                np.quantile(upper, 0.75),
                nearest.mean(),
            ],
            dtype=np.float32,
        )
    else:
        distance_stats = np.zeros(6, dtype=np.float32)

    return np.concatenate(
        [
            np.array(
                [
                    float(len(atoms)),
                    float(len(element_counts)),
                    float(volume),
                ],
                dtype=np.float32,
            ),
            lattice_abc.astype(np.float32),
            bbox.astype(np.float32),
            distance_stats,
            top_fracs.astype(np.float32),
        ]
    )


class GeometryCache:
    def __init__(self, cif_dir: Path):
        self.cif_dir = cif_dir
        self.cache: dict[str, np.ndarray] = {}

    def get(self, coreid: str) -> np.ndarray:
        coreid = str(coreid).strip()
        if coreid not in self.cache:
            cif_path = self.cif_dir / f"{coreid}.cif"
            if not cif_path.exists():
                raise FileNotFoundError(f"CIF not found for coreid={coreid}: {cif_path}")
            structure = Structure.from_file(str(cif_path))
            frame = structure.as_dataframe()
            atoms = frame["Species"].astype(str).map(normalize_atom).to_numpy(dtype=object)
            coords = frame[["x", "y", "z"]].to_numpy(dtype=np.float32)
            self.cache[coreid] = build_geometry_features(
                atoms=atoms,
                coords=coords,
                lattice_abc=np.asarray(structure.lattice.abc, dtype=np.float32),
                volume=np.float32(structure.lattice.volume),
            )
        return self.cache[coreid]
