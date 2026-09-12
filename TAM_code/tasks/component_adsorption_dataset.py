import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    from ..tools.geometry_features import GeometryCache
except ImportError:
    from tools.geometry_features import GeometryCache


SUPPORTED_GASES = {"CO2", "H2", "N2", "O2", "CH4", "C2H4", "C3H6", "C3H8"}

DESCRIPTOR_ALIASES = {
    "LCD": ["LCD (A)"],
    "PLD": ["PLD (A)"],
    "Density": ["Density (g/cm3)"],
    "ASA_m2_cm3": ["ASA (m2/cm3)"],
    "ASA_m2_g": ["ASA (m2/g)"],
    "VF": ["VF"],
    "PV_cm3_g": ["PV (cm3/g)"],
}

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

COREID_ALIASES = ["coreid"]
MOF_CHEM_ID_COLUMNS = {
    "case_name",
    "test_refcode",
    "matched_refcode",
    "coreid",
    "refcode",
    "cif_path",
    "mof_id",
    "metal_symbols",
    "main_metal",
    "localchem_cif_path",
    "ffchem_cif_path",
    "ffchem_force_field_path",
    "cifcharge_cif_path",
    "poreesp_cif_path",
}


def normalize_coreid(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return np.nan if text == "" or text.lower() == "nan" else text


def normalize_gas_name(name: object) -> str:
    text = str(name).strip()
    if text not in SUPPORTED_GASES:
        raise KeyError(f"Gas name '{text}' is not supported. Use one of {sorted(SUPPORTED_GASES)}")
    return text


def canonicalize_column_name(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+", "", text)
    return text.replace("_", "").replace("-", "").lower()


def first_existing_column(df: pd.DataFrame, aliases: list[str]) -> str:
    canonical = {column: canonicalize_column_name(column) for column in df.columns}
    for alias in aliases:
        alias_key = canonicalize_column_name(alias)
        for column, column_key in canonical.items():
            if column_key == alias_key:
                return column
    raise KeyError(f"None of these columns were found: {aliases}")


def is_mof_feature(column: str) -> bool:
    return column in MOF_PHYSICAL_COLUMNS or column.startswith(MOF_PREFIXES)


def split_mof_feature_indices(feature_cols: list[str]) -> tuple[list[int], list[int], list[str], list[str]]:
    mof_indices = [idx for idx, column in enumerate(feature_cols) if is_mof_feature(column)]
    mof_index_set = set(mof_indices)
    other_indices = [idx for idx, _ in enumerate(feature_cols) if idx not in mof_index_set]
    mof_cols = [feature_cols[idx] for idx in mof_indices]
    other_cols = [feature_cols[idx] for idx in other_indices]
    return mof_indices, other_indices, mof_cols, other_cols


def load_structure_mapping(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    coreid_column = first_existing_column(header, COREID_ALIASES)
    refcode_column = first_existing_column(header, ["refcode"])
    df = pd.read_csv(path, dtype=str, usecols=[coreid_column, refcode_column]).copy()
    df = df.rename(columns={coreid_column: "coreid", refcode_column: "refcode"})
    df["case_key"] = df["refcode"].astype(str).str.extract(r"^([A-Za-z0-9]+)")[0]
    df["coreid"] = df["coreid"].map(normalize_coreid)
    df = df.dropna(subset=["case_key", "coreid"]).copy()
    duplicate_count = int(df["case_key"].duplicated().sum())
    if duplicate_count:
        print(f"Warning: {duplicate_count} duplicate refcode keys in {path}; keeping the first occurrence.")
    return df.drop_duplicates("case_key", keep="first")[["case_key", "coreid", "refcode"]]


def load_descriptors(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, sheet_name="Sheet1").copy()
    coreid_column = first_existing_column(df, COREID_ALIASES)

    renamed = {"coreid": df[coreid_column].map(normalize_coreid)}
    descriptor_cols = []
    for clean_name, aliases in DESCRIPTOR_ALIASES.items():
        original = first_existing_column(df, aliases)
        renamed[clean_name] = pd.to_numeric(df[original], errors="coerce")
        descriptor_cols.append(clean_name)
    descriptors = pd.DataFrame(renamed).dropna(subset=["coreid"]).copy()
    duplicate_count = int(descriptors["coreid"].duplicated().sum())
    if duplicate_count:
        print(f"Warning: {duplicate_count} duplicate coreid rows in {path}; keeping the first occurrence.")
    return descriptors.drop_duplicates("coreid", keep="first"), descriptor_cols


def load_mof_chemical_descriptors(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"MOF chemical descriptor file not found: {path}")
    df = pd.read_csv(path).copy()
    key_column = "case_name" if "case_name" in df.columns else "coreid"
    if key_column not in df.columns:
        raise KeyError(f"{path} must contain either a case_name or coreid column")

    renamed = {key_column: df[key_column].astype(str).str.strip()}
    descriptor_cols = []
    for column in df.columns:
        if column in MOF_CHEM_ID_COLUMNS:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() == 0 or numeric.nunique(dropna=True) < 2:
            continue
        clean_column = re.sub(r"[^A-Za-z0-9_]+", "_", str(column)).strip("_")
        has_known_prefix = clean_column.startswith(("mofchem_", "ffchem_", "localchem_"))
        feature_name = clean_column if has_known_prefix else f"mofchem_{clean_column}"
        renamed[feature_name] = numeric
        descriptor_cols.append(feature_name)

    descriptors = pd.DataFrame(renamed).dropna(subset=[key_column]).copy()
    duplicate_count = int(descriptors[key_column].duplicated().sum())
    if duplicate_count:
        print(f"Warning: {duplicate_count} duplicate {key_column} rows in {path}; keeping the first occurrence.")
    return descriptors.drop_duplicates(key_column, keep="first"), descriptor_cols


def load_mof_chemical_descriptor_tables(
    data_dir: Path,
    primary_csv: str,
    extra_csvs: object = "",
) -> tuple[pd.DataFrame, list[str]]:
    paths = [str(primary_csv)]
    if isinstance(extra_csvs, str):
        paths.extend([item.strip() for item in extra_csvs.split(",") if item.strip()])
    elif extra_csvs:
        paths.extend([str(item).strip() for item in extra_csvs if str(item).strip()])

    merged = None
    all_cols = []
    seen_paths = set()
    for item in paths:
        path = (data_dir / item).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        table, cols = load_mof_chemical_descriptors(path)
        duplicate_cols = [col for col in cols if col in all_cols]
        if duplicate_cols:
            table = table.drop(columns=duplicate_cols)
            cols = [col for col in cols if col not in duplicate_cols]
        merge_key = "case_name" if "case_name" in table.columns else "coreid"
        merged = table if merged is None else merged.merge(table, on=merge_key, how="left")
        all_cols.extend(cols)

    if merged is None:
        raise ValueError("No MOF chemical descriptor tables were loaded.")
    for column in all_cols:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged, all_cols


def load_gas_properties(path: Path, prefix: str = "gas0") -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, sheet_name="Sheet1").copy()
    base_cols = ["Tc", "pc", "w", "d", "MW"]
    required = ["gas", *base_cols]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing gas columns: {missing}")

    feature_cols = []
    for column in df.columns:
        if column == "gas":
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() > 0:
            df[column] = numeric
            feature_cols.append(column)

    df = df[["gas", *feature_cols]].copy()
    df["gas_norm"] = df["gas"].map(normalize_gas_name)
    renamed_cols = {column: f"{prefix}_{column}" for column in feature_cols}
    result = df.drop(columns=["gas"]).rename(columns=renamed_cols)
    return result, [renamed_cols[column] for column in feature_cols]


def load_source_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)
    required = [
        "dataset_name",
        "case_name",
        "output_file",
        "component0_name",
        "component0_uptake_molkg",
        "component1_name",
        "component1_uptake_molkg",
        "mol_ratio",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Source table is missing required columns: {missing}")
    optional = ["coreid"]
    return df[[*required, *[column for column in optional if column in df.columns]]].copy()


def parse_output_conditions(output_file: object) -> tuple[float, float]:
    text = str(output_file)
    match = re.search(r"output_([0-9]+(?:\.[0-9]+)?)_([^/\\]+)\.s0\.txt$", text)
    if not match:
        raise ValueError(f"Could not parse temperature/pressure from output_file: {text}")
    temperature = float(match.group(1))
    pressure = float(match.group(2))
    return temperature, pressure


def build_mixture_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        mol_ratio = float(row["mol_ratio"])
        frac0 = mol_ratio / (1.0 + mol_ratio)
        frac1 = 1.0 / (1.0 + mol_ratio)
        temperature, pressure = parse_output_conditions(row["output_file"])
        rows.append(
            {
                "dataset_name": row["dataset_name"],
                "case_name": row["case_name"],
                **({"coreid": row["coreid"]} if "coreid" in row.index else {}),
                "output_file": row["output_file"],
                "component0_name": row["component0_name"],
                "component1_name": row["component1_name"],
                "temperature_K": temperature,
                "pressure_Pa": pressure,
                "log_pressure_Pa": math.log1p(pressure),
                "mol_ratio": mol_ratio,
                "component0_mol_fraction": frac0,
                "component1_mol_fraction": frac1,
                "target_component0_uptake_molkg": row["component0_uptake_molkg"],
                "target_component1_uptake_molkg": row["component1_uptake_molkg"],
            }
        )
    result = pd.DataFrame(rows)
    result["case_key"] = result["case_name"].astype(str).str.extract(r"^\d+_([^_]+)")[0]
    result["component0_name_norm"] = result["component0_name"].map(normalize_gas_name)
    result["component1_name_norm"] = result["component1_name"].map(normalize_gas_name)
    return result


def build_dataset(args) -> tuple[pd.DataFrame, list[str], list[str]]:
    script_dir = Path(__file__).resolve().parents[1]
    source_path = (script_dir / args.source_xlsx).resolve()
    data_dir = (script_dir / args.data_dir).resolve()
    cif_dir = Path(args.cif_dir).resolve()

    source = build_mixture_rows(load_source_table(source_path))
    component1_filter = str(getattr(args, "component1_filter", "all")).strip()
    if component1_filter and component1_filter.lower() != "all":
        selected_gas = normalize_gas_name(component1_filter)
        source = source[source["component1_name_norm"] == selected_gas].copy()
        if source.empty:
            raise ValueError(f"No rows left after --component1-filter {component1_filter!r}")
    descriptors, descriptor_cols = load_descriptors(data_dir / args.descriptors_xlsx)
    mof_chem, mof_chem_cols = load_mof_chemical_descriptor_tables(
        data_dir,
        args.mof_chem_csv,
        getattr(args, "extra_mof_chem_csvs", ""),
    )
    gas0_props, gas0_feature_cols = load_gas_properties(data_dir / args.gas_xlsx, prefix="gas0")
    gas1_props, gas1_feature_cols = load_gas_properties(data_dir / args.gas_xlsx, prefix="gas1")

    if "coreid" not in source.columns:
        structure_csv = (script_dir / args.structure_csv).resolve()
        structure_mapping = load_structure_mapping(structure_csv)
        df = source.merge(structure_mapping, on="case_key", how="left")
    else:
        df = source.copy()
        df["coreid"] = df["coreid"].map(normalize_coreid)
    missing_coreid = int(df["coreid"].isna().sum())
    if missing_coreid:
        print(f"Warning: dropping {missing_coreid} rows without coreid.")
    df = df.merge(descriptors, on="coreid", how="left")
    mof_chem_key = "case_name" if "case_name" in mof_chem.columns else "coreid"
    df = df.merge(mof_chem, on=mof_chem_key, how="left")
    df = df.merge(gas0_props, left_on="component0_name_norm", right_on="gas_norm", how="left")
    df = df.drop(columns=["gas_norm"])
    df = df.merge(gas1_props, left_on="component1_name_norm", right_on="gas_norm", how="left")
    df = df.drop(columns=["gas_norm"])

    # Gas identities remain metadata for property lookup and splitting only.
    # No gas identity one-hot columns are generated or passed to the model.

    base_feature_cols = [
        *descriptor_cols,
        *mof_chem_cols,
        *gas0_feature_cols,
        *gas1_feature_cols,
        "temperature_K",
        "pressure_Pa",
        "log_pressure_Pa",
        "component0_mol_fraction",
        "component1_mol_fraction",
        "mol_ratio",
    ]

    target_cols = ["target_component0_uptake_molkg", "target_component1_uptake_molkg"]
    for column in [*base_feature_cols, *target_cols]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["coreid", *base_feature_cols, *target_cols]).reset_index(drop=True)

    geometry_cache = GeometryCache(cif_dir)
    geometry_rows = []
    keep_indices = []
    failures = []
    for idx, row in df.iterrows():
        try:
            geometry_rows.append(geometry_cache.get(row["coreid"]))
            keep_indices.append(idx)
        except Exception as exc:
            failures.append(
                {
                    "row_index": int(idx),
                    "case_name": row["case_name"],
                    "coreid": row["coreid"],
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            if not args.drop_missing_cifs:
                raise

    if failures:
        output_dir = (script_dir / args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(output_dir / "cif_failures.csv", index=False, encoding="utf-8-sig")
        df = df.loc[keep_indices].reset_index(drop=True)
    geometry = np.stack(geometry_rows).astype(np.float32)
    geometry_cols = [f"geometry_{i:02d}" for i in range(geometry.shape[1])]
    geometry_df = pd.DataFrame(geometry, columns=geometry_cols)
    df = pd.concat([df.reset_index(drop=True), geometry_df], axis=1)
    return df, base_feature_cols, geometry_cols


def split_dataset(
    df: pd.DataFrame,
    ratios: list[float],
    seed: int,
    mode: str = "random",
    private_train_fraction: float = 0.0,
    private_train_count: int = 0,
):
    train_ratio, valid_ratio, test_ratio = ratios
    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"--split-ratios must sum to 1.0, got {total}")
    if mode == "common_holdout":
        if "is_common_benchmark" not in df.columns:
            raise KeyError("common_holdout split requires is_common_benchmark column")
        train_df = df[df["is_common_benchmark"]].copy()
        holdout_df = df[~df["is_common_benchmark"]].copy()
        if train_df.empty or holdout_df.empty:
            raise ValueError(
                f"common_holdout split failed: train rows={len(train_df)}, holdout rows={len(holdout_df)}"
            )

        valid_case_names = []
        test_case_names = []
        valid_share = valid_ratio / (valid_ratio + test_ratio)
        for gas_name, group in holdout_df.groupby("component1_name_norm", sort=True):
            cases = sorted(group["case_name"].dropna().astype(str).unique())
            if len(cases) < 2:
                valid_case_names.extend(cases)
                continue
            valid_cases, test_cases = train_test_split(
                cases,
                train_size=valid_share,
                random_state=seed,
                shuffle=True,
            )
            valid_case_names.extend(valid_cases)
            test_case_names.extend(test_cases)

        valid_set = set(valid_case_names)
        test_set = set(test_case_names)
        valid_df = holdout_df[holdout_df["case_name"].astype(str).isin(valid_set)].copy()
        test_df = holdout_df[holdout_df["case_name"].astype(str).isin(test_set)].copy()
        return (
            train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
    if mode == "common_plus_private":
        if "is_common_benchmark" not in df.columns:
            raise KeyError("common_plus_private split requires is_common_benchmark column")
        common_df = df[df["is_common_benchmark"]].copy()
        private_df = df[~df["is_common_benchmark"]].copy()
        if common_df.empty or private_df.empty:
            raise ValueError(
                f"common_plus_private split failed: common rows={len(common_df)}, private rows={len(private_df)}"
            )
        if private_train_count <= 0 and not (0.0 < private_train_fraction < 1.0):
            raise ValueError(
                "common_plus_private requires either --private-train-count > 0 "
                "or --private-train-fraction between 0 and 1"
            )

        private_train_cases = []
        remaining_cases = []
        for gas_name, group in private_df.groupby("component1_name_norm", sort=True):
            cases = sorted(group["case_name"].dropna().astype(str).unique())
            if len(cases) < 3:
                remaining_cases.extend(cases)
                continue
            if private_train_count > 0:
                n_train = min(private_train_count, max(1, len(cases) - 2))
            else:
                n_train = max(1, int(round(len(cases) * private_train_fraction)))
            train_cases, rest_cases = train_test_split(
                cases,
                train_size=n_train,
                random_state=seed,
                shuffle=True,
            )
            private_train_cases.extend(train_cases)
            remaining_cases.extend(rest_cases)

        remaining_df = private_df[private_df["case_name"].astype(str).isin(set(remaining_cases))].copy()
        valid_case_names = []
        test_case_names = []
        valid_share = valid_ratio / (valid_ratio + test_ratio)
        for gas_name, group in remaining_df.groupby("component1_name_norm", sort=True):
            cases = sorted(group["case_name"].dropna().astype(str).unique())
            if len(cases) < 2:
                valid_case_names.extend(cases)
                continue
            valid_cases, test_cases = train_test_split(
                cases,
                train_size=valid_share,
                random_state=seed,
                shuffle=True,
            )
            valid_case_names.extend(valid_cases)
            test_case_names.extend(test_cases)

        train_df = pd.concat(
            [
                common_df,
                private_df[private_df["case_name"].astype(str).isin(set(private_train_cases))],
            ],
            ignore_index=True,
        )
        valid_df = remaining_df[remaining_df["case_name"].astype(str).isin(set(valid_case_names))].copy()
        test_df = remaining_df[remaining_df["case_name"].astype(str).isin(set(test_case_names))].copy()
        return (
            train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
    if mode == "mof_random":
        if "coreid" not in df.columns:
            raise KeyError("mof_random split requires coreid column")
        mof_ids = sorted(df["coreid"].dropna().astype(str).unique())
        if len(mof_ids) < 3:
            raise ValueError(f"mof_random split requires at least 3 MOFs, got {len(mof_ids)}")
        train_mofs, holdout_mofs = train_test_split(
            mof_ids,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
        )
        valid_share = valid_ratio / (valid_ratio + test_ratio)
        valid_mofs, test_mofs = train_test_split(
            holdout_mofs,
            train_size=valid_share,
            random_state=seed,
            shuffle=True,
        )
        coreid = df["coreid"].astype(str)
        train_df = df[coreid.isin(set(train_mofs))].copy()
        valid_df = df[coreid.isin(set(valid_mofs))].copy()
        test_df = df[coreid.isin(set(test_mofs))].copy()
        return (
            train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
    if mode == "per_gas_mof_random":
        if "coreid" not in df.columns or "component1_name_norm" not in df.columns:
            raise KeyError("per_gas_mof_random split requires coreid and component1_name_norm columns")
        train_parts = []
        valid_parts = []
        test_parts = []
        for gas_index, (gas_name, group) in enumerate(df.groupby("component1_name_norm", sort=True)):
            mof_ids = sorted(group["coreid"].dropna().astype(str).unique())
            if len(mof_ids) < 3:
                raise ValueError(
                    f"per_gas_mof_random split requires at least 3 MOFs for {gas_name}, got {len(mof_ids)}"
                )
            train_mofs, holdout_mofs = train_test_split(
                mof_ids,
                train_size=train_ratio,
                random_state=seed + gas_index,
                shuffle=True,
            )
            valid_share = valid_ratio / (valid_ratio + test_ratio)
            valid_mofs, test_mofs = train_test_split(
                holdout_mofs,
                train_size=valid_share,
                random_state=seed + gas_index,
                shuffle=True,
            )
            coreid = group["coreid"].astype(str)
            train_parts.append(group[coreid.isin(set(train_mofs))].copy())
            valid_parts.append(group[coreid.isin(set(valid_mofs))].copy())
            test_parts.append(group[coreid.isin(set(test_mofs))].copy())
        train_df = pd.concat(train_parts, ignore_index=True)
        valid_df = pd.concat(valid_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        return (
            train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
    if mode != "random":
        raise ValueError(f"Unknown split mode: {mode}")
    train_df, holdout_df = train_test_split(df, train_size=train_ratio, random_state=seed, shuffle=True)
    valid_share = valid_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(holdout_df, train_size=valid_share, random_state=seed, shuffle=True)
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class ComponentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cols: list[str], target_cols: list[str]):
        self.x = torch.tensor(frame[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y = torch.tensor(frame[target_cols].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
