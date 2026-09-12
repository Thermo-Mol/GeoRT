import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models import AdsorptionRegressor, TabularTransformerRegressor
from tasks import ComponentDataset, build_dataset, split_dataset


TARGET_COLS = ["target_component0_uptake_molkg", "target_component1_uptake_molkg"]
SCALED_TARGET_COLS = ["target0_scaled", "target1_scaled"]


def target_columns(target_component: str) -> list[str]:
    if target_component in {"co2", "component0"}:
        return ["target_component0_uptake_molkg"]
    if target_component in {"x", "component1"}:
        return ["target_component1_uptake_molkg"]
    return TARGET_COLS.copy()


def target_label_names(target_component: str, count: int) -> list[str]:
    if count == 1:
        return ["component1"] if target_component in {"x", "component1"} else ["component0"]
    return ["component0", "component1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train/valid/test predictions from the best checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/component_adsorption_geometry/checkpoint_best.pt")
    parser.add_argument("--source-xlsx", default="data/demo_prediction_input.xlsx")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--descriptors-xlsx", default="descriptors_example.xlsx")
    parser.add_argument("--mof-chem-csv", default="mof_local_chemical_descriptors_example.csv")
    parser.add_argument(
        "--extra-mof-chem-csvs",
        default="",
        help="Comma-separated extra MOF descriptor CSV files in --data-dir, e.g. mof_forcefield_chemical_descriptors.csv.",
    )
    parser.add_argument("--gas-xlsx", default="gas.xlsx")
    parser.add_argument("--structure-csv", default="data/CR_data_CSD_modified_20250227.csv")
    parser.add_argument("--cif-dir", required=True)
    parser.add_argument(
        "--component1-filter",
        default="all",
        help="Export predictions for only one CO2/X system, e.g. C2H4, C3H6, propane, CH4, H2, N2, or O2.",
    )
    parser.add_argument("--output-dir", default="prediction_outputs")
    parser.add_argument(
        "--split-mode",
        choices=["random", "mof_random", "per_gas_mof_random", "common_holdout", "common_plus_private"],
        default="random",
    )
    parser.add_argument("--private-train-fraction", type=float, default=0.0)
    parser.add_argument(
        "--private-train-count",
        type=int,
        default=0,
        help="For common_plus_private, fixed number of private MOFs per gas added to train.",
    )
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.7, 0.2, 0.1])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model-type", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--coupled-co2-head", action="store_true")
    parser.add_argument("--use-mof-encoder", action="store_true")
    parser.add_argument("--mof-pretrained-encoder", default="")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--drop-missing-cifs", action="store_true")
    parser.add_argument("--target-transform", choices=["none", "log1p", "signed_log1p"], default="log1p")
    parser.add_argument(
        "--target-mode",
        choices=["direct", "x_delta"],
        default="direct",
        help="direct predicts [CO2, X]; x_delta predicts [CO2-X, X] and reconstructs CO2 during export.",
    )
    parser.add_argument(
        "--target-component",
        choices=["both", "co2", "component0", "x", "component1"],
        default="both",
        help="both exports [CO2, X]; co2/component0 exports only CO2; x/component1 exports only the second gas.",
    )
    parser.add_argument("--target-scale-mode", choices=["global", "component1_gas"], default="global")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_target(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    if mode == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    return values


def inverse_target(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(values)
    if mode == "signed_log1p":
        return np.sign(values) * np.expm1(np.abs(values))
    return values


def to_model_targets(values: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "direct":
        return values
    if target_mode == "x_delta":
        co2 = values[:, 0]
        x = values[:, 1]
        return np.stack([co2 - x, x], axis=1).astype(np.float32)
    raise ValueError(f"Unknown target mode: {target_mode}")


def from_model_targets(values: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "direct":
        return values
    if target_mode == "x_delta":
        delta = values[:, 0]
        x = values[:, 1]
        return np.stack([delta + x, x], axis=1).astype(np.float32)
    raise ValueError(f"Unknown target mode: {target_mode}")


def fit_target_scaling(train_df, frames, target_transform: str, scale_mode: str, target_mode: str, target_component: str):
    target_cols = target_columns(target_component)
    if len(target_cols) == 1 and target_mode != "direct":
        raise ValueError("--target-mode x_delta is only valid when --target-component both")
    model_cols = [f"target{i}_model" for i in range(len(target_cols))]
    scaled_target_cols = [f"target{i}_scaled" for i in range(len(target_cols))]
    target_mean_cols = [f"target_mean{i}" for i in range(len(target_cols))]
    target_std_cols = [f"target_std{i}" for i in range(len(target_cols))]
    for frame in frames:
        raw_target = frame[target_cols].to_numpy(dtype=np.float32)
        model_target = raw_target if len(target_cols) == 1 else to_model_targets(raw_target, target_mode)
        frame[model_cols] = transform_target(model_target, target_transform)

    global_mean = train_df[model_cols].mean(axis=0).to_numpy(dtype=np.float32)
    global_std = train_df[model_cols].std(axis=0).to_numpy(dtype=np.float32)
    global_std = np.where(np.abs(global_std) < 1e-8, 1.0, global_std).astype(np.float32)

    if scale_mode == "global":
        for frame in frames:
            mean = np.tile(global_mean.reshape(1, -1), (len(frame), 1))
            std = np.tile(global_std.reshape(1, -1), (len(frame), 1))
            frame[target_mean_cols] = mean
            frame[target_std_cols] = std
            frame[scaled_target_cols] = ((frame[model_cols].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
        return

    if scale_mode != "component1_gas":
        raise ValueError(f"Unknown target scale mode: {scale_mode}")

    stats = {}
    for gas_name, group in train_df.groupby("component1_name_norm", sort=True):
        mean = group[model_cols].mean(axis=0).to_numpy(dtype=np.float32)
        std = group[model_cols].std(axis=0).to_numpy(dtype=np.float32)
        std = np.where(np.abs(std) < 1e-8, 1.0, std).astype(np.float32)
        stats[str(gas_name)] = (mean, std)

    for frame in frames:
        means = []
        stds = []
        for gas_name in frame["component1_name_norm"].astype(str):
            mean, std = stats.get(gas_name, (global_mean, global_std))
            means.append(mean)
            stds.append(std)
        mean_arr = np.asarray(means, dtype=np.float32)
        std_arr = np.asarray(stds, dtype=np.float32)
        frame[target_mean_cols] = mean_arr
        frame[target_std_cols] = std_arr
        frame[scaled_target_cols] = ((frame[model_cols].to_numpy(dtype=np.float32) - mean_arr) / std_arr).astype(np.float32)


def prepare_splits(args: argparse.Namespace):
    df, numeric_cols, geometry_cols = build_dataset(args)
    feature_cols = [*numeric_cols, *geometry_cols]
    train_df, valid_df, test_df = split_dataset(
        df,
        args.split_ratios,
        args.random_state,
        mode=args.split_mode,
        private_train_fraction=args.private_train_fraction,
        private_train_count=args.private_train_count,
    )

    x_scaler = StandardScaler()
    train_df[feature_cols] = x_scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float)).astype(np.float32)
    valid_df[feature_cols] = x_scaler.transform(valid_df[feature_cols].to_numpy(dtype=float)).astype(np.float32)
    test_df[feature_cols] = x_scaler.transform(test_df[feature_cols].to_numpy(dtype=float)).astype(np.float32)

    fit_target_scaling(
        train_df,
        [train_df, valid_df, test_df],
        target_transform=args.target_transform,
        scale_mode=args.target_scale_mode,
        target_mode=args.target_mode,
        target_component=args.target_component,
    )

    return {"train": train_df, "valid": valid_df, "test": test_df}, feature_cols


def resolve_checkpoint(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required best checkpoint not found: {path}")
    return path


def load_state_dict(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned[key] = value
    return cleaned


def predict(
    model,
    frame: pd.DataFrame,
    feature_cols: list[str],
    scaled_target_cols: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(ComponentDataset(frame, feature_cols, scaled_target_cols), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def split_metrics(frame: pd.DataFrame, pred_raw: np.ndarray, target_component: str) -> dict:
    target_cols = target_columns(target_component)
    y_true = frame[target_cols].to_numpy(dtype=float)
    result = {
        "overall": metric_block(y_true, pred_raw),
        "rows": int(len(frame)),
    }
    for idx, name in enumerate(target_label_names(target_component, len(target_cols))):
        result[name] = metric_block(y_true[:, idx], pred_raw[:, idx])
    return result


def output_prediction_frame(frame: pd.DataFrame, pred_raw: np.ndarray, target_component: str) -> pd.DataFrame:
    out = frame.copy()
    label_names = target_label_names(target_component, pred_raw.shape[1])
    for idx, name in enumerate(label_names):
        if name == "component0":
            source_col = "target_component0_uptake_molkg"
        else:
            source_col = "target_component1_uptake_molkg"
        out[f"{name}_uptake_molkg_true"] = out[source_col]
        out[f"{name}_uptake_molkg_pred"] = pred_raw[:, idx]
        out[f"{name}_uptake_error"] = np.abs(
            out[f"{name}_uptake_molkg_true"] - out[f"{name}_uptake_molkg_pred"]
        )
    keep = [
        "dataset_name",
        "case_name",
        "coreid",
        "component0_name",
        "component1_name",
        "temperature_K",
        "pressure_Pa",
        "log_pressure_Pa",
        "component0_mol_fraction",
        "component1_mol_fraction",
        "mol_ratio",
        "component0_uptake_molkg_true",
        "component0_uptake_molkg_pred",
        "component0_uptake_error",
        "component1_uptake_molkg_true",
        "component1_uptake_molkg_pred",
        "component1_uptake_error",
    ]
    return out[[column for column in keep if column in out.columns]]


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = (script_dir / args.checkpoint).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(checkpoint_path)

    frames, feature_cols = prepare_splits(args)
    output_dim = len(target_columns(args.target_component))
    if args.model_type == "transformer":
        model = TabularTransformerRegressor(
            input_dim=len(feature_cols),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            output_dim=output_dim,
            transformer_layers=args.transformer_layers,
            attention_heads=args.attention_heads,
            ffn_dim=args.ffn_dim,
        ).to(device)
    else:
        model = AdsorptionRegressor(
            len(feature_cols),
            args.hidden_dim,
            args.dropout,
            output_dim=output_dim,
            coupled_co2_head=args.coupled_co2_head,
            feature_cols=feature_cols,
            use_mof_encoder=args.use_mof_encoder,
            mof_pretrained_encoder=args.mof_pretrained_encoder,
        ).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path), strict=True)

    all_metrics = {
        "checkpoint": str(checkpoint_path),
        "target_transform": args.target_transform,
        "target_component": args.target_component,
        "feature_count": len(feature_cols),
        "splits": {},
    }
    scaled_target_cols = [f"target{i}_scaled" for i in range(output_dim)]
    target_mean_cols = [f"target_mean{i}" for i in range(output_dim)]
    target_std_cols = [f"target_std{i}" for i in range(output_dim)]
    for split, frame in frames.items():
        pred_scaled = predict(model, frame, feature_cols, scaled_target_cols, args.batch_size, device)
        target_std = frame[target_std_cols].to_numpy(dtype=np.float32)
        target_mean = frame[target_mean_cols].to_numpy(dtype=np.float32)
        pred_model = pred_scaled * target_std + target_mean
        pred_inverse = inverse_target(pred_model, args.target_transform)
        pred_raw = pred_inverse if output_dim == 1 else from_model_targets(pred_inverse, args.target_mode)
        pred_raw = np.clip(pred_raw, a_min=0.0, a_max=None)
        all_metrics["splits"][split] = split_metrics(frame, pred_raw, args.target_component)
        output_prediction_frame(frame, pred_raw, args.target_component).to_csv(
            output_dir / f"predictions_{split}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    with (output_dir / "metrics_best.json").open("w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2, ensure_ascii=False)
    summary_rows = []
    for split, split_info in all_metrics["splits"].items():
        for group, values in split_info.items():
            if group == "rows":
                continue
            summary_rows.append({"split": split, "target": group, **values, "rows": split_info["rows"]})
    pd.DataFrame(summary_rows).to_csv(output_dir / "metrics_best.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"Saved prediction CSVs and metrics to {output_dir}")


if __name__ == "__main__":
    main()
