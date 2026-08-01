from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import MoleculeCache, ThermalConductivityDataset, collate_batch, get_group_columns, read_thermal_table, resolve_condition_columns
from .chem import EXPLICIT_DESCRIPTOR_NAMES
from .model import GeometryAwareStateTransformer, MultibranchGeometryStateTransformer, ResidualDescriptorCorrectedTransformer
from .splits import apply_group_split, make_group_split
from .utils import (
    Standardizer,
    TargetTransform,
    WarmupCosine,
    load_json,
    load_yaml,
    regression_metrics,
    save_json,
    set_seed,
)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight / torch.clamp(weight.mean(), min=1e-12)
    return torch.mean(weight * (pred - target) ** 2)


def samplewise_loss(pred: torch.Tensor, target: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    loss_cfg = config.get("loss", {})
    kind = str(loss_cfg.get("kind", "mse"))
    if kind == "mse":
        return (pred - target) ** 2
    if kind == "smooth_l1":
        beta = float(loss_cfg.get("smooth_l1_beta", 0.5))
        return F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    if kind == "l1":
        return torch.abs(pred - target)
    raise ValueError(f"Unknown loss kind: {kind}")


def group_robust_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, groups: list[str], config: dict[str, Any]) -> torch.Tensor:
    loss_cfg = config.get("loss", {})
    losses = samplewise_loss(pred, target, config)
    weight = weight / torch.clamp(weight.mean(), min=1e-12)
    weighted_losses = losses * weight
    if not bool(loss_cfg.get("group_balanced", False)) and not bool(loss_cfg.get("group_dro", False)):
        return torch.mean(weighted_losses)

    unique_groups = list(dict.fromkeys(groups))
    group_losses = []
    for group in unique_groups:
        mask = torch.tensor([g == group for g in groups], dtype=torch.bool, device=pred.device)
        if mask.any():
            group_losses.append(weighted_losses[mask].mean())
    stacked = torch.stack(group_losses)
    mean_group_loss = stacked.mean()
    if bool(loss_cfg.get("group_dro", False)):
        dro_weight = float(loss_cfg.get("group_dro_weight", 0.5))
        worst_group_loss = stacked.max()
        return (1.0 - dro_weight) * mean_group_loss + dro_weight * worst_group_loss
    return mean_group_loss


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    architecture = str(config.get("model", {}).get("architecture", "geometry_transformer"))
    if architecture == "geometry_transformer":
        return GeometryAwareStateTransformer(config)
    if architecture == "multibranch_geometry_state_transformer":
        return MultibranchGeometryStateTransformer(config)
    if architecture == "residual_descriptor_corrected_transformer":
        return ResidualDescriptorCorrectedTransformer(config)
    raise ValueError(f"Unknown model architecture: {architecture}")


def training_loss(outputs: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, torch.Tensor], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(outputs, dict):
        pred = outputs["target"]
    else:
        pred = outputs
    main_loss = group_robust_loss(pred, batch["target"], batch["weight"], batch["refrigerant"], config)
    aux_loss = pred.new_tensor(0.0)
    aux_cfg = config.get("auxiliary", {})
    if isinstance(outputs, dict) and bool(aux_cfg.get("predict_critical_properties", False)):
        mask = batch["critical_mask"].bool()
        if mask.any():
            aux_loss = torch.mean((outputs["critical"][mask] - batch["critical_target"][mask]) ** 2)
    loss = main_loss + float(aux_cfg.get("critical_loss_weight", 0.0)) * aux_loss
    return loss, {"main_loss": float(main_loss.detach().cpu()), "critical_loss": float(aux_loss.detach().cpu())}


def monotonicity_loss(model: torch.nn.Module, batch: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    mono_cfg = config.get("monotonicity", {})
    if not bool(mono_cfg.get("enabled", False)):
        target = batch["target"]
        return target.new_tensor(0.0)
    condition_cols = list(config["data"].get("condition_cols", [config["data"]["temperature_col"], config["data"]["pressure_col"]]))
    temperature_col = str(config["data"].get("temperature_col", "T"))
    pressure_col = str(config["data"].get("pressure_col", "p Mpa"))
    if temperature_col not in condition_cols or pressure_col not in condition_cols:
        return batch["target"].new_tensor(0.0)

    delta = float(mono_cfg.get("delta_standardized", 0.05))
    t_index = condition_cols.index(temperature_col)
    p_index = condition_cols.index(pressure_col)
    base_pred = model(batch)
    batch_t = dict(batch)
    batch_p = dict(batch)
    batch_t["condition"] = batch["condition"].clone()
    batch_p["condition"] = batch["condition"].clone()
    batch_t["condition"][:, t_index] = batch_t["condition"][:, t_index] + delta
    batch_p["condition"][:, p_index] = batch_p["condition"][:, p_index] + delta
    pred_t_plus = model(batch_t)
    pred_p_plus = model(batch_p)
    margin = float(mono_cfg.get("margin", 0.0))
    temp_penalty = torch.relu(pred_t_plus - base_pred + margin) ** 2
    press_penalty = torch.relu(base_pred - pred_p_plus + margin) ** 2
    return temp_penalty.mean() + press_penalty.mean()


@torch.no_grad()
def evaluate_batches(
    model: torch.nn.Module,
    loader: DataLoader,
    target_transform: TargetTransform,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    y_true, y_pred, groups = [], [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        pred_t = model(batch).detach().cpu().numpy()
        pred = target_transform.inverse(pred_t)
        true = batch["target_raw"].detach().cpu().numpy()
        y_true.extend(true.tolist())
        y_pred.extend(pred.tolist())
        groups.extend(batch["refrigerant"])
    metrics = regression_metrics(np.asarray(y_true), np.asarray(y_pred))
    frame = pd.DataFrame({"refrigerant": groups, "target": y_true, "prediction": y_pred})
    return metrics, frame


def build_loaders(
    config: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    condition_standardizers: dict[str, Standardizer],
    target_transform: TargetTransform,
    molecule_cache: MoleculeCache,
    auxiliary_standardizers: dict[str, Standardizer] | None = None,
    descriptor_standardizers: dict[str, Standardizer] | None = None,
    group_standardizers: dict[str, Standardizer] | None = None,
) -> dict[str, DataLoader]:
    train_cfg = config["train"]
    loaders = {}
    for name, frame in split_frames.items():
        ds = ThermalConductivityDataset(
            frame,
            config=config,
            condition_standardizers=condition_standardizers,
            target_transform=target_transform,
            molecule_cache=molecule_cache,
            auxiliary_standardizers=auxiliary_standardizers,
            descriptor_standardizers=descriptor_standardizers,
            group_standardizers=group_standardizers,
        )
        sampler = None
        shuffle = name == "train"
        if name == "train" and str(config["train"].get("sampler", "shuffle")) == "balanced_refrigerant":
            counts = frame["refrigerant"].value_counts().to_dict()
            weights = frame["refrigerant"].map(lambda x: 1.0 / float(counts[x])).to_numpy(dtype=np.float64).copy()
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
        loaders[name] = DataLoader(
            ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
    return loaders


def run_training(config: dict[str, Any], resume_split: str | None = None) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    out_dir = Path(config["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, out_dir / "config.resolved.json")

    df = read_thermal_table(config)
    split_cfg = config["split"]
    if resume_split:
        split = load_json(resume_split)
    else:
        split = make_group_split(
            df,
            group_col="refrigerant",
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            seed=int(config["seed"]),
            forced_val_groups=split_cfg.get("forced_val_groups", []),
            forced_test_groups=split_cfg.get("forced_test_groups", []),
        )
    save_json(split, out_dir / "split.json")
    split_frames = apply_group_split(df, "refrigerant", split)
    molecule_cache = MoleculeCache(config)
    for refrigerant in sorted(df["refrigerant"].unique()):
        molecule_cache.get(str(refrigerant))
    save_json(molecule_cache.paths, out_dir / "molecule_paths.json")

    train_df = split_frames["train"]
    condition_cols = resolve_condition_columns(config, train_df)
    condition_standardizers = {
        col: Standardizer.fit(train_df[col].to_numpy())
        for col in condition_cols
    }
    target_transform = TargetTransform.fit(
        train_df["target"].to_numpy(),
        mode=config["model"].get("target_transform", "log_standard"),
    )
    auxiliary_standardizers = None
    if bool(config.get("auxiliary", {}).get("predict_critical_properties", False)):
        aux_fit_df = train_df.drop_duplicates("refrigerant")
        auxiliary_standardizers = {
            "critical_temperature": Standardizer.fit(aux_fit_df["critical_temperature"].to_numpy()),
            "critical_pressure": Standardizer.fit(aux_fit_df["critical_pressure"].to_numpy()),
        }
    descriptor_standardizers = None
    if bool(config["molecule"].get("use_explicit_graph_descriptors", False)):
        train_groups = sorted(train_df["refrigerant"].unique().tolist())
        descriptor_matrix = np.stack([molecule_cache.get(str(group)).explicit_descriptor for group in train_groups], axis=0)
        descriptor_standardizers = {
            f"desc_{i}": Standardizer.fit(descriptor_matrix[:, i])
            for i in range(descriptor_matrix.shape[1])
        }
    group_standardizers = None
    if bool(config.get("model", {}).get("use_group_features", False)):
        group_cols = get_group_columns(config)
        group_fit_df = train_df.drop_duplicates("refrigerant")
        group_standardizers = {col: Standardizer.fit(group_fit_df[col].to_numpy()) for col in group_cols}
    transforms = {
        "condition": {k: v.to_dict() for k, v in condition_standardizers.items()},
        "target": target_transform.to_dict(),
    }
    if auxiliary_standardizers:
        transforms["auxiliary"] = {k: v.to_dict() for k, v in auxiliary_standardizers.items()}
    if descriptor_standardizers:
        transforms["explicit_descriptor"] = {
            "names": EXPLICIT_DESCRIPTOR_NAMES,
            "standardizers": {k: v.to_dict() for k, v in descriptor_standardizers.items()},
        }
    if group_standardizers:
        transforms["group_feature"] = {
            "names": get_group_columns(config),
            "standardizers": {k: v.to_dict() for k, v in group_standardizers.items()},
        }
    save_json(transforms, out_dir / "transforms.json")

    split_summary = {
        name: {
            "rows": int(len(frame)),
            "groups": sorted(frame["refrigerant"].unique().tolist()),
        }
        for name, frame in split_frames.items()
    }
    save_json(split_summary, out_dir / "split_summary.json")
    print("Strict refrigerant-level split:")
    for name, info in split_summary.items():
        print(f"  {name}: {info['rows']} rows, {len(info['groups'])} refrigerants -> {info['groups']}")

    loaders = build_loaders(
        config,
        split_frames,
        condition_standardizers,
        target_transform,
        molecule_cache,
        auxiliary_standardizers,
        descriptor_standardizers,
        group_standardizers,
    )
    device_name = str(config["train"].get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = build_model(config).to(device)
    pretrained_checkpoint = str(config["train"].get("pretrained_checkpoint", "")).strip()
    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        state = checkpoint.get("model_state", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained checkpoint: {pretrained_checkpoint}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    total_steps = int(config["train"]["epochs"]) * max(1, len(loaders["train"]))
    warmup_steps = int(config["train"].get("warmup_epochs", 0)) * max(1, len(loaders["train"]))
    scheduler = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["train"].get("amp", True)) and device.type == "cuda")

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        losses = []
        main_losses = []
        critical_losses = []
        monotonicity_losses = []
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(batch, return_aux=True)
                loss, loss_parts = training_loss(outputs, batch, config)
                mono_loss = monotonicity_loss(model, batch, config)
                loss = loss + float(config.get("monotonicity", {}).get("weight", 0.0)) * mono_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            main_losses.append(loss_parts["main_loss"])
            critical_losses.append(loss_parts["critical_loss"])
            monotonicity_losses.append(float(mono_loss.detach().cpu()))
            pbar.set_postfix(loss=np.mean(losses), main=np.mean(main_losses), mono=np.mean(monotonicity_losses), crit=np.mean(critical_losses))

        val_metrics, _ = evaluate_batches(model, loaders["val"], target_transform, device)
        test_metrics, _ = evaluate_batches(model, loaders["test"], target_transform, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_main_loss": float(np.mean(main_losses)),
            "train_critical_loss": float(np.mean(critical_losses)),
            "train_monotonicity_loss": float(np.mean(monotonicity_losses)),
            "val": val_metrics,
            "test_monitor_only": test_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        save_json(history, out_dir / "history.json")
        print(
            f"epoch={epoch:04d} loss={record['train_loss']:.6f} "
            f"val_RMSE={val_metrics['RMSE']:.6g} val_MAPE={val_metrics['MAPE_percent']:.3f}%"
        )

        if val_metrics["RMSE"] < best_val:
            best_val = val_metrics["RMSE"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(
                {
                    "model_state": best_state,
                    "config": config,
                    "split": split,
                    "transforms": transforms,
                    "best_epoch": epoch,
                    "best_val_rmse": best_val,
                },
                out_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["train"]["patience"]):
                print(f"Early stopping at epoch {epoch}. Best val RMSE: {best_val:.6g}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final = {}
    for split_name in ("train", "val", "test"):
        metrics, pred_frame = evaluate_batches(model, loaders[split_name], target_transform, device)
        final[split_name] = metrics
        r1216 = pred_frame[pred_frame["refrigerant"] == "R1216"]
        if not r1216.empty:
            final["R1216"] = regression_metrics(
                r1216["target"].to_numpy(dtype=float),
                r1216["prediction"].to_numpy(dtype=float),
            )
            final["R1216_split"] = split_name
        pred_frame.to_csv(out_dir / f"predictions_{split_name}.csv", index=False)
    save_json(final, out_dir / "final_metrics.json")
    print("Final metrics:", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r1216_geometry_transformer.yaml")
    parser.add_argument("--resume-split", default=None, help="Optional existing split.json to reproduce a split.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_training(config, resume_split=args.resume_split)


if __name__ == "__main__":
    main()
