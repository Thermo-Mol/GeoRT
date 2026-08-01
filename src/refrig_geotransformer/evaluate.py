from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import MoleculeCache, ThermalConductivityDataset, collate_batch, read_thermal_table
from .model import GeometryAwareStateTransformer
from .splits import apply_group_split
from .train import evaluate_batches
from .utils import Standardizer, TargetTransform, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    transforms = ckpt["transforms"]
    condition_standardizers = {
        key: Standardizer.from_dict(value)
        for key, value in transforms["condition"].items()
    }
    target_transform = TargetTransform.from_dict(transforms["target"])
    descriptor_standardizers = None
    if "explicit_descriptor" in transforms:
        descriptor_standardizers = {
            key: Standardizer.from_dict(value)
            for key, value in transforms["explicit_descriptor"]["standardizers"].items()
        }

    df = read_thermal_table(config)
    split_frames = apply_group_split(df, "refrigerant", ckpt["split"])
    molecule_cache = MoleculeCache(config)
    ds = ThermalConductivityDataset(
        split_frames[args.split],
        config,
        condition_standardizers,
        target_transform,
        molecule_cache,
        descriptor_standardizers=descriptor_standardizers,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryAwareStateTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    metrics, pred_frame = evaluate_batches(model, loader, target_transform, device)

    output = Path(args.output) if args.output else Path(args.checkpoint).with_name(f"eval_{args.split}.json")
    save_json(metrics, output)
    pred_frame.to_csv(output.with_suffix(".csv"), index=False)
    print(metrics)


if __name__ == "__main__":
    main()
