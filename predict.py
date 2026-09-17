#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pepgeosite.dataset import CachedComplexDataset, collate_complexes
from pepgeosite.model import PepGeoSite
from train import move_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run residue-level PepGeoSite inference on a cached manifest."
    )
    parser.add_argument("--checkpoint", required=True, help="Full best.pt checkpoint.")
    parser.add_argument("--manifest", required=True, help="CSV produced by preprocess.py.")
    parser.add_argument("--output", required=True, help="Output residue-level CSV.")
    parser.add_argument("--threshold", type=float, help="Override checkpoint threshold.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Expected a full best.pt checkpoint containing model and config.")

    config = checkpoint["config"]
    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint.get("metrics", {}).get("threshold")
    if threshold is None:
        raise ValueError("Checkpoint has no validation threshold; pass --threshold explicitly.")
    threshold = float(threshold)

    configured_device = config.get("device", "cuda")
    device = torch.device(configured_device if torch.cuda.is_available() else "cpu")
    model = PepGeoSite(
        esm_dim=config["data"]["esm_dim"],
        scales=len(config["data"]["radii"]),
        **config["model"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = CachedComplexDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_complexes,
    )

    rows: list[dict] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="predict"):
            batch = move_batch(batch, device)
            probabilities = torch.sigmoid(model(batch)["logits"]).cpu()
            batch_index = batch["batch_index"].cpu()
            for local_index, complex_id in enumerate(batch["complex_ids"]):
                local_probabilities = probabilities[batch_index == local_index]
                for residue_index, probability in enumerate(local_probabilities.tolist(), start=1):
                    rows.append(
                        {
                            "complex_id": complex_id,
                            "residue_index": residue_index,
                            "probability": probability,
                            "prediction": int(probability >= threshold),
                            "threshold": threshold,
                        }
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} residue predictions for {len(dataset)} complexes to {output}")


if __name__ == "__main__":
    main()
