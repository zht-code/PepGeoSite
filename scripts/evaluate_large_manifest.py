#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from pepgeosite.dataset import CachedComplexDataset, collate_complexes
from pepgeosite.metrics import compute_metrics
from pepgeosite.model import PepGeoSite
from train import move_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-efficient pooled evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    threshold = checkpoint.get("metrics", {}).get("threshold")
    if threshold is None:
        raise ValueError("checkpoint has no validation-selected threshold")
    threshold = float(threshold)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
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
        pin_memory=True,
        collate_fn=collate_complexes,
    )
    label_chunks, probability_chunks = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=args.name):
            batch = move_batch(batch, device)
            probabilities = torch.sigmoid(model(batch)["logits"]).float().cpu().numpy()
            labels = batch["labels"].to(torch.uint8).cpu().numpy()
            label_chunks.append(labels)
            probability_chunks.append(probabilities.astype(np.float32, copy=False))

    labels = np.concatenate(label_chunks).astype(np.uint8, copy=False)
    probabilities = np.concatenate(probability_chunks).astype(np.float32, copy=False)
    metrics = compute_metrics(labels, probabilities, threshold=threshold)
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics.update({
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    })
    result = {
        "dataset": args.name,
        "complexes": len(dataset),
        "residues": int(labels.size),
        "positives": int(labels.sum()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "threshold_source": "training-validation checkpoint; not fitted on training-set evaluation",
        "aggregation": "pooled residue level",
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
