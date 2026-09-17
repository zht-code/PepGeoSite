#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from pepgeosite.dataset import CachedComplexDataset, collate_complexes
from pepgeosite.metrics import compute_metrics
from pepgeosite.model import PepGeoSite
from train import move_batch


def add_confusion_metrics(metrics: dict, labels, probabilities, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics.update(
        {
            "sensitivity": float(tp / max(1, tp + fn)),
            "specificity": float(tn / max(1, tn + fp)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PepGeoSite without fitting on test data.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--sota-reference", default="configs/sota_reference.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Use best.pt (full checkpoint), not the weights-only file.")
    config = checkpoint["config"]
    validation_metrics = checkpoint.get("metrics", {})
    threshold = args.threshold
    if threshold is None:
        threshold = validation_metrics.get("threshold")
    if threshold is None:
        raise ValueError("No validation-selected threshold in checkpoint; pass --threshold explicitly.")
    threshold = float(threshold)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = PepGeoSite(
        esm_dim=config["data"]["esm_dim"], scales=len(config["data"]["radii"]),
        **config["model"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = CachedComplexDataset(args.manifest)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_complexes,
    )
    all_labels, all_probabilities, prediction_rows, complex_rows = [], [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=args.benchmark_name):
            batch = move_batch(batch, device)
            probabilities = torch.sigmoid(model(batch)["logits"]).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            batch_index = batch["batch_index"].cpu().numpy()
            all_labels.extend(labels.tolist())
            all_probabilities.extend(probabilities.tolist())
            for local_index, complex_id in enumerate(batch["complex_ids"]):
                keep = batch_index == local_index
                local_labels = labels[keep]
                local_probabilities = probabilities[keep]
                local_metrics = add_confusion_metrics(
                    compute_metrics(local_labels, local_probabilities, threshold=threshold),
                    local_labels, local_probabilities, threshold,
                )
                local_metrics.update({"complex_id": complex_id, "residues": int(keep.sum())})
                complex_rows.append(local_metrics)
                for residue_index, (label, probability) in enumerate(
                    zip(local_labels, local_probabilities), start=1
                ):
                    prediction_rows.append(
                        {
                            "complex_id": complex_id, "residue_index": residue_index,
                            "label": int(label), "probability": float(probability),
                            "prediction": int(probability >= threshold),
                        }
                    )

    pooled = add_confusion_metrics(
        compute_metrics(all_labels, all_probabilities, threshold=threshold),
        all_labels, all_probabilities, threshold,
    )
    complex_frame = pd.DataFrame(complex_rows)
    macro_fields = ["auroc", "auprc", "mcc", "precision", "sensitivity", "specificity", "f1"]
    macro = {
        field: float(np.nanmean(complex_frame[field].to_numpy(dtype=float)))
        for field in macro_fields
    }
    with open(args.sota_reference, "r", encoding="utf-8") as handle:
        references = json.load(handle)
    reference = references.get(args.benchmark_name, {})
    comparison_fields = ["auroc", "mcc", "precision", "sensitivity", "specificity"]
    delta = {
        field: float(pooled[field] - reference[field])
        for field in comparison_fields if field in reference
    }
    result = {
        "benchmark": args.benchmark_name,
        "actual_complexes": len(dataset),
        "nominal_complexes_in_paper": reference.get("nominal_n"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "threshold_source": "training-validation checkpoint; not fitted on benchmark",
        "aggregation_primary": "pooled residue level",
        "pooled": pooled,
        "complex_macro": macro,
        "paper_table1_reference": reference,
        "delta_from_paper_reference": delta,
        "protocol_note": (
            "Read-only evaluation of the executable server manifest. Nominal and actual sample "
            "counts are both retained; missing structures are not imputed and test metrics do not "
            "select the checkpoint or threshold."
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.benchmark_name}.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    complex_frame.to_csv(output_dir / f"{args.benchmark_name}.per_complex.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / f"{args.benchmark_name}.per_residue.csv", index=False
    )
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
