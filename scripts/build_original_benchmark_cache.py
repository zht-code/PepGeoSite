#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pepgeosite.pdb_utils import parse_pdb, radius_edges, sequence
from preprocess import SequenceEmbedder


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build PepGeoSite benchmark tensors with preserved distributed labels."
    )
    parser.add_argument("--benchmark", required=True, choices=["TS251", "TS639", "TS125"])
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--existing-manifest", required=True)
    parser.add_argument("--flex-label-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--esm-model", default="models/esm2_t6_8M_UR50D.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radii", type=float, nargs="+", default=[6.0, 10.0, 14.0])
    args = parser.parse_args()

    source = pd.read_csv(args.source_manifest).set_index("complex_id", drop=False)
    existing_frame = pd.read_csv(args.existing_manifest)
    existing = dict(zip(existing_frame.complex_id.astype(str), existing_frame.cache_path))
    flex_files = sorted(Path(args.flex_label_cache).glob("*.pt"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = None
    rows, failures = [], []

    for label_path in tqdm(flex_files, desc=f"original labels {args.benchmark}"):
        complex_id = label_path.stem
        try:
            if complex_id not in source.index:
                raise ValueError("complex absent from source manifest")
            row = source.loc[complex_id]
            label_obj = torch.load(label_path, map_location="cpu", weights_only=False)
            labels = label_obj["y"].float()
            receptor_seq = str(label_obj["protein_seq"])
            peptide_seq = str(label_obj["peptide_seq"])
            if not receptor_seq or not peptide_seq:
                raise ValueError("empty authoritative receptor or peptide sequence")

            target = output_dir / f"{complex_id}.pt"
            reused = False
            if complex_id in existing and Path(existing[complex_id]).is_file():
                payload = torch.load(existing[complex_id], map_location="cpu", weights_only=False)
                if (
                    str(payload.get("receptor_seq", "")) == receptor_seq
                    and int(payload["labels"].numel()) == int(labels.numel())
                ):
                    payload["labels"] = labels
                    payload["peptide_seq"] = peptide_seq
                    reused = True

            if not reused:
                receptor, _ = parse_pdb(row.receptor_path)
                parsed_receptor_seq = sequence(receptor)
                if parsed_receptor_seq != receptor_seq:
                    raise ValueError("authoritative label sequence differs from parsed receptor")
                if len(receptor) != labels.numel():
                    raise ValueError("authoritative label length differs from parsed receptor")
                if embedder is None:
                    embedder = SequenceEmbedder(args.esm_model, args.device, "esm2")
                coords = np.stack([residue.coord for residue in receptor]).astype(np.float32)
                edge_indices, edge_distances = zip(
                    *(radius_edges(coords, radius) for radius in args.radii)
                )
                payload = {
                    "receptor_esm": torch.from_numpy(embedder.encode(receptor_seq)).half(),
                    "peptide_esm": torch.from_numpy(embedder.encode(peptide_seq)).half(),
                    "coords": torch.from_numpy(coords),
                    "labels": labels,
                    "edge_indices": [torch.from_numpy(value) for value in edge_indices],
                    "edge_distances": [torch.from_numpy(value) for value in edge_distances],
                    "receptor_seq": receptor_seq,
                    "peptide_seq": peptide_seq,
                    "metadata": {
                        "radii": args.radii,
                        "esm_model": args.esm_model,
                        "esm_dim": embedder.dimension,
                        "representative_atom": "CA_or_heavy_atom_centroid",
                    },
                }

            metadata = dict(payload.get("metadata", {}))
            metadata.update({
                "benchmark": args.benchmark,
                "label_source": "FlexPepSite preserved distributed PepCA/MGAPep benchmark labels",
                "label_mode": str(label_obj.get("label_mode", "preserved")),
                "contact_labels_recomputed": False,
            })
            payload["metadata"] = metadata
            torch.save(payload, target)
            rows.append({
                "complex_id": complex_id,
                "pdb_id": str(row.pdb_id),
                "cache_path": str(target),
                "label_mode": metadata["label_mode"],
                "positives": int(labels.sum().item()),
                "length": int(labels.numel()),
                "features_reused": reused,
            })
        except Exception as error:
            failures.append({"complex_id": complex_id, "error": repr(error)})

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_manifest, index=False)
    pd.DataFrame(failures).to_csv(output_manifest.with_suffix(".failed.csv"), index=False)
    summary = {
        "benchmark": args.benchmark,
        "authoritative_label_caches": len(flex_files),
        "cached": len(rows),
        "failed": len(failures),
        "residues": int(sum(row["length"] for row in rows)),
        "positives": int(sum(row["positives"] for row in rows)),
        "reused_features": int(sum(bool(row["features_reused"]) for row in rows)),
    }
    print(json.dumps(summary, indent=2))
    return 0 if rows and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
