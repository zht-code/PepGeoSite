#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cached tensor/sequence alignment.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--esm-dim", type=int, default=320)
    args = parser.parse_args()

    report = {"manifests": {}, "critical_errors": []}
    for manifest in args.manifests:
        frame = pd.read_csv(manifest)
        stats = {
            "complexes": int(len(frame)), "residues": 0, "positives": 0,
            "duplicate_complex_ids": int(frame.complex_id.astype(str).duplicated().sum()),
            "duplicate_cache_paths": int(frame.cache_path.astype(str).duplicated().sum()),
            "nonfinite_embedding_values": 0, "nonfinite_coordinate_values": 0,
        }
        errors = []
        for row in frame.itertuples(index=False):
            path = Path(row.cache_path)
            item = torch.load(path, map_location="cpu", weights_only=False)
            receptor_len = int(item["receptor_esm"].shape[0])
            peptide_len = int(item["peptide_esm"].shape[0])
            checks = {
                "receptor_embedding_dim": int(item["receptor_esm"].shape[-1]) == args.esm_dim,
                "peptide_embedding_dim": int(item["peptide_esm"].shape[-1]) == args.esm_dim,
                "receptor_label_length": int(item["labels"].numel()) == receptor_len,
                "receptor_coordinate_length": int(item["coords"].shape[0]) == receptor_len,
                "receptor_sequence_length": len(item["receptor_seq"]) == receptor_len,
                "peptide_sequence_length": len(item["peptide_seq"]) == peptide_len,
                "binary_labels": bool(torch.all((item["labels"] == 0) | (item["labels"] == 1))),
            }
            if not all(checks.values()):
                errors.append({"complex_id": str(row.complex_id), "checks": checks})
            stats["residues"] += receptor_len
            stats["positives"] += int(item["labels"].sum().item())
            stats["nonfinite_embedding_values"] += int(
                (~torch.isfinite(item["receptor_esm"])).sum().item()
                + (~torch.isfinite(item["peptide_esm"])).sum().item()
            )
            stats["nonfinite_coordinate_values"] += int((~torch.isfinite(item["coords"])).sum().item())
        stats["positive_rate"] = stats["positives"] / max(1, stats["residues"])
        stats["alignment_errors"] = len(errors)
        report["manifests"][manifest] = stats
        report["critical_errors"].extend(errors[:100])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["critical_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
