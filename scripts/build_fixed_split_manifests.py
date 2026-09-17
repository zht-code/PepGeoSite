#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the deterministic PepGeoSite split.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    groups = frame.get("pdb_id", frame.complex_id.astype(str).str.split("_").str[0])
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.val_fraction, random_state=args.seed
    )
    train_index, valid_index = next(splitter.split(frame, groups=groups))
    train_frame = frame.iloc[train_index].copy()
    valid_frame = frame.iloc[valid_index].copy()
    train_groups = set(groups.iloc[train_index].astype(str))
    valid_groups = set(groups.iloc[valid_index].astype(str))
    overlap = sorted(train_groups & valid_groups)
    if overlap:
        raise ValueError(f"group leakage detected: {overlap[:10]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(output_dir / "train.csv", index=False)
    valid_frame.to_csv(output_dir / "valid.csv", index=False)
    summary = {
        "source_manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_complexes": len(train_frame),
        "valid_complexes": len(valid_frame),
        "train_pdb_groups": len(train_groups),
        "valid_pdb_groups": len(valid_groups),
        "pdb_group_overlap": len(overlap),
    }
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
