#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from preprocess import SequenceEmbedder


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def atomic_save(value, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace ESM tensors while preserving labels and graphs.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--sequence-cache", required=True)
    parser.add_argument("--esm-model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = pd.read_csv(args.source_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_cache = Path(args.sequence_cache)
    sequence_cache.mkdir(parents=True, exist_ok=True)
    embedder = SequenceEmbedder(args.esm_model, args.device, "esm2")
    rows = []

    def embedding(sequence: str) -> torch.Tensor:
        target = sequence_cache / f"{sequence_key(sequence)}.pt"
        if not target.exists():
            tensor = torch.from_numpy(embedder.encode(sequence)).half().cpu()
            atomic_save(tensor, target)
        return torch.load(target, map_location="cpu", weights_only=True)

    for row in tqdm(source.itertuples(index=False), total=len(source), desc="re-embed"):
        target = output_dir / f"{row.complex_id}.pt"
        if not target.exists():
            item = torch.load(row.cache_path, map_location="cpu", weights_only=False)
            receptor_seq = str(item["receptor_seq"])
            peptide_seq = str(item["peptide_seq"])
            item["receptor_esm"] = embedding(receptor_seq)
            item["peptide_esm"] = embedding(peptide_seq)
            item.setdefault("metadata", {})
            item["metadata"].update({
                "esm_model": args.esm_model,
                "esm_dim": int(embedder.dimension),
                "reembedded_from": str(row.cache_path),
                "labels_and_graphs_preserved": True,
            })
            atomic_save(item, target)
        rows.append({"complex_id": row.complex_id, "pdb_id": row.pdb_id, "cache_path": str(target)})

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_manifest, index=False)
    print({"source": len(source), "output": len(rows), "esm_dim": embedder.dimension})


if __name__ == "__main__":
    main()
