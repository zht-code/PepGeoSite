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

from pepgeosite.pdb_utils import contact_labels, parse_pdb, radius_edges, sequence


AA = "ACDEFGHIKLMNPQRSTVWYX"


class SequenceEmbedder:
    def __init__(self, model_name: str, device: str, mode: str):
        self.mode = mode
        self.device = torch.device(device)
        if mode == "esm2":
            checkpoint = Path(model_name)
            if checkpoint.exists() and checkpoint.suffix == ".pt":
                import esm

                self.backend = "fair-esm"
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
                self.model = self.model.eval().to(self.device)
                self.batch_converter = self.alphabet.get_batch_converter()
                self.representation_layer = int(self.model.num_layers)
                self.dimension = int(self.model.embed_dim)
            else:
                from transformers import AutoModel, AutoTokenizer

                self.backend = "transformers"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
                self.dimension = int(self.model.config.hidden_size)
        else:
            self.backend = "onehot"
            self.dimension = len(AA)

    @torch.inference_mode()
    def _encode_chunk(self, sequence_text: str) -> np.ndarray:
        if self.backend == "fair-esm":
            _, _, tokens = self.batch_converter([("sequence", sequence_text)])
            tokens = tokens.to(self.device)
            output = self.model(tokens, repr_layers=[self.representation_layer], return_contacts=False)
            hidden = output["representations"][self.representation_layer][0, 1 : len(sequence_text) + 1]
            return hidden.float().cpu().numpy()
        encoded = self.tokenizer(
            sequence_text, return_tensors="pt", add_special_tokens=True,
            truncation=False,
        ).to(self.device)
        hidden = self.model(**encoded).last_hidden_state[0, 1:-1]
        return hidden.float().cpu().numpy()

    def encode(self, sequence_text: str, chunk_size: int = 1000, overlap: int = 128) -> np.ndarray:
        sequence_text = str(sequence_text).upper().replace(" ", "")
        if self.mode == "onehot":
            result = np.zeros((len(sequence_text), len(AA)), dtype=np.float32)
            for index, amino_acid in enumerate(sequence_text):
                result[index, AA.find(amino_acid) if amino_acid in AA else -1] = 1.0
            return result
        if len(sequence_text) <= chunk_size:
            return self._encode_chunk(sequence_text)
        result = np.zeros((len(sequence_text), self.dimension), dtype=np.float32)
        counts = np.zeros(len(sequence_text), dtype=np.float32)
        step = chunk_size - overlap
        for start in range(0, len(sequence_text), step):
            end = min(start + chunk_size, len(sequence_text))
            result[start:end] += self._encode_chunk(sequence_text[start:end])
            counts[start:end] += 1.0
            if end == len(sequence_text):
                break
        return result / counts[:, None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PepGeoSite tensors using FlexPepSite-compatible labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--esm-model", default="models/esm2_t6_8M_UR50D.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-mode", choices=["esm2", "onehot"], default="esm2")
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--radii", type=float, nargs="+", default=[6.0, 10.0, 14.0])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    if args.limit:
        frame = frame.head(args.limit).copy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = SequenceEmbedder(args.esm_model, args.device, args.embedding_mode)
    rows, failures = [], []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="preprocess"):
        complex_id = str(row.complex_id)
        target = output_dir / f"{complex_id}.pt"
        try:
            if target.exists() and not args.overwrite:
                rows.append({"complex_id": complex_id, "pdb_id": row.pdb_id, "cache_path": str(target)})
                continue
            receptor, _ = parse_pdb(row.receptor_path)
            peptide, peptide_atoms = parse_pdb(row.peptide_path)
            receptor_sequence, peptide_sequence = sequence(receptor), sequence(peptide)
            if not receptor_sequence or not peptide_sequence:
                raise ValueError("empty receptor or peptide after PDB parsing")
            coords = np.stack([residue.coord for residue in receptor]).astype(np.float32)
            labels = contact_labels(receptor, peptide_atoms, args.contact_cutoff)
            edge_indices, edge_distances = zip(
                *(radius_edges(coords, radius) for radius in args.radii)
            )
            payload = {
                "receptor_esm": torch.from_numpy(embedder.encode(receptor_sequence)).half(),
                "peptide_esm": torch.from_numpy(embedder.encode(peptide_sequence)).half(),
                "coords": torch.from_numpy(coords),
                "labels": torch.from_numpy(labels),
                "edge_indices": [torch.from_numpy(value) for value in edge_indices],
                "edge_distances": [torch.from_numpy(value) for value in edge_distances],
                "receptor_seq": receptor_sequence,
                "peptide_seq": peptide_sequence,
                "metadata": {
                    "contact_cutoff": args.contact_cutoff, "radii": args.radii,
                    "embedding_mode": args.embedding_mode, "esm_model": args.esm_model,
                    "esm_dim": embedder.dimension, "representative_atom": "CA_or_heavy_atom_centroid",
                },
            }
            torch.save(payload, target)
            rows.append({"complex_id": complex_id, "pdb_id": row.pdb_id, "cache_path": str(target)})
        except Exception as error:
            failures.append({"complex_id": complex_id, "error": repr(error)})

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_manifest, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_manifest.with_suffix(".failed.csv"), index=False)
    summary = {"input": len(frame), "cached": len(rows), "failed": len(failures), "esm_dim": embedder.dimension}
    print(json.dumps(summary, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
