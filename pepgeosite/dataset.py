from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedComplexDataset(Dataset):
    def __init__(self, manifest: str | Path, indices: list[int] | None = None):
        frame = pd.read_csv(manifest)
        self.frame = frame.iloc[indices].reset_index(drop=True) if indices is not None else frame

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        item = torch.load(row.cache_path, map_location="cpu", weights_only=False)
        item["complex_id"] = str(row.complex_id)
        item["pdb_id"] = str(row.get("pdb_id", str(row.complex_id).split("_")[0]))
        return item


def collate_complexes(items: list[dict]) -> dict:
    receptor_embeddings, labels, coordinates, batch_index = [], [], [], []
    peptide_embeddings, peptide_sequences = [], []
    edge_indices: list[list[torch.Tensor]] = [[] for _ in items[0]["edge_indices"]]
    edge_distances: list[list[torch.Tensor]] = [[] for _ in items[0]["edge_indices"]]
    node_offset = 0
    for batch_id, item in enumerate(items):
        length = int(item["receptor_esm"].shape[0])
        receptor_embeddings.append(item["receptor_esm"].float())
        labels.append(item["labels"].float())
        coordinates.append(item["coords"].float())
        batch_index.append(torch.full((length,), batch_id, dtype=torch.long))
        peptide_embeddings.append(item["peptide_esm"].float())
        peptide_sequences.append(item["peptide_seq"])
        for scale, (edges, distances) in enumerate(
            zip(item["edge_indices"], item["edge_distances"])
        ):
            edge_indices[scale].append(edges.long() + node_offset)
            edge_distances[scale].append(distances.float())
        node_offset += length

    peptide_lengths = torch.tensor([embedding.shape[0] for embedding in peptide_embeddings])
    max_peptide = int(peptide_lengths.max())
    peptide_dim = int(peptide_embeddings[0].shape[-1])
    padded_peptide = torch.zeros(len(items), max_peptide, peptide_dim)
    peptide_mask = torch.zeros(len(items), max_peptide, dtype=torch.bool)
    for index, embedding in enumerate(peptide_embeddings):
        length = embedding.shape[0]
        padded_peptide[index, :length] = embedding
        peptide_mask[index, :length] = True

    return {
        "receptor_esm": torch.cat(receptor_embeddings),
        "labels": torch.cat(labels),
        "coords": torch.cat(coordinates),
        "batch_index": torch.cat(batch_index),
        "peptide_esm": padded_peptide,
        "peptide_mask": peptide_mask,
        "peptide_sequences": peptide_sequences,
        "edge_indices": [torch.cat(values, dim=1) for values in edge_indices],
        "edge_distances": [torch.cat(values) for values in edge_distances],
        "complex_ids": [item["complex_id"] for item in items],
    }

