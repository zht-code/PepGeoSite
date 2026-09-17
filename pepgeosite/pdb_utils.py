from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    name3: str
    name1: str
    atoms: list[tuple[str, np.ndarray]]
    coord: np.ndarray


def parse_pdb(path: str | Path) -> tuple[list[Residue], np.ndarray]:
    """Parse residues exactly as FlexPepSite: standard amino acids and heavy atoms."""
    residues: dict[tuple[str, int, str, str], list[tuple[str, np.ndarray]]] = {}
    atoms: list[np.ndarray] = []
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_1:
                continue
            atom = line[12:16].strip()
            if atom.startswith("H"):
                continue
            chain = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
                coord = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            key = (chain, resseq, line[26].strip(), resname)
            residues.setdefault(key, []).append((atom, coord))
            atoms.append(coord)

    parsed: list[Residue] = []
    for (chain, resseq, icode, resname), residue_atoms in residues.items():
        ca = [xyz for atom, xyz in residue_atoms if atom == "CA"]
        representative = ca[0] if ca else np.mean(
            [xyz for _, xyz in residue_atoms], axis=0, dtype=np.float32
        )
        parsed.append(
            Residue(
                chain, resseq, icode, resname, AA3_TO_1[resname],
                residue_atoms, np.asarray(representative, dtype=np.float32),
            )
        )
    parsed.sort(key=lambda residue: (residue.chain, residue.resseq, residue.icode))
    atom_array = np.stack(atoms) if atoms else np.zeros((0, 3), dtype=np.float32)
    return parsed, atom_array


def sequence(residues: list[Residue]) -> str:
    return "".join(residue.name1 for residue in residues)


def contact_labels(
    receptor_residues: list[Residue], peptide_atoms: np.ndarray, cutoff: float = 8.0
) -> np.ndarray:
    """Label a receptor residue positive if any heavy-atom pair is below cutoff."""
    labels = np.zeros(len(receptor_residues), dtype=np.float32)
    if peptide_atoms.size == 0:
        return labels
    cutoff2 = float(cutoff) ** 2
    for index, residue in enumerate(receptor_residues):
        receptor_atoms = np.stack([coord for _, coord in residue.atoms])
        squared = ((receptor_atoms[:, None, :] - peptide_atoms[None, :, :]) ** 2).sum(-1)
        labels[index] = float((squared < cutoff2).any())
    return labels


def radius_edges(coords: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a directed radius graph with deterministic self-loops."""
    squared = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    src, dst = np.where(squared <= float(radius) ** 2)
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    distances = np.sqrt(squared[src, dst]).astype(np.float32)
    return np.stack([src, dst]).astype(np.int64), distances

