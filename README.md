# PepGeoSite

PepGeoSite is a peptide-conditioned geometric neural network for residue-level
prediction of protein--peptide binding sites. It combines protein language-model
embeddings, asymmetric receptor-to-peptide cross-attention, and multi-scale
distance-aware graph attention over receptor residues.

## Repository layout

```text
pepgeosite/       Core model, data loading, losses, metrics, and PDB utilities
configs/          Reproducible training configurations
scripts/          Dataset auditing and experiment helpers
tests/            Lightweight forward/backward smoke test
preprocess.py     Convert receptor/peptide PDB pairs into cached tensors
train.py          Train and select a checkpoint on an internal validation split
predict.py        Run residue-level inference from a trained checkpoint
evaluate.py       Evaluate labelled benchmark manifests
```

Datasets, cached tensors, ESM checkpoints, trained weights, logs, and experiment
outputs are intentionally excluded from version control.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/zht-code/PepGeoSite.git
cd PepGeoSite
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download an ESM-2 checkpoint separately, or pass a Hugging Face model identifier
to `preprocess.py`. The default configuration expects the ESM-2 T6 checkpoint at
`models/esm2_t6_8M_UR50D.pt`.

## Input manifest

Preprocessing expects a CSV with these columns:

```csv
complex_id,pdb_id,receptor_path,peptide_path
1ABC_A_P,1ABC,/path/to/receptor.pdb,/path/to/peptide.pdb
```

PDB residues are ordered by chain, residue number, and insertion code. Hydrogen
atoms are ignored. For labelled preprocessing, a receptor residue is positive if
any retained receptor atom lies within the configured cutoff of a retained peptide
atom. Peptide coordinates are used to construct labels; the model receives receptor
geometry and receptor/peptide sequence embeddings.

## Preprocessing

```bash
python preprocess.py \
  --manifest data/raw/complexes.csv \
  --output-dir data/cache/esm2_t6 \
  --output-manifest data/manifests/train.csv \
  --esm-model models/esm2_t6_8M_UR50D.pt \
  --contact-cutoff 8.0 \
  --radii 6 10 14 \
  --device cuda
```

For a dependency-free pipeline check, use `--embedding-mode onehot` and set
`data.esm_dim: 21` in a copied configuration. One-hot mode is not intended for
production training.

## Training

```bash
python train.py \
  --config configs/pepgeosite.yaml \
  --manifest data/manifests/train.csv
```

The trainer groups records by PDB identifier before the 90/10 split. Model
selection uses validation AUPRC by default, with MCC and AUROC as tie-breakers.
The output directory contains `best.pt`, `best_model_weights.pt`, `last.pt`,
`history.jsonl`, and `best_metrics.json`.

## Inference

```bash
python predict.py \
  --checkpoint outputs/default/best.pt \
  --manifest data/manifests/test.csv \
  --output predictions.csv
```

The output contains a probability and binary prediction for every receptor residue.
By default, `predict.py` uses the validation-selected threshold stored in the
checkpoint. Use `--threshold` to override it explicitly.

## Benchmark evaluation

```bash
python evaluate.py \
  --checkpoint outputs/default/best.pt \
  --manifest data/manifests/TS251.csv \
  --benchmark-name TS251 \
  --output-dir results/TS251
```

The evaluator does not fit a threshold on benchmark data. It writes pooled metrics,
complex-level metrics, and residue-level predictions.

## Smoke test

```bash
python -c "import runpy; d=runpy.run_path('tests/test_smoke.py'); d['test_forward_and_backward']()"
```

## Reproducibility notes

The default configuration records graph radii, model dimensions, loss weights,
optimizer settings, split seed, and checkpoint-selection rules. Benchmark reports
should additionally identify the exact dataset manifests, structure sources,
language-model checkpoint, and trained PepGeoSite checkpoint used in the run.
