#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from pepgeosite.dataset import CachedComplexDataset, collate_complexes
from pepgeosite.losses import total_loss
from pepgeosite.metrics import compute_metrics
from pepgeosite.model import PepGeoSite


def metric_rank(metrics: dict, selection: dict) -> tuple[float, ...]:
    """Return a lexicographic rank: monitored metric, then configured tie breakers."""
    names = [selection["monitor"], *selection.get("tie_breakers", [])]
    direction = 1.0 if selection.get("mode", "max") == "max" else -1.0
    rank = []
    for name in names:
        value = float(metrics.get(name, float("nan")))
        rank.append(direction * value if np.isfinite(value) else -float("inf"))
    return tuple(rank)


def is_better(metrics: dict, best_rank: tuple[float, ...] | None, selection: dict) -> bool:
    candidate = metric_rank(metrics, selection)
    if best_rank is None:
        return True
    min_delta = float(selection.get("min_delta", 0.0))
    if candidate[0] > best_rank[0] + min_delta:
        return True
    if abs(candidate[0] - best_rank[0]) <= min_delta:
        return candidate[1:] > best_rank[1:]
    return False


def atomic_torch_save(payload, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    for key in ("receptor_esm", "labels", "coords", "batch_index", "peptide_esm", "peptide_mask"):
        batch[key] = batch[key].to(device, non_blocking=True)
    batch["edge_indices"] = [value.to(device, non_blocking=True) for value in batch["edge_indices"]]
    batch["edge_distances"] = [value.to(device, non_blocking=True) for value in batch["edge_distances"]]
    return batch


def build_loaders(manifest: str, config: dict, limit: int | None):
    frame = pd.read_csv(manifest)
    if limit:
        frame = frame.head(limit).copy()
    groups = frame.get("pdb_id", frame.complex_id.astype(str).str.split("_").str[0])
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config["training"]["val_fraction"], random_state=config["seed"]
    )
    train_index, valid_index = next(splitter.split(frame, groups=groups))
    train_data = CachedComplexDataset(manifest, frame.index[train_index].tolist())
    valid_data = CachedComplexDataset(manifest, frame.index[valid_index].tolist())
    common = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
        "collate_fn": collate_complexes, "pin_memory": True,
    }
    # Keep the final incomplete batch. Pair loss safely becomes zero for batch size one,
    # while the supervised site and geometry losses still provide a valid update.
    train_loader = DataLoader(train_data, shuffle=True, drop_last=False, **common)
    valid_loader = DataLoader(valid_data, shuffle=False, **common)
    return train_loader, valid_loader


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in tqdm(loader, desc="validation", leave=False):
        batch = move_batch(batch, device)
        output = model(batch)
        labels.extend(batch["labels"].cpu().numpy().tolist())
        probabilities.extend(torch.sigmoid(output["logits"]).cpu().numpy().tolist())
    return compute_metrics(labels, probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pepgeosite.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    seed_everything(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = build_loaders(args.manifest, config, args.limit)
    model = PepGeoSite(
        esm_dim=config["data"]["esm_dim"], scales=len(config["data"]["radii"]),
        **config["model"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    selection = config["training"].setdefault(
        "selection",
        {"monitor": "auprc", "mode": "max", "min_delta": 1e-5,
         "tie_breakers": ["mcc", "auroc"], "restore_best_at_end": True},
    )
    scheduler_cfg = config["training"].get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=selection.get("mode", "max"),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 5)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["amp"] and device.type == "cuda")
    start_epoch, best_rank, best_metrics = 1, None, None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metrics = checkpoint.get("best_metrics")
        if best_metrics:
            best_rank = metric_rank(best_metrics, selection)

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    patience = 0
    accumulation = max(1, int(config["training"]["grad_accumulation"]))
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {"total": [], "site": [], "cluster": [], "pair": []}
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}"), 1):
            batch = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(batch)
                loss, components = total_loss(outputs, batch, config)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running["total"].append(float(loss.detach().cpu()))
            for name, value in components.items():
                running[name].append(float(value.cpu()))
        if len(train_loader) % accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics = validate(model, valid_loader, device)
        monitor_value = float(metrics[selection["monitor"]])
        scheduler.step(monitor_value)
        record = {
            "epoch": epoch, "train": {key: float(np.mean(value)) for key, value in running.items()},
            "validation": metrics, "lr": float(optimizer.param_groups[0]["lr"]),
        }
        print(json.dumps(record, ensure_ascii=False))
        with open(output_dir / "history.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        improved = is_better(metrics, best_rank, selection)
        if improved:
            best_rank = metric_rank(metrics, selection)
            best_metrics = dict(metrics)
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "config": config, "metrics": metrics, "best_metrics": best_metrics,
            "selection": selection,
        }
        atomic_torch_save(state, output_dir / "last.pt")
        if improved:
            patience = 0
            atomic_torch_save(state, output_dir / "best.pt")
            atomic_torch_save(model.state_dict(), output_dir / "best_model_weights.pt")
            with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as handle:
                json.dump({"epoch": epoch, "selection": selection, "metrics": metrics}, handle, indent=2)
            print(f"saved new best model at epoch {epoch}: {selection['monitor']}={monitor_value:.6f}")
        else:
            patience += 1
            if patience >= config["training"]["early_stop_patience"]:
                print(
                    f"early stopping after {patience} epochs without validation "
                    f"{selection['monitor']} improvement"
                )
                break

    best_path = output_dir / "best.pt"
    if selection.get("restore_best_at_end", True) and best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        atomic_torch_save(model.state_dict(), output_dir / "best_model_weights.pt")
        print(f"restored best weights from epoch {best_checkpoint['epoch']}: {best_checkpoint['metrics']}")


if __name__ == "__main__":
    main()
