from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_site_loss(
    logits,
    targets,
    batch_index,
    positive_weight: float,
    negative_weight: float,
):
    """Class-weighted BCE, averaged per complex and then across the batch."""
    per_residue = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(targets > 0.5, positive_weight, negative_weight)
    weighted = per_residue * weights
    groups = int(batch_index.max().item()) + 1
    sums = weighted.new_zeros(groups)
    counts = weighted.new_zeros(groups)
    sums.index_add_(0, batch_index, weighted)
    counts.index_add_(0, batch_index, torch.ones_like(weighted))
    return (sums / counts.clamp_min(1.0)).mean()


def graph_smoothness(probabilities, edges, distances, sigma: float):
    src, dst = edges
    keep = src != dst
    if not keep.any():
        return probabilities.sum() * 0.0
    src, dst, distances = src[keep], dst[keep], distances[keep]
    weights = torch.exp(-distances.square() / (2.0 * float(sigma) ** 2))
    return (weights * (probabilities[src] - probabilities[dst]).square()).sum() / weights.sum().clamp_min(1e-12)


def in_batch_pair_loss(scores, peptide_sequences: list[str], temperature: float):
    batch_size = scores.shape[0]
    if batch_size < 2:
        return scores.sum() * 0.0
    normalised = ["".join(str(seq).upper().split()) for seq in peptide_sequences]
    valid = torch.ones_like(scores, dtype=torch.bool)
    for row in range(batch_size):
        for column in range(batch_size):
            if row != column and normalised[row] == normalised[column]:
                valid[row, column] = False
    usable = valid.sum(dim=1) > 1
    if not usable.any():
        return scores.sum() * 0.0
    masked = (scores / float(temperature)).masked_fill(~valid, -torch.inf)
    targets = torch.arange(batch_size, device=scores.device)
    return F.cross_entropy(masked[usable], targets[usable])


def total_loss(outputs, batch, config):
    cfg = config["loss"]
    logits, targets = outputs["logits"], batch["labels"]
    site = weighted_site_loss(
        logits,
        targets,
        batch["batch_index"],
        cfg["positive_weight"],
        cfg["negative_weight"],
    )
    cluster = graph_smoothness(
        torch.sigmoid(logits), batch["edge_indices"][-1],
        batch["edge_distances"][-1], cfg["cluster_sigma"],
    )
    pair = in_batch_pair_loss(
        outputs["pair_scores"], batch["peptide_sequences"], cfg["pair_temperature"]
    )
    total = site + cfg["lambda_cluster"] * cluster + cfg["lambda_pair"] * pair
    return total, {"site": site.detach(), "cluster": cluster.detach(), "pair": pair.detach()}
