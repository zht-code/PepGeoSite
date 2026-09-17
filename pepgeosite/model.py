from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_softmax(values: torch.Tensor, index: torch.Tensor, groups: int) -> torch.Tensor:
    # AMP may evaluate exp in FP32 even when ``values`` is FP16/BF16.  Keep the
    # complete reduction in FP32 so index_add_ receives matching scalar types,
    # then return weights in the input dtype for the downstream value product.
    original_dtype = values.dtype
    work_values = (
        values.float()
        if original_dtype in (torch.float16, torch.bfloat16)
        else values
    )
    maxima = work_values.new_full((groups,), -torch.inf)
    maxima.scatter_reduce_(0, index, work_values, reduce="amax", include_self=True)
    exp_values = torch.exp(work_values - maxima[index])
    denominator = work_values.new_zeros(groups)
    denominator.index_add_(0, index, exp_values)
    weights = exp_values / denominator[index].clamp_min(1e-12)
    return weights.to(original_dtype)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def padded(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(states).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bp,bpd->bd", weights, states)

    def segmented(self, states: torch.Tensor, batch_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        logits = self.score(states).squeeze(-1)
        weights = segment_softmax(logits, batch_index, batch_size)
        pooled = states.new_zeros(batch_size, states.shape[-1])
        pooled.index_add_(0, batch_index, weights[:, None] * states)
        return pooled


class DistanceGraphAttention(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gamma_raw = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, edges: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        src, dst = edges
        query, key, value = self.query(states), self.key(states), self.value(states)
        gamma = F.softplus(self.gamma_raw)
        logits = (query[src] * key[dst]).sum(-1) / math.sqrt(states.shape[-1])
        logits = logits - gamma * distances.square()
        weights = segment_softmax(logits, src, states.shape[0])
        output = states.new_zeros(states.shape)
        output.index_add_(0, src, self.dropout(weights)[:, None] * value[dst])
        return output


class MultiScaleLayer(nn.Module):
    def __init__(self, dim: int, scales: int, dropout: float):
        super().__init__()
        self.attention = nn.ModuleList(
            [DistanceGraphAttention(dim, dropout) for _ in range(scales)]
        )
        self.gates = nn.ModuleList([nn.Linear(dim * 2, 1) for _ in range(scales)])
        self.update = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, states, edge_indices, edge_distances):
        scale_states = [
            layer(states, edges, distances)
            for layer, edges, distances in zip(self.attention, edge_indices, edge_distances)
        ]
        gate_logits = torch.cat(
            [gate(torch.cat([states, value], dim=-1)) for gate, value in zip(self.gates, scale_states)],
            dim=-1,
        )
        gates = torch.softmax(gate_logits, dim=-1)
        geometry = sum(gates[:, scale : scale + 1] * value for scale, value in enumerate(scale_states))
        return self.norm(states + self.update(torch.cat([states, geometry], dim=-1)))


class CrossInteractionLayer(nn.Module):
    """Residual peptide-to-receptor interaction block for deeper conditioning."""

    def __init__(self, dim: int, attention_heads: int, dropout: float):
        super().__init__()
        self.receptor_norm = nn.LayerNorm(dim)
        self.peptide_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )

    def forward(
        self,
        receptor: torch.Tensor,
        peptide: torch.Tensor,
        peptide_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.receptor_norm(receptor)
        key_value = self.peptide_norm(peptide)
        update, attention = self.attention(
            query, key_value, key_value,
            key_padding_mask=~peptide_mask, need_weights=True,
        )
        receptor = receptor + self.attention_dropout(update)
        receptor = receptor + self.ffn(self.ffn_norm(receptor))
        return receptor, attention


class PepGeoSite(nn.Module):
    def __init__(
        self, esm_dim: int = 320, hidden_dim: int = 192, attention_heads: int = 4,
        graph_layers: int = 2, interaction_layers: int = 1,
        scales: int = 3, dropout: float = 0.15,
    ):
        super().__init__()
        self.receptor_projection = nn.Sequential(
            nn.LayerNorm(esm_dim), nn.Linear(esm_dim, hidden_dim), nn.Dropout(dropout)
        )
        self.peptide_projection = nn.Sequential(
            nn.LayerNorm(esm_dim), nn.Linear(esm_dim, hidden_dim), nn.Dropout(dropout)
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.interaction_layer_count = max(1, int(interaction_layers))
        if self.interaction_layer_count > 1:
            self.initial_cross_norm = nn.LayerNorm(hidden_dim)
        self.interaction_layers = nn.ModuleList(
            [
                CrossInteractionLayer(hidden_dim, attention_heads, dropout)
                for _ in range(self.interaction_layer_count - 1)
            ]
        )
        self.peptide_pool = AttentionPool(hidden_dim)
        self.pre_geometry_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.pre_geometry_norm = nn.LayerNorm(hidden_dim)
        self.graph_layers = nn.ModuleList(
            [MultiScaleLayer(hidden_dim, scales, dropout) for _ in range(graph_layers)]
        )
        self.receptor_pool = AttentionPool(hidden_dim)
        self.final_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.site_head = nn.Linear(hidden_dim, 1)
        self.intrinsic_receptor_pool = AttentionPool(hidden_dim)
        self.pair_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pad_receptor(states: torch.Tensor, batch_index: torch.Tensor, batch_size: int):
        lengths = torch.bincount(batch_index, minlength=batch_size)
        maximum = int(lengths.max())
        padded = states.new_zeros(batch_size, maximum, states.shape[-1])
        mask = torch.zeros(batch_size, maximum, dtype=torch.bool, device=states.device)
        offsets = torch.zeros(batch_size, dtype=torch.long, device=states.device)
        positions = torch.empty_like(batch_index)
        for node, group in enumerate(batch_index.tolist()):
            positions[node] = offsets[group]
            offsets[group] += 1
        padded[batch_index, positions] = states
        mask[batch_index, positions] = True
        return padded, mask, positions

    def forward(self, batch: dict) -> dict:
        receptor_esm = batch["receptor_esm"]
        peptide_esm = batch["peptide_esm"]
        batch_index = batch["batch_index"]
        batch_size = peptide_esm.shape[0]
        receptor = self.receptor_projection(receptor_esm)
        peptide = self.peptide_projection(peptide_esm)
        peptide_mask = batch["peptide_mask"]
        padded_receptor, receptor_mask, positions = self._pad_receptor(
            receptor, batch_index, batch_size
        )
        cross, attention = self.cross_attention(
            padded_receptor, peptide, peptide,
            key_padding_mask=~peptide_mask, need_weights=True,
        )
        if self.interaction_layer_count > 1:
            cross = self.initial_cross_norm(padded_receptor + cross)
            for layer in self.interaction_layers:
                cross, attention = layer(cross, peptide, peptide_mask)
        cross_flat = cross[batch_index, positions]
        peptide_global = self.peptide_pool.padded(peptide, peptide_mask)
        fused = self.pre_geometry_fusion(
            torch.cat([receptor, cross_flat, peptide_global[batch_index]], dim=-1)
        )
        states = self.pre_geometry_norm(receptor + fused)
        for layer in self.graph_layers:
            states = layer(states, batch["edge_indices"], batch["edge_distances"])
        receptor_context = self.receptor_pool.segmented(states, batch_index, batch_size)
        final = self.final_norm(
            states + self.final_fusion(
                torch.cat([states, receptor_context[batch_index], peptide_global[batch_index]], dim=-1)
            )
        )
        logits = self.site_head(final).squeeze(-1)

        intrinsic = self.intrinsic_receptor_pool.segmented(receptor, batch_index, batch_size)
        rec = intrinsic[:, None, :].expand(-1, batch_size, -1)
        pep = peptide_global[None, :, :].expand(batch_size, -1, -1)
        pair_features = torch.cat([rec, pep, rec * pep, (rec - pep).abs()], dim=-1)
        pair_scores = self.pair_head(pair_features).squeeze(-1)
        return {"logits": logits, "pair_scores": pair_scores, "attention": attention}
