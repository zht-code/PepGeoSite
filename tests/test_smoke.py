import torch

from pepgeosite.dataset import collate_complexes
from pepgeosite.losses import total_loss
from pepgeosite.model import PepGeoSite


def item(name, receptor_length, peptide_length, esm_dim=21):
    coords = torch.randn(receptor_length, 3)
    distance = torch.cdist(coords, coords)
    edge_indices, edge_distances = [], []
    for radius in (1.5, 2.5, 4.0):
        src, dst = torch.where(distance <= radius)
        edge_indices.append(torch.stack([src, dst]))
        edge_distances.append(distance[src, dst])
    return {
        "complex_id": name, "pdb_id": name.split("_")[0],
        "receptor_esm": torch.randn(receptor_length, esm_dim),
        "peptide_esm": torch.randn(peptide_length, esm_dim),
        "coords": coords, "labels": (torch.rand(receptor_length) > 0.8).float(),
        "edge_indices": edge_indices, "edge_distances": edge_distances,
        "peptide_seq": "ACDEFG"[:peptide_length],
    }


def test_forward_and_backward():
    batch = collate_complexes([item("1AAA_1", 17, 5), item("2BBB_1", 13, 6)])
    model = PepGeoSite(esm_dim=21, hidden_dim=32, attention_heads=4, graph_layers=1, scales=3)
    output = model(batch)
    assert output["logits"].shape == (30,)
    assert output["pair_scores"].shape == (2, 2)
    config = {"loss": {
        "positive_weight": 5.0, "negative_weight": 0.5,
        "lambda_cluster": 0.1, "cluster_sigma": 2.0,
        "lambda_pair": 0.1, "pair_temperature": 0.1,
    }}
    loss, components = total_loss(output, batch, config)
    assert torch.isfinite(loss)
    assert set(components) == {"site", "cluster", "pair"}
    loss.backward()

