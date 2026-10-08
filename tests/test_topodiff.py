import numpy as np
import scipy.sparse as sp
import torch

from topodiff_dga.graphs import (
    build_filter_parts,
    degree_corrected_gene_graph,
    drug_neighbors,
)
from topodiff_dga.model import TopoDiffDGA


def toy_matrix():
    return sp.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
    )


def test_graphs_and_model_shapes():
    x = toy_matrix()
    adjacency = degree_corrected_gene_graph(x, neighbor_cap=3)
    assert adjacency.shape == (5, 5)
    assert np.allclose(adjacency.toarray(), adjacency.toarray().T)
    neighbors, weights = drug_neighbors(x, neighbor_cap=2, block_size=2)
    assert neighbors.shape == weights.shape == (4, 2)

    model = TopoDiffDGA(
        n_drugs=4,
        n_genes=5,
        hidden_dim=8,
        filter_parts=build_filter_parts(adjacency, rank=2),
        drug_neighbors=neighbors,
        drug_neighbor_weights=weights,
        steps=4,
    )
    profiles = torch.from_numpy(x.toarray()[:2])
    drug_ids = torch.tensor([0, 1])
    loss, parts = model.training_loss(profiles, drug_ids)
    assert loss.ndim == 0
    assert set(parts) == {"diffusion", "association"}
    reconstructed = model.reconstruct(profiles, drug_ids)
    assert reconstructed.shape == profiles.shape
    assert torch.isfinite(reconstructed).all()

