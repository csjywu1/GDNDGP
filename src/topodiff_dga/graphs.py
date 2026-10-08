from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from torch import nn


def _topk_rows(matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        indices = matrix.indices[start:end]
        data = matrix.data[start:end]
        keep = np.flatnonzero(indices != row)
        if keep.size > k:
            keep = keep[np.argpartition(data[keep], -k)[-k:]]
        rows.extend([row] * keep.size)
        cols.extend(indices[keep].tolist())
        vals.extend(data[keep].tolist())
    return sp.coo_matrix((vals, (rows, cols)), shape=matrix.shape, dtype=np.float32).tocsr()


def degree_corrected_gene_graph(
    x: sp.csr_matrix,
    neighbor_cap: int = 30,
    beta: float = 0.5,
    gamma: float = 0.5,
    delta: float = 0.5,
) -> sp.csr_matrix:
    """Equation (10): Dg^-delta X^T Dd^-gamma X Dg^-beta."""
    x = x.astype(np.float32).tocsr()
    drug_degree = np.maximum(np.asarray(x.sum(axis=1)).ravel(), 1.0)
    gene_degree = np.maximum(np.asarray(x.sum(axis=0)).ravel(), 1.0)
    middle = x.multiply(np.power(drug_degree, -gamma)[:, None])
    adjacency = (x.T @ middle).tocsr()
    adjacency = adjacency.multiply(np.power(gene_degree, -delta)[:, None])
    adjacency = adjacency.multiply(np.power(gene_degree, -beta)[None, :]).tocsr()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    adjacency = _topk_rows(adjacency, neighbor_cap)
    if abs(beta - delta) < 1e-12:
        adjacency = ((adjacency + adjacency.T) * 0.5).tocsr()
    adjacency.eliminate_zeros()
    return adjacency


def drug_neighbors(
    x: sp.csr_matrix, neighbor_cap: int = 30, block_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Construct top-k D-G-D neighbors without materializing the full drug graph."""
    x = x.astype(np.float32).tocsr()
    n_drugs = x.shape[0]
    neighbors = np.full((n_drugs, neighbor_cap), -1, dtype=np.int64)
    weights = np.zeros((n_drugs, neighbor_cap), dtype=np.float32)
    for begin in range(0, n_drugs, block_size):
        end = min(begin + block_size, n_drugs)
        block = (x[begin:end] @ x.T).tocsr()
        for local_row, drug in enumerate(range(begin, end)):
            start, stop = block.indptr[local_row], block.indptr[local_row + 1]
            idx = block.indices[start:stop]
            val = block.data[start:stop]
            keep = np.flatnonzero(idx != drug)
            if keep.size > neighbor_cap:
                keep = keep[np.argpartition(val[keep], -neighbor_cap)[-neighbor_cap:]]
            order = keep[np.argsort(val[keep])[::-1]] if keep.size else keep
            neighbors[drug, : order.size] = idx[order]
            weights[drug, : order.size] = val[order]
    return neighbors, weights


def _to_torch_sparse(matrix: sp.csr_matrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


@dataclass
class FilterParts:
    local: sp.csr_matrix
    eigenvectors: np.ndarray
    spectral_norm: float


def build_filter_parts(adjacency: sp.csr_matrix, rank: int = 128) -> FilterParts:
    n = adjacency.shape[0]
    if n == 0:
        raise ValueError("gene graph is empty")
    if adjacency.nnz == 0:
        return FilterParts(adjacency, np.zeros((n, 0), np.float32), 1.0)
    spectral_norm = float(abs(eigsh(adjacency, k=1, which="LM", return_eigenvectors=False)[0]))
    spectral_norm = max(spectral_norm, 1e-8)
    k = min(rank, max(n - 1, 0))
    if k == 0:
        vectors = np.zeros((n, 0), dtype=np.float32)
    elif n <= 256 and k >= n - 1:
        _, dense_vectors = np.linalg.eigh(adjacency.toarray())
        vectors = dense_vectors[:, -k:].astype(np.float32)
    else:
        _, vectors = eigsh(adjacency, k=k, which="LA")
        vectors = vectors.astype(np.float32)
    return FilterParts((adjacency / spectral_norm).tocsr(), vectors, spectral_norm)


class GraphFilter(nn.Module):
    """Combined local and low-rank graph filter in Eq. (12)."""

    def __init__(self, parts: FilterParts, omega: float = 0.1, alpha: float = 1.0):
        super().__init__()
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.register_buffer("local", _to_torch_sparse(parts.local))
        self.register_buffer("eigenvectors", torch.from_numpy(parts.eigenvectors))

    def propagate(self, profiles: torch.Tensor) -> torch.Tensor:
        local = torch.sparse.mm(self.local, profiles.T).T
        if self.eigenvectors.shape[1]:
            low_rank = (profiles @ self.eigenvectors) @ self.eigenvectors.T
        else:
            low_rank = torch.zeros_like(profiles)
        return (local + self.omega * low_rank) / (1.0 + self.omega)

    def forward(self, profiles: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(tau):
            tau = profiles.new_tensor(tau)
        while tau.ndim < profiles.ndim:
            tau = tau.unsqueeze(-1)
        return (1.0 - tau * self.alpha) * profiles + tau * self.alpha * self.propagate(profiles)

