from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def load_association_matrix(path: str | Path, layout: str = "genes-by-drugs") -> sp.csr_matrix:
    """Load a binary association matrix and return it in drugs-by-genes layout."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        matrix = sp.load_npz(path)
    elif suffix == ".npy":
        matrix = np.load(path)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        matrix = np.loadtxt(path, delimiter=delimiter)
    else:
        with path.open("rb") as handle:
            matrix = pickle.load(handle)

    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    if layout == "genes-by-drugs":
        matrix = matrix.T.tocsr()
    elif layout != "drugs-by-genes":
        raise ValueError("layout must be 'genes-by-drugs' or 'drugs-by-genes'")
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def validation_split(
    train: sp.csr_matrix, fraction: float = 0.1, seed: int = 31
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Move a fraction of associations to validation while retaining a drug anchor."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    train = train.tolil(copy=True)
    row_degree = np.asarray(train.getnnz(axis=1), dtype=np.int64)
    coo = train.tocoo()
    order = rng.permutation(coo.nnz)
    target = max(1, int(round(coo.nnz * fraction)))
    val_rows: list[int] = []
    val_cols: list[int] = []
    for location in order:
        drug = int(coo.row[location])
        gene = int(coo.col[location])
        if row_degree[drug] <= 1:
            continue
        val_rows.append(drug)
        val_cols.append(gene)
        row_degree[drug] -= 1
        if len(val_rows) >= target:
            break
    held_by_row: dict[int, set[int]] = {}
    for drug, gene in zip(val_rows, val_cols):
        held_by_row.setdefault(drug, set()).add(gene)
    for drug, held in held_by_row.items():
        old_rows = train.rows[drug]
        old_data = train.data[drug]
        train.rows[drug] = [gene for gene in old_rows if gene not in held]
        train.data[drug] = [value for gene, value in zip(old_rows, old_data) if gene not in held]
    val = sp.coo_matrix(
        (np.ones(len(val_rows), dtype=np.float32), (val_rows, val_cols)),
        shape=train.shape,
    ).tocsr()
    return train.tocsr(), val


def dense_profiles(matrix: sp.csr_matrix, rows: np.ndarray) -> np.ndarray:
    return matrix[rows].toarray().astype(np.float32, copy=False)
