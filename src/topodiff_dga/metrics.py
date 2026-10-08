from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score


def sampled_pairs(
    positives: sp.csr_matrix,
    training: sp.csr_matrix,
    negatives_per_positive: int = 2,
    seed: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    eligible = np.asarray(training.getnnz(axis=1)).ravel() > 0
    positives = positives.multiply(eligible[:, None]).tocsr()
    positives.eliminate_zeros()
    rows, cols = positives.nonzero()
    pair_drugs = rows.tolist()
    pair_genes = cols.tolist()
    labels = [1] * len(rows)
    occupied = (training + positives).astype(bool).tocsr()
    n_genes = training.shape[1]
    for drug in np.unique(rows):
        count = int((rows == drug).sum()) * negatives_per_positive
        forbidden = set(occupied[drug].indices.tolist())
        available = np.setdiff1d(np.arange(n_genes), np.fromiter(forbidden, dtype=np.int64))
        if available.size == 0:
            continue
        sampled = rng.choice(available, size=min(count, available.size), replace=False)
        pair_drugs.extend([int(drug)] * sampled.size)
        pair_genes.extend(sampled.tolist())
        labels.extend([0] * sampled.size)
    return np.asarray(pair_drugs), np.asarray(pair_genes), np.asarray(labels)


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.unique(scores)
    if thresholds.size > 1000:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 1001))
    values = [f1_score(labels, scores >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(values))])


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = scores >= threshold
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }
