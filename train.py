from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from topodiff_dga.data import dense_profiles, load_association_matrix, validation_split
from topodiff_dga.graphs import (
    build_filter_parts,
    degree_corrected_gene_graph,
    drug_neighbors,
)
from topodiff_dga.metrics import (
    best_f1_threshold,
    classification_metrics,
    sampled_pairs,
)
from topodiff_dga.model import TopoDiffDGA


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TopoDiff-DGA")
    parser.add_argument("--train-matrix", required=True)
    parser.add_argument("--test-matrix", required=True)
    parser.add_argument("--validation-matrix")
    parser.add_argument(
        "--layout", choices=("genes-by-drugs", "drugs-by-genes"), default="genes-by-drugs"
    )
    parser.add_argument("--output", default="outputs/topodiff_dga")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--neighbor-cap", type=int, default=30)
    parser.add_argument("--spectral-rank", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=0.1)
    parser.add_argument("--mask-probability", type=float, default=0.2)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--association-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def score_pairs(
    model: TopoDiffDGA,
    train,
    pair_drugs: np.ndarray,
    pair_genes: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores = np.empty(pair_drugs.size, dtype=np.float32)
    unique_drugs = np.unique(pair_drugs)
    order = np.argsort(pair_drugs, kind="stable")
    sorted_drugs = pair_drugs[order]
    boundaries = np.searchsorted(sorted_drugs, unique_drugs)
    ends = np.r_[boundaries[1:], order.size]
    locations_by_drug = {
        int(drug): order[begin:end]
        for drug, begin, end in zip(unique_drugs, boundaries, ends)
    }
    for begin in range(0, unique_drugs.size, batch_size):
        batch_drugs = unique_drugs[begin : begin + batch_size]
        profiles = torch.from_numpy(dense_profiles(train, batch_drugs)).to(device)
        drug_ids = torch.from_numpy(batch_drugs).long().to(device)
        reconstructed = model.reconstruct(profiles, drug_ids).cpu().numpy()
        for local, drug in enumerate(batch_drugs):
            locations = locations_by_drug[int(drug)]
            scores[locations] = reconstructed[local, pair_genes[locations]]
    return scores


def evaluate(
    model: TopoDiffDGA,
    train,
    positives,
    device: torch.device,
    batch_size: int,
    seed: int,
    threshold: float | None = None,
) -> tuple[dict[str, float], float]:
    drugs, genes, labels = sampled_pairs(positives, train, seed=seed)
    scores = score_pairs(model, train, drugs, genes, device, batch_size)
    if threshold is None:
        threshold = best_f1_threshold(labels, scores)
    return classification_metrics(labels, scores, threshold), threshold


def main() -> None:
    args = arguments()
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train = load_association_matrix(args.train_matrix, args.layout)
    test = load_association_matrix(args.test_matrix, args.layout)
    if train.shape != test.shape:
        raise ValueError(f"train and test shapes differ: {train.shape} versus {test.shape}")
    if args.validation_matrix:
        validation = load_association_matrix(args.validation_matrix, args.layout)
    else:
        train, validation = validation_split(train, seed=args.seed)

    gene_graph = degree_corrected_gene_graph(
        train,
        neighbor_cap=args.neighbor_cap,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
    )
    filter_parts = build_filter_parts(gene_graph, rank=args.spectral_rank)
    neighbors, neighbor_weights = drug_neighbors(train, neighbor_cap=args.neighbor_cap)
    model = TopoDiffDGA(
        n_drugs=train.shape[0],
        n_genes=train.shape[1],
        hidden_dim=args.hidden_dim,
        filter_parts=filter_parts,
        drug_neighbors=neighbors,
        drug_neighbor_weights=neighbor_weights,
        steps=args.steps,
        omega=args.omega,
        alpha=args.alpha,
        sigma_max=args.sigma_max,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    train_drugs = np.flatnonzero(np.asarray(train.sum(axis=1)).ravel() > 0)
    best_auc = -np.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint = output / "best_model.pt"

    for epoch in trange(1, args.epochs + 1, desc="training"):
        model.train()
        shuffled = np.random.permutation(train_drugs)
        losses: list[float] = []
        for begin in range(0, shuffled.size, args.batch_size):
            drug_batch = shuffled[begin : begin + args.batch_size]
            profiles = torch.from_numpy(dense_profiles(train, drug_batch)).to(device)
            drug_ids = torch.from_numpy(drug_batch).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.training_loss(
                profiles,
                drug_ids,
                mask_probability=args.mask_probability,
                positive_weight=args.positive_weight,
                association_weight=args.association_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        validation_metrics, threshold = evaluate(
            model,
            train,
            validation,
            device,
            args.evaluation_batch_size,
            args.seed,
        )
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **validation_metrics}
        history.append(row)
        print(json.dumps(row))
        if validation_metrics["auc"] > best_auc:
            best_auc = validation_metrics["auc"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arguments": vars(args),
                    "shape": train.shape,
                    "threshold": threshold,
                    "epoch": epoch,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    validation_metrics, threshold = evaluate(
        model, train, validation, device, args.evaluation_batch_size, args.seed
    )
    test_metrics, _ = evaluate(
        model, train, test, device, args.evaluation_batch_size, args.seed, threshold
    )
    result = {
        "best_epoch": saved["epoch"],
        "validation": validation_metrics,
        "test": test_metrics,
        "threshold": threshold,
    }
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
