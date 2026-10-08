from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from topodiff_dga.data import dense_profiles, load_association_matrix
from topodiff_dga.graphs import build_filter_parts, degree_corrected_gene_graph, drug_neighbors
from topodiff_dga.model import TopoDiffDGA


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank genes with a trained TopoDiff-DGA model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-matrix", required=True)
    parser.add_argument("--drug-ids", type=int, nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default="outputs/rankings.json")
    parser.add_argument(
        "--layout", choices=("genes-by-drugs", "drugs-by-genes"), default="genes-by-drugs"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    settings = checkpoint["arguments"]
    train = load_association_matrix(args.train_matrix, args.layout)
    gene_graph = degree_corrected_gene_graph(
        train,
        neighbor_cap=settings["neighbor_cap"],
        beta=settings["beta"],
        gamma=settings["gamma"],
        delta=settings["delta"],
    )
    filter_parts = build_filter_parts(gene_graph, rank=settings["spectral_rank"])
    neighbors, weights = drug_neighbors(train, neighbor_cap=settings["neighbor_cap"])
    model = TopoDiffDGA(
        n_drugs=train.shape[0],
        n_genes=train.shape[1],
        hidden_dim=settings["hidden_dim"],
        filter_parts=filter_parts,
        drug_neighbors=neighbors,
        drug_neighbor_weights=weights,
        steps=settings["steps"],
        omega=settings["omega"],
        alpha=settings["alpha"],
        sigma_max=settings["sigma_max"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    drug_ids = np.asarray(args.drug_ids, dtype=np.int64)
    if drug_ids.min() < 0 or drug_ids.max() >= train.shape[0]:
        raise ValueError(f"drug ids must be between 0 and {train.shape[0] - 1}")
    profiles = torch.from_numpy(dense_profiles(train, drug_ids)).to(device)
    with torch.no_grad():
        scores = model.reconstruct(profiles, torch.from_numpy(drug_ids).long().to(device))
    scores = scores.cpu().numpy()

    result: dict[str, list[dict[str, float | int]]] = {}
    for row, drug in enumerate(drug_ids):
        known = train[int(drug)].indices
        scores[row, known] = -np.inf
        count = min(args.top_k, train.shape[1] - known.size)
        ranked = np.argpartition(scores[row], -count)[-count:]
        ranked = ranked[np.argsort(scores[row, ranked])[::-1]]
        result[str(int(drug))] = [
            {"gene_id": int(gene), "score": float(scores[row, gene])} for gene in ranked
        ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved rankings to {output}")


if __name__ == "__main__":
    main()

