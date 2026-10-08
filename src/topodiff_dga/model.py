from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .graphs import FilterParts, GraphFilter


def sinusoidal_embedding(steps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=steps.device) * -scale)
    angles = steps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class DrugContextEncoder(nn.Module):
    def __init__(
        self,
        n_drugs: int,
        hidden_dim: int,
        neighbors: np.ndarray,
        weights: np.ndarray,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_drugs, hidden_dim)
        self.transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("neighbors", torch.from_numpy(neighbors))
        self.register_buffer("edge_weights", torch.from_numpy(weights))
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        center = self.embedding(drug_ids)
        neighbor_ids = self.neighbors[drug_ids]
        mask = neighbor_ids.ge(0)
        safe_ids = neighbor_ids.clamp_min(0)
        neighbor_emb = self.embedding(safe_ids)
        query = self.query(center).unsqueeze(1)
        keys = self.key(neighbor_emb)
        logits = (query * keys).sum(dim=-1) / math.sqrt(center.shape[-1])
        logits = logits + self.edge_weights[drug_ids].clamp_min(1.0).log()
        logits = logits.masked_fill(~mask, -1e9)
        attention = torch.softmax(logits, dim=1) * mask
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        aggregate = (attention.unsqueeze(-1) * self.transform(neighbor_emb)).sum(dim=1)
        return center + self.dropout(F.gelu(aggregate))


class ConditionalDenoiser(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int) -> None:
        super().__init__()
        self.gene_embeddings = nn.Parameter(torch.empty(n_genes, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        nn.init.xavier_uniform_(self.gene_embeddings)

    def forward(
        self,
        diffused: torch.Tensor,
        masked: torch.Tensor,
        drug_context: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        scale = math.sqrt(max(diffused.shape[1], 1))
        z_hidden = diffused @ self.gene_embeddings / scale
        x_hidden = masked @ self.gene_embeddings / scale
        joined = torch.cat((z_hidden, x_hidden, drug_context, time_embedding), dim=-1)
        hidden = self.output_norm(self.mlp(joined) + self.input_norm(z_hidden + x_hidden))
        return hidden @ self.gene_embeddings.T / math.sqrt(hidden.shape[-1]) + self.gene_bias


class TopoDiffDGA(nn.Module):
    def __init__(
        self,
        n_drugs: int,
        n_genes: int,
        hidden_dim: int,
        filter_parts: FilterParts,
        drug_neighbors: np.ndarray,
        drug_neighbor_weights: np.ndarray,
        steps: int = 100,
        omega: float = 0.1,
        alpha: float = 1.0,
        sigma_max: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.steps = int(steps)
        self.sigma_max = float(sigma_max)
        self.graph_filter = GraphFilter(filter_parts, omega=omega, alpha=alpha)
        self.context_encoder = DrugContextEncoder(
            n_drugs, hidden_dim, drug_neighbors, drug_neighbor_weights, dropout
        )
        self.denoiser = ConditionalDenoiser(n_genes, hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def sigma(self, steps: torch.Tensor) -> torch.Tensor:
        return self.sigma_max * torch.sqrt(steps.float() / self.steps)

    def mask_profile(self, profiles: torch.Tensor, probability: float) -> torch.Tensor:
        keep = torch.rand_like(profiles).ge(probability)
        return profiles * keep

    def predict_clean(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        drug_ids: torch.Tensor,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_encoder(drug_ids)
        time = self.time_projection(sinusoidal_embedding(steps, context.shape[-1]))
        return self.denoiser(state, condition, context, time)

    def training_loss(
        self,
        profiles: torch.Tensor,
        drug_ids: torch.Tensor,
        mask_probability: float = 0.2,
        positive_weight: float = 2.0,
        association_weight: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = profiles.shape[0]
        steps = torch.randint(1, self.steps + 1, (batch,), device=profiles.device)
        tau = steps.float() / self.steps
        noise = torch.randn_like(profiles)
        state = self.graph_filter(profiles, tau) + self.sigma(steps).unsqueeze(1) * noise
        condition = self.mask_profile(profiles, mask_probability)
        prediction = self.predict_clean(state, condition, drug_ids, steps)
        squared = (prediction - profiles).square()
        diffusion_loss = squared.mean()
        association_loss = (squared * (1.0 + positive_weight * profiles)).mean()
        total = diffusion_loss + association_weight * association_loss
        return total, {"diffusion": diffusion_loss.detach(), "association": association_loss.detach()}

    @torch.no_grad()
    def reconstruct(self, profiles: torch.Tensor, drug_ids: torch.Tensor) -> torch.Tensor:
        condition = profiles
        state = self.graph_filter(profiles, 1.0)
        for step in range(self.steps, 0, -1):
            steps = torch.full((profiles.shape[0],), step, device=profiles.device, dtype=torch.long)
            clean = self.predict_clean(state, condition, drug_ids, steps)
            sigma_t = self.sigma(steps).unsqueeze(1)
            if step == 1:
                sigma_previous = torch.zeros_like(sigma_t)
            else:
                previous = torch.full_like(steps, step - 1)
                sigma_previous = self.sigma(previous).unsqueeze(1)
            forward_t = self.graph_filter(clean, step / self.steps)
            forward_previous = self.graph_filter(clean, (step - 1) / self.steps)
            state = forward_previous + (sigma_previous / sigma_t) * (state - forward_t)
        return state

