"""Cluster analyzer — extract what the model has learned.

Given a trained model and its tokenizer, produces a readable map of:
  - Which tokens route to which fine cluster (top-N per cluster)
  - Which fine clusters route to which coarse group
  - Entropy and concentration metrics per cluster
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClusterProfile:
    """Profile of a single fine cluster."""
    cluster_id: int
    top_tokens: list[tuple[str, float]]  # (token_str, routing_weight)
    coarse_assignment: int  # dominant coarse group
    coarse_probs: list[float]  # soft assignment over coarse groups
    entropy: float  # routing entropy (how spread the assignments are)
    token_count: int  # how many tokens primarily route here


@dataclass
class HierarchySnapshot:
    """Complete snapshot of hierarchy state at a given step."""
    step: int
    fine_profiles: list[ClusterProfile]
    coarse_groups: dict[int, list[int]]  # coarse_id → [fine_cluster_ids]
    fine_centroids: torch.Tensor  # [K, D]
    coarse_centroids: torch.Tensor  # [M, D]
    fine_entropy: float  # H(fine routing) averaged
    coarse_entropy: float  # H(coarse routing) averaged
    score: float = 0.0  # LLM quality score (filled later)


class ClusterAnalyzer:
    """Analyzes learned clusters to produce human/LLM-readable profiles."""

    def __init__(self, model, tokenizer=None, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def analyze(self, step: int = 0, top_k: int = 20) -> HierarchySnapshot:
        """Produce a full hierarchy snapshot.

        Runs every token in the vocabulary through the routers to see
        where each token primarily routes.
        """
        model = self.model
        embedding = model.embedding

        # Get components
        fine_router = embedding.fine_router
        fine_centroids_mod = embedding.fine_centroids
        coarse_router = getattr(embedding, "coarse_router", None)
        coarse_centroids_mod = getattr(embedding, "coarse_centroids", None)

        vocab_size = embedding.local_embedding.embedding.num_embeddings
        K = fine_centroids_mod.n_clusters
        M = coarse_centroids_mod.n_clusters if coarse_centroids_mod else 0

        # Get all token embeddings
        all_ids = torch.arange(vocab_size, device=self.device)
        e_local = embedding.local_embedding(all_ids.unsqueeze(0))  # [1, V, D]
        e_local = e_local.squeeze(0)  # [V, D]

        # Fine routing for all tokens
        fine_weights = fine_router(e_local, temperature=0.3)  # [V, K] — sharp temp
        fine_assignments = fine_weights.argmax(dim=-1)  # [V]

        # Coarse routing
        if coarse_router is not None:
            e_cluster = fine_centroids_mod(fine_weights)  # [V, D]
            coarse_weights = coarse_router(e_cluster, temperature=0.2)  # [V, M]
        else:
            coarse_weights = None

        # Build per-cluster profiles
        profiles = []
        coarse_groups: dict[int, list[int]] = {g: [] for g in range(M)}

        for k in range(K):
            # Tokens that primarily route to this cluster
            mask = fine_assignments == k
            token_count = mask.sum().item()

            # Get routing weight for cluster k across all tokens
            cluster_affinity = fine_weights[:, k]  # [V]
            top_values, top_indices = torch.topk(
                cluster_affinity, min(top_k, vocab_size)
            )

            # Decode token strings
            top_tokens = []
            for idx, val in zip(top_indices.tolist(), top_values.tolist()):
                if self.tokenizer is not None:
                    try:
                        tok_str = self.tokenizer.decode([idx])
                    except Exception:
                        tok_str = f"<{idx}>"
                else:
                    tok_str = f"<{idx}>"
                top_tokens.append((tok_str, val))

            # Coarse assignment for this cluster
            coarse_id = 0
            coarse_probs = []
            if coarse_weights is not None and token_count > 0:
                # Average coarse routing for tokens in this fine cluster
                avg_coarse = coarse_weights[mask].mean(dim=0) if token_count > 0 else coarse_weights.mean(dim=0)
                coarse_id = avg_coarse.argmax().item()
                coarse_probs = avg_coarse.tolist()
                coarse_groups[coarse_id].append(k)

            # Entropy of fine routing for tokens in this cluster
            if token_count > 0:
                p = fine_weights[mask]  # [N, K]
                ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()
            else:
                ent = 0.0

            profiles.append(ClusterProfile(
                cluster_id=k,
                top_tokens=top_tokens,
                coarse_assignment=coarse_id,
                coarse_probs=coarse_probs,
                entropy=ent,
                token_count=token_count,
            ))

        # Global metrics
        fine_ent = -(fine_weights * (fine_weights + 1e-8).log()).sum(dim=-1).mean().item()
        coarse_ent = 0.0
        if coarse_weights is not None:
            coarse_ent = -(coarse_weights * (coarse_weights + 1e-8).log()).sum(dim=-1).mean().item()

        return HierarchySnapshot(
            step=step,
            fine_profiles=profiles,
            coarse_groups=coarse_groups,
            fine_centroids=fine_centroids_mod.centroids.detach().clone(),
            coarse_centroids=coarse_centroids_mod.centroids.detach().clone() if coarse_centroids_mod else torch.zeros(0),
            fine_entropy=fine_ent,
            coarse_entropy=coarse_ent,
        )

    def format_for_llm(self, snapshot: HierarchySnapshot, max_clusters: int = 64) -> str:
        """Format snapshot as a prompt for LLM evaluation.

        Returns a structured text description of clusters that an LLM
        can analyze and score.
        """
        lines = []
        lines.append(f"=== Cluster Hierarchy Analysis (step {snapshot.step}) ===")
        lines.append(f"Fine clusters: {len(snapshot.fine_profiles)}, Coarse groups: {len(snapshot.coarse_groups)}")
        lines.append(f"Fine entropy: {snapshot.fine_entropy:.2f}, Coarse entropy: {snapshot.coarse_entropy:.2f}")
        lines.append("")

        # Coarse groups overview
        lines.append("## Coarse Groups (which fine clusters belong to each)")
        for g_id in sorted(snapshot.coarse_groups.keys()):
            members = snapshot.coarse_groups[g_id]
            lines.append(f"  Group {g_id}: {len(members)} fine clusters → {members}")
        lines.append("")

        # Fine cluster details
        lines.append("## Fine Cluster Details (top tokens per cluster)")
        for p in snapshot.fine_profiles[:max_clusters]:
            tokens_str = ", ".join(
                f"'{t}' ({w:.3f})" for t, w in p.top_tokens[:10]
            )
            lines.append(
                f"  Cluster {p.cluster_id} → Coarse {p.coarse_assignment} "
                f"| {p.token_count} tokens | H={p.entropy:.2f} | [{tokens_str}]"
            )

        return "\n".join(lines)
