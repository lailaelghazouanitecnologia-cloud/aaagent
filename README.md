# HCLM-D — Hierarchical Clustered Language Model with Masked Diffusion

A **20M-parameter research prototype** combining hierarchical clustered embeddings with masked diffusion language modeling ([LLaDA](https://arxiv.org/abs/2502.09992)-style), trained from scratch.

---

## 1. What This Project Is

HCLM-D is an experimental language model that merges two ideas:

1. **Structured embeddings**: Instead of a flat embedding table, each token is represented as a composition of *lexical identity* + *semantic cluster membership* + *hierarchical abstraction level*, with a learned gate controlling structural influence.

2. **Masked diffusion generation**: Instead of predicting the next token left-to-right (autoregressive), the model learns to reconstruct randomly masked sequences — generating text by iteratively unmasking from fully masked to fully revealed, following the paradigm introduced by [LLaDA](https://arxiv.org/abs/2502.09992) (Large Language Diffusion with Masking).

The goal is to test whether structured, multi-scale embeddings provide meaningful benefits when paired with a non-autoregressive generative framework — specifically in terms of **sample efficiency**, **semantic organization**, and **cluster-guided generation**.

> This is a research prototype at ~20M parameters, designed to run on a **single consumer GPU** (8–16 GB VRAM) and produce measurable results within days, not months.

---

## 2. Why These Two Ideas Together

### Why masked diffusion instead of autoregressive?

Autoregressive models generate one token at a time, left to right. This is simple and proven, but it creates a sequential bottleneck at inference and forces the model into a rigid causal structure that can't naturally reason bidirectionally.

LLaDA showed that masked diffusion models — which mask random portions of text and train a Transformer to predict all masked tokens simultaneously — can match autoregressive models at scale while enabling:

- **Bidirectional context**: Every token attends to every other token (no causal mask).
- **Parallel generation**: Multiple tokens are revealed per step during sampling.
- **Reversal capability**: The model isn't constrained by left-to-right ordering.
- **Iterative refinement**: The model can revisit and correct its predictions across multiple denoising steps.

### Why hierarchical clustered embeddings?

Standard embeddings assign each token a single learned vector. All semantic structure (grouping, abstraction, similarity) must be discovered implicitly by the Transformer layers above. This works, but it means:

- The Transformer wastes capacity rediscovering structure that could be given for free.
- Rare tokens get poor representations due to limited gradient signal.
- There's no explicit organization in the embedding space — no clusters, no hierarchy.

Our embedding adds three structural components on top of the standard lookup:

- **Cluster membership** (fine-grained): Each token is softly assigned to learned prototype vectors, sharing structure with semantically related tokens.
- **Hierarchical level** (coarse-grained): Clusters are grouped into broader semantic categories, creating a multi-scale representation.
- **Gated composition**: A learned gate controls how much structural information is mixed into the base embedding, preventing the structure from dominating when it's not useful.

### Why they fit together

In a masked diffusion model, the Transformer sees the full sequence bidirectionally. This means the router that assigns tokens to clusters has access to **complete context from step one** — unlike in an autoregressive model where early tokens have almost no context. This addresses one of the key weaknesses identified in the critique of the original architecture: the **router bootstrapping problem**.

Additionally, the cluster structure may help the diffusion process: tokens in the same semantic cluster can be unmasked in coherent groups, and the hierarchical levels can provide coarse-to-fine generation guidance.

---

## 3. Architecture

### 3.1 Model Specifications

| Parameter | Value | Notes |
|---|---|---|
| Total parameters | ~20M | Fits on consumer GPU |
| Vocabulary size | 8,192 | BPE tokenizer |
| Embedding dimension | 384 | Shared across all components |
| Transformer layers | 8 | Bidirectional (no causal mask) |
| Attention heads | 6 | d_k = d_v = 64 |
| Feed-forward dim | 1,536 | 4x expansion, SwiGLU activation |
| Max sequence length | 512 | Fixed context window |
| Fine clusters (K) | 64 | Soft routing via softmax |
| Coarse clusters (M) | 8 | Bottom-up from fine clusters |
| Normalization | RMSNorm | Pre-norm configuration |
| Mask token | `[MASK]` | Special token ID for diffusion |

### 3.2 Architecture Schema

```
                         ┌─────────────────────────────────┐
                         │         OUTPUT HEAD              │
                         │   logits = W_local^T · h_final   │
                         │   (weight-tied with E_local)     │
                         └────────────────┬────────────────┘
                                          │
                         ┌────────────────┴────────────────┐
                         │   TRANSFORMER BACKBONE (×8)      │
                         │                                  │
                         │  ┌──────────────────────────┐   │
                         │  │  RMSNorm → MHSA (bidir)  │   │
                         │  │  6 heads, d_k=64          │   │
                         │  │  + residual               │   │
                         │  └──────────────────────────┘   │
                         │  ┌──────────────────────────┐   │
                         │  │  RMSNorm → SwiGLU FFN    │   │
                         │  │  384 → 1536 → 384        │   │
                         │  │  + residual               │   │
                         │  └──────────────────────────┘   │
                         └────────────────┬────────────────┘
                                          │
                         ┌────────────────┴────────────────┐
                         │     z₀ = COMPOSITE EMBEDDING     │
                         │                                  │
                         │  z₀ = e_local                    │
                         │     + g(x) ⊙ (α·e_clust + β·e_hier)
                         │     + e_pos                      │
                         └────────────────┬────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
  ┌─────────┴─────────┐       ┌──────────┴──────────┐       ┌─────────┴─────────┐
  │    E_LOCAL (W)     │       │  STRUCTURAL BRANCH  │       │   E_POSITIONAL     │
  │                    │       │                     │       │                    │
  │  nn.Embedding      │       │  ┌───────────────┐  │       │  Learned pos enc   │
  │  vocab=8192        │       │  │  GATE g(x)    │  │       │  max_len=512       │
  │  dim=384           │       │  │ max(σ(G·e),   │  │       │  dim=384           │
  │  mask_token=[MASK] │       │  │     g_min=0.1)│  │       └────────────────────┘
  └─────────┬──────────┘       │  └───────┬───────┘  │
            │                  │          │          │
            │                  │     ┌────┴────┐     │
            │                  │     │    ⊙    │     │
            │                  │     └────┬────┘     │
            │                  │          │          │
            │           ┌──────┴──────────┴──────┐   │
            │           │  α·e_cluster + β·e_hier │   │
            │           └──────┬──────────┬──────┘   │
            │                  │          │          │
            │       ┌──────────┴───┐  ┌───┴──────────┐
            │       │ FINE CLUSTER │  │COARSE CLUSTER│
            │       │              │  │  (bottom-up) │
            │       │ Router R_f   │  │              │
            │       │ softmax(     │  │ Router R_c   │
            │       │  R_f·e_local)│  │ softmax(     │
            │       │ K=64 clusters│  │  R_c·e_clust)│
            │       │ Centroids Cᵢ │  │ M=8 clusters │
            │       │              │  │ Centroids Hⱼ │
            │       └──────────────┘  └──────────────┘
            │
  ┌─────────┴──────────────────────────────────────────────┐
  │                    INPUT TOKENS                         │
  │                                                        │
  │  x = [The, little, [MASK], [MASK], a, [MASK], story]   │
  │                                                        │
  │  Masking ratio t ~ U[0,1] applied during training      │
  └────────────────────────────────────────────────────────┘
```

**Data flow:**
1. Input tokens are masked with ratio `t ~ U[0,1]`
2. `E_local` provides base embedding (standard lookup)
3. Fine router assigns tokens to K=64 clusters via softmax
4. Coarse router operates on fine cluster output (bottom-up) → M=8 clusters
5. Gate `g(x)` controls structural influence (floor at 0.1)
6. Composite: `z₀ = e_local + g ⊙ (α·e_cluster + β·e_hier) + e_pos`
7. Bidirectional Transformer (8 layers) processes all positions simultaneously
8. Output head predicts original tokens at masked positions (weight-tied)

### 3.3 Parameter Budget

```
Component                    Params       Share
─────────────────────────────────────────────────
E_local (W):                 3,146K       17.3%
Fine clusters + router:         89K        0.5%
Coarse clusters + router:       11K        0.1%
Gate network:                  148K        0.8%
Positional encoding:           197K        1.1%
Transformer (8 layers):     14,168K       77.8%
Meta-model (optional):         500K        2.7%
─────────────────────────────────────────────────
Total:                      ~18,259K      ~18.3M
```

### 3.3 Composite Embedding

For each token `x`, the final embedding is:

```
e_local   = W[x]                                    # Standard lookup
e_cluster = Σᵢ pᵢ(x) · Cᵢ                          # Soft cluster mix
e_hier    = Σⱼ qⱼ(x) · Hⱼ                           # Hierarchical mix
g(x)      = max(σ(G · e_local), g_min)              # Gate with floor

z₀(x)    = e_local + g(x) ⊙ (α·e_cluster + β·e_hier) + e_pos
```

Where:
- `p(x) = softmax(R_fine · e_local)` — fine router over K=64 clusters
- `q(x) = softmax(R_coarse · e_cluster)` — coarse router over M=8 clusters (bottom-up)
- `g_min = 0.1` — prevents gate from permanently shutting off structure
- `α, β` — learnable scalars, initialized to 0.1 and 0.05

### 3.4 Masked Diffusion Training (LLaDA paradigm)

**Forward process (masking):**

Given a clean sequence `x₀ = (x₁, x₂, ..., xₙ)`, sample a masking ratio `t ~ U[0, 1]`. Each token is independently replaced with `[MASK]` with probability `t`:

```
xₜ[i] = [MASK]  with probability t
xₜ[i] = x₀[i]  with probability 1 - t
```

**Training objective:**

```
L_diffusion = -𝔼_{t~U[0,1]} [ (1/|masked|) Σ_{i ∈ masked} log p_θ(x₀[i] | xₜ) ]
```

Cross-entropy loss over masked tokens only, with the masking ratio varying uniformly between 0 and 1. This is mathematically an upper bound on the negative log-likelihood.

**Key difference from BERT**: BERT masks at a fixed 15% ratio. LLaDA varies the ratio from 0 to 1, which makes it a proper generative model with principled likelihood estimation.

**Sampling (inference):**

```
1. Start: x = [MASK, MASK, ..., MASK]  (prompt tokens unmasked)
2. For each step s from S to 1:
   a. Predict logits for all masked positions
   b. Select most confident predictions
   c. Unmask those positions
   d. Repeat
3. Return fully unmasked sequence
```

Remasking strategy: `low_confidence` (unmask highest confidence first, can remask tokens that become less confident in subsequent steps).

### 3.5 Combined Loss

```
L_total = L_diffusion + λ₁·L_balance + λ₂·L_diversity + λ₃·L_hierarchy
```

Where:
- `L_balance`: Penalizes uneven cluster usage across a batch
- `L_diversity`: Penalizes high cosine similarity between centroids
- `L_hierarchy`: Encourages coarse clusters to be genuine abstractions of fine clusters
- Default lambdas: `λ₁=0.01, λ₂=0.001, λ₃=0.01`

---

## 4. Critical Design Decisions

### 4.1 Router bootstrapping (solved by bidirectional attention)

In an autoregressive model, the router `p(x) = softmax(R · e_local)` depends on `E_local`, which is random at initialization — making early routing meaningless.

In masked diffusion, the Transformer uses bidirectional attention. Even at step 1, every unmasked token sees every other unmasked token. This gives the embedding richer context earlier in training.

**Structural warmup:**
- Steps 0–500: Gate forced to 0. Only `E_local` trains. Clusters frozen.
- Steps 500–2000: Gate opens linearly from 0 to learned value. Clusters unfreeze.
- Steps 2000+: Full training with all components active.

### 4.2 Hierarchy is enforced structurally

The coarse router operates on `e_cluster` (the fine cluster output), NOT on `e_local`:

```
q(x) = softmax(R_coarse · e_cluster)
```

This forces the coarse level to be a genuine abstraction of the fine level.

### 4.3 Gate collapse prevention

- Gate floor: `g(x) = max(σ(G · e_local), 0.1)`
- During warmup, gate is externally controlled (not learned)
- Gate activation distribution is monitored per epoch

### 4.4 Positional encoding is added after composite embedding

```
z₀ = composite_embedding(x) + E_pos(position)
```

The router never sees position. Cluster assignment is position-independent.

### 4.5 Meta-optimizer (optional, Phase 5)

A small MLP (~500K params) that modulates gradient magnitude per parameter group. Trained via simple online regression on loss deltas. Applied every N steps.

> Optional and disabled by default. Enable only after the base system is stable.

---

## 5. Project Structure

```
hclm-d/
├── README.md                       # This file
├── config.toml                     # Unified project configuration
├── pyproject.toml                  # Dependencies + z86 CLI entry point
├── Makefile                        # Legacy shortcuts (prefer z86 CLI)
│
├── cli/                            # z86 unified CLI
│   ├── main.py                     # Argparse entry point (12 subcommands)
│   ├── ui.py                       # ANSI terminal output (ZARNETTI palette)
│   ├── registry.py                 # Version manifest manager
│   ├── cmd_init.py                 # z86 init / z86 doctor
│   ├── cmd_train.py                # z86 train
│   ├── cmd_versions.py             # z86 versions / diff / delete
│   ├── cmd_eval.py                 # z86 eval
│   ├── cmd_generate.py             # z86 generate / z86 serve
│   ├── cmd_ablation.py             # z86 ablation
│   ├── cmd_dashboard.py            # z86 dashboard
│   └── cmd_cloud.py                # z86 cloud (RunPod + Cloudflare)
│
├── cloud/                          # Cloud provider integrations
│   ├── runpod.py                   # RunPod GPU pod management (GraphQL)
│   ├── cloudflare.py               # Cloudflare DNS management
│   └── groq.py                     # Groq API (LLM judge)
│
├── configs/
│   ├── base.yaml                   # Default 20M config
│   └── ablations/                  # A0–A5 ablation configs
│
├── dashboard/                      # Web dashboard (Bun + Hono + React)
│   ├── server/                     # API server + SQLite + WebSocket
│   │   ├── index.ts                # Hono routes + WS broadcast
│   │   └── db.ts                   # SQLite (metrics + evals tables)
│   ├── src/                        # React 19 + Vite 6 + Tailwind 4
│   │   ├── app.tsx                 # Router (3 pages)
│   │   ├── components/             # MetricCell, CanvasChart, Layout...
│   │   ├── pages/                  # Overview, Evals, Versions
│   │   └── lib/                    # API hooks, fetch helpers
│   └── GUIDE.md                    # Full dashboard & CLI documentation
│
├── checkpoints/                    # Model checkpoints + manifest.json
│
├── data/                           # Data loading and preprocessing
│   ├── tokenizer.py                # BPE tokenizer (8192 vocab)
│   ├── dataset.py                  # Streaming dataset with masking
│   ├── masking.py                  # Diffusion masking logic (t ~ U[0,1])
│   └── prep.py                     # Download + preprocess TinyStories
│
├── model/                          # Model architecture
│   ├── config.py                   # ModelConfig dataclass
│   ├── lm.py                       # Full model: embedding → backbone → head
│   ├── head.py                     # Output projection (weight-tied with E_local)
│   ├── embedding/                  # Composite embedding components
│   │   ├── local.py                # E_local: nn.Embedding + mask token
│   │   ├── router.py               # Fine router + coarse router (bottom-up)
│   │   ├── clusters.py             # Centroid matrices (K=64, M=8)
│   │   ├── gate.py                 # Learned gate with floor
│   │   ├── positional.py           # Learned positional encoding
│   │   └── composite.py            # Full composite: local + gate*(cluster+hier) + pos
│   └── transformer/                # Bidirectional Transformer backbone
│       ├── attention.py            # Multi-head self-attention (BIDIRECTIONAL)
│       ├── ffn.py                  # SwiGLU feed-forward
│       ├── block.py                # Pre-norm Transformer block
│       └── backbone.py             # Stack of N blocks
│
├── diffusion/                      # Masked diffusion framework
│   ├── noise.py                    # Forward process: masking with t ~ U[0,1]
│   ├── sampling.py                 # Reverse process: iterative unmasking
│   └── schedules.py                # Step schedules for generation
│
├── losses/                         # Loss functions
│   ├── diffusion_loss.py           # Masked token prediction (cross-entropy)
│   ├── balance.py                  # Cluster usage balance
│   ├── diversity.py                # Centroid diversity
│   ├── hierarchy.py                # Coarse-fine consistency
│   └── combined.py                 # L_total = L_diff + λ₁L_bal + λ₂L_div + λ₃L_hier
│
├── meta/                           # Optional meta-optimizer (Phase 5)
│   ├── stats.py                    # Per-group gradient statistics
│   ├── meta_model.py               # Small MLP: stats → α_k
│   ├── groups.py                   # Parameter group definitions
│   └── scheduler.py                # Meta-model invocation schedule
│
├── training/                       # Training loop and utilities
│   ├── trainer.py                  # Main training loop
│   ├── optimizer.py                # AdamW + cosine LR schedule
│   ├── warmup.py                   # Structural warmup (gate schedule)
│   ├── checkpointing.py            # Save/load model + optimizer state
│   └── dashboard_reporter.py       # Real-time metric push to dashboard
│
├── eval/                           # Evaluation pipeline
│   ├── perplexity.py               # NLL / bits-per-byte
│   ├── generation.py               # Sample generation quality
│   ├── cluster_health.py           # Usage entropy, dead clusters
│   ├── hierarchy_metrics.py        # Coarse-fine alignment scores
│   ├── gate_analysis.py            # Gate activation distribution
│   ├── llm_judge.py                # Groq LLM judge (6 criteria)
│   └── agent_eval.py               # Agno agent wrapper for eval
│
├── scripts/                        # Entry point scripts
│   ├── train.py                    # python scripts/train.py --config ...
│   ├── pod_bootstrap.sh            # RunPod bootstrap script
│   └── runpod_ssh.py               # Paramiko SSH helper
│
└── tests/                          # Unit tests
```

---

## 6. Training Pipeline

### 6.1 Data

**TinyStories** (~500M tokens) for the initial prototype. Simple enough to show whether the embedding structure helps with sample efficiency at small scale.

Later: WikiText-103 or a SlimPajama subset for harder evaluation.

### 6.2 Training Phases

| Phase | Steps | What happens |
|---|---|---|
| 0 | 0–500 | Only `E_local` trains. Gate=0. Clusters frozen. |
| 1 | 500–2K | Gate opens 0→learned. Clusters unfreeze. Router starts. |
| 2 | 2K–50K | Full training. All components active. |
| 3 | 50K–100K | Fine-tune lambdas. Monitor cluster health. |
| 4 | 100K+ | (Optional) Enable meta-optimizer on embedding groups only. |

### 6.3 Optimizer

AdamW with:
- Peak LR: `3e-4`
- Warmup: 1000 steps (cosine)
- Weight decay: `0.1`
- Gradient clipping: `1.0`
- Batch size: `64 sequences × 512 tokens = 32K tokens/batch`

### 6.4 Hardware Target

- Single GPU: RTX 3090 / 4090 (24 GB) or A100 (40 GB)
- Training time estimate: ~2–4 days on TinyStories
- Mixed precision: bfloat16

---

## 7. Ablation Plan

| ID | Config | What it tests |
|---|---|---|
| A0 | `flat_autoregressive.yaml` | Reference: standard AR model, flat embedding |
| A1 | `flat_baseline.yaml` | Diffusion + flat embedding (is diffusion alone ok?) |
| A2 | `clusters_only.yaml` | Diffusion + flat clusters, no hierarchy |
| A3 | `no_gate.yaml` | Diffusion + clusters + hierarchy, no gate |
| A4 | `no_meta.yaml` | Full structure, standard AdamW |
| A5 | `full.yaml` | Everything enabled (if meta-model Phase 5 reached) |

### Metrics per run

- **Quality**: Train/val loss, NLL, bits-per-byte
- **Cluster health**: Usage entropy (should be high), dead cluster count (should be 0), centroid pairwise cosine (should be low)
- **Hierarchy**: Coarse-fine alignment score
- **Gate**: Mean activation, variance, per-token distribution
- **Efficiency**: Tokens/sec throughput, GPU memory peak
- **Generation**: Qualitative samples at checkpoints

---

## 8. Relationship to Prior Work

**[LLaDA](https://arxiv.org/abs/2502.09992)** (Nie et al., 2025) — The diffusion training paradigm. We adopt LLaDA's core formulation. Our contribution is in the embedding layer, not the diffusion framework itself.

**[LLaDA 2.1](https://arxiv.org/abs/2602.08676)** (Bie et al., 2026) — Token-to-Token editing, MoE at 100B scale, RL alignment. Compatible with future integration.

**[LLaDA-MoE](https://arxiv.org/abs/2509.24389)** (Zhu et al., 2025) — MoE with masked diffusion. Our hierarchical embedding shares the spirit of routing but operates at the embedding level.

**[SMDM](https://github.com/ML-GSAI/SMDM)** (Nie et al., 2025) — Scaling laws for masked diffusion. Closest reference implementation.

---

## 9. What Success Looks Like

1. **A2 > A1**: Adding clusters to diffusion should improve val loss.
2. **A3 ≈ A2 or A3 > A2**: Hierarchy should help or not hurt.
3. **Cluster health**: High entropy (>80% of max), no dead clusters, low centroid similarity.
4. **Gate partially open**: Mean activation between 0.2–0.8.
5. **Generation quality**: Coherent diffusion samples at 512 tokens.

---

## 10. Quick Start

### Using the z86 CLI (recommended)

```bash
# Install (registers the z86 command)
pip install -e .

# Check environment (Python, torch, CUDA, GPU, Bun)
z86 doctor

# Initialize project (install deps, prepare data, smoke test)
z86 init

# Train with default config
z86 train base

# Train and resume from a version
z86 train base --resume v3

# List model versions
z86 versions

# Compare two versions side-by-side
z86 diff v1 v2

# Evaluate a version (auto metrics + LLM judge)
z86 eval v2

# Generate text
z86 generate v2 --prompt "Once upon a time"

# Interactive generation REPL
z86 generate v2 --interactive

# Start HTTP generation server
z86 serve v2 --port 8080

# Run all ablations (A0–A5)
z86 ablation run

# Launch the web dashboard
z86 dashboard
```

### Manual (scripts)

```bash
pip install -e .
python scripts/train.py --config configs/base.yaml
python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"
```

### Dashboard

```bash
cd dashboard && bun install && bun run dev
# Open http://localhost:5173
# API server at http://localhost:3000
# WebSocket at ws://localhost:3000/ws
```

---

## 11. Dashboard & Eval Pipeline

### Web Dashboard

Real-time monitoring dashboard built with Bun + Hono + React 19 + Vite 6 + Tailwind 4. ZARNETTI terminal aesthetic (pure black, Geist Mono, dense panels).

**3 tabs**: Overview, Evals, Versions

**Key features**:
- Real-time metrics via WebSocket
- SQLite storage (WAL mode) for metrics and evals
- Version registry browser with multi-select comparison
- Eval results with radar charts, failure mode analysis, sample browser

### Eval Pipeline

Two-tier evaluation system:

1. **Auto metrics**: distinct-n, self-BLEU, repetition ratio, keyword hit rate, vocab richness
2. **LLM judge** (via Groq): coherence, fluency, relevance, creativity, consistency, instruction-following (1–5 scale)

Results are posted to `POST /api/evals` and displayed on the `/evals` dashboard page.

See `dashboard/GUIDE.md` for complete API reference, CLI documentation, and deployment guide.

---

## 12. Cloud Infrastructure

### RunPod GPU Management

```bash
z86 cloud status                  # Overview of all cloud resources
z86 cloud gpus                    # Available GPUs + pricing
z86 cloud volumes                 # Network volumes
z86 cloud start --preset train-fast  # Create GPU pod (A100)
z86 cloud stop <pod_id>           # Stop pod (pause billing)
z86 cloud terminate <pod_id>      # Destroy pod permanently
z86 cloud ssh <pod_id>            # Get SSH command
```

### GPU Presets

| Preset | GPU | VRAM | Use case |
|---|---|---|---|
| `train-small` | RTX A6000 | 24 GB | Quick runs (~24h for 100k steps) |
| `train-fast` | A100 80GB | 80 GB | Full training (~12h) |
| `inference` | RTX 4090 | 24 GB | Low-cost inference serving |

### Network Volumes

Persistent storage mounted at `/workspace`. Configure in `config.toml`:

```toml
[cloud.runpod]
volume_id = "YOUR_VOLUME_ID"
```

### Cloudflare DNS

```bash
z86 cloud dns                     # List DNS records
z86 cloud dns-set z86.dev 1.2.3.4 # Set A record
```

### Diffusion Sampling Schema

```
Step 0 (t=1.0):  [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
                   ↓ predict all → select most confident → unmask
Step 1 (t≈0.75): [MASK]  The  [MASK] [MASK] [MASK] [MASK]  was  [MASK]
                   ↓ predict masked → select most confident → unmask
Step 2 (t≈0.50): [MASK]  The   little [MASK]  had  [MASK]  was   happy
                   ↓ predict masked → select most confident → unmask
Step 3 (t≈0.25):  Once   The   little  girl   had  [MASK]  was   happy
                   ↓ predict remaining
Step 4 (t=0.0):   Once   The   little  girl   had    a     was   happy
                   ↓ done
Final:            "Once The little girl had a was happy"
```

Unlike autoregressive (left→right), diffusion fills in **any position** at each step,
prioritizing high-confidence predictions. This enables bidirectional coherence.

---

## 13. License

Apache 2.0

---

## 14. Citation

```bibtex
@misc{hclmd2026,
  title={HCLM-D: Hierarchical Clustered Embeddings for Masked Diffusion Language Models},
  year={2026}
}

@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and others},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}

@misc{bie2026llada21,
  title={LLaDA2.1: Speeding Up Text Diffusion via Token Editing},
  author={Bie, Tiwei and Cao, Maosong and others},
  year={2026},
  eprint={2602.08676},
  archivePrefix={arXiv}
}
```
