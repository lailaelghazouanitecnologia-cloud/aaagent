# HCLM-D Training Diagnostics Report

**Date:** 2026-03-17
**Run:** 100K steps, base.yaml config
**Status at report:** Step 8800 / 100000

---

## 1. Summary

The training run shows healthy loss convergence (7.6 → 3.0) driven by the transformer backbone. However, **two auxiliary losses (balance and hierarchy) were never computed due to bugs in the trainer**, making it impossible to evaluate whether the hierarchical clustering adds value.

## 2. Bugs Found

### Bug 1: Balance loss never computed

**Location:** `training/trainer.py:323`

The `CombinedLoss.__call__()` accepts `routing_weights: Tensor | None` (the fine router's soft assignment probabilities, shape `[B, S, K]`). The trainer never passes this argument.

```python
# trainer.py:323 — current (broken)
loss_output = self.criterion(
    logits=logits,
    targets=input_ids,
    mask=mask,
    fine_centroids=fine_centroids,
    coarse_centroids=coarse_centroids,
    # routing_weights=???  ← MISSING
)
```

In `losses/combined.py:76`:
```python
if routing_weights is not None and self.lambda_balance > 0:
    l_bal = balance_loss(routing_weights)
```

Since `routing_weights` is always `None`, balance loss is always `0.0`. The `bal: 0.0000` in logs tells us nothing about cluster usage — it's a dead metric.

### Bug 2: Hierarchy loss never computed

**Location:** `training/trainer.py:323`

Same issue. `CombinedLoss.__call__()` accepts `fine_to_coarse_weights: Tensor | None` (the coarse router's assignment over fine cluster embeddings). The trainer never passes it.

In `losses/combined.py:84-90`:
```python
if (
    fine_centroids is not None
    and coarse_centroids is not None
    and fine_to_coarse_weights is not None  # ← always None
    and self.lambda_hierarchy > 0
):
    l_hier = hierarchy_loss(...)
```

The `hier: 0.0000` in logs is NOT evidence that the hierarchy is inactive — it's evidence that the loss function was never called.

### Bug 3: Routing weights not cached in forward pass

**Location:** `model/embedding/composite.py:127-133`

The forward pass computes `fine_weights` and `coarse_weights` but doesn't store them. `get_routing_info()` only returns the centroid modules and scalar params, not the per-token routing distributions needed for the loss.

### Root cause

The `get_routing_info()` method was designed to return model components (centroids, gate), but the trainer needs **per-batch tensor outputs** (routing weights) that only exist during the forward pass.

## 3. What The Logs Actually Tell Us

### Reliable metrics

| Metric | Status | Interpretation |
|--------|--------|----------------|
| **diff (diffusion loss)** | 7.6 → 3.0 | Transformer learning to predict masked tokens. Healthy convergence. |
| **div (diversity loss)** | 0 → 0.15 → 0.09 | Computed correctly (uses centroids, not routing weights). Fine centroid similarity peaked then decreased. |
| **val loss** | 5.79 → 3.21 | Tracks train loss, no overfitting. |

### Unreliable / broken metrics

| Metric | Status | Why |
|--------|--------|-----|
| **bal** | Always 0.0000 | Never computed (bug #1) |
| **hier** | Always 0.0000 | Never computed (bug #2) |

### Unknown (not logged)

| Metric | Why it matters |
|--------|----------------|
| **gate mean/std** | Is the gate letting structural info through? At what level? |
| **alpha value** | How much weight does the fine cluster component have? |
| **beta value** | How much weight does the hierarchy component have? |
| **router entropy** | Are clusters actually specializing, or is routing near-uniform? |

## 4. What We Can and Cannot Conclude

### Can conclude
- The transformer backbone (8 layers, 384 dim, ~17M params) is learning well
- Diversity loss is working and centroids are differentiating
- No overfitting (val tracks train)

### Cannot conclude
- Whether the 64 fine clusters contribute to prediction quality (need ablation)
- Whether the 8 coarse clusters do anything (hierarchy loss was never active)
- Whether the gate is open or closed (not logged)
- Whether balance loss would help prevent cluster collapse (never computed)
- Whether the router is actually routing or producing near-uniform distributions (entropy not logged)

## 5. Fixes Applied

### Fix 1: Cache routing weights in `composite.py`

Store `fine_weights`, `coarse_weights`, and gate values during forward pass. Expose them in `get_routing_info()`.

### Fix 2: Pass routing data to loss in `trainer.py`

Extract `routing_weights` and `fine_to_coarse_weights` from cached routing info and pass to `CombinedLoss`.

### Fix 3: Add diagnostic logging in `trainer.py`

Log every `log_every` steps:
- `alpha`, `beta` (learned scalar values)
- `gate_mean`, `gate_std` (gate activation statistics)
- `router_entropy` (measures cluster specialization; log2(64)=6.0 = uniform, lower = more specialized)

## 6. Recommended Next Steps

1. **Resume from checkpoint `step_5000.pt`** with fixes applied to get proper metrics
2. **Run to step 15K-20K** with proper logging, then evaluate if clusters help
3. **Ablation at step 20K**: compare val loss with `use_gate: false` (FlatEmbedding baseline)
4. **If clusters don't help**: increase alpha to 0.5, beta to 0.3, retrain from scratch
5. **If clusters help**: continue to 50K and evaluate hierarchy contribution

## 7. Time Estimates

| Scenario | Steps | Time |
|----------|-------|------|
| Current run to completion | 91,200 remaining | ~7.3 hours |
| Resume from 5K with fixes, run to 20K | 15,000 | ~1.2 hours |
| Quick ablation (flat vs hierarchical, 10K each) | 20,000 total | ~1.6 hours |

---

*Note: This report was generated after peer review identified the logging bugs. The original analysis incorrectly attributed `bal=0` and `hier=0` to model behavior rather than code bugs.*
