# Compositional Muon (CM) — implementation notes

Source: <https://blog.tilderesearch.com/blog/compositional-muon>
Reference impl: <https://github.com/tilde-research/comp-muon-release> (`src/compositional_muon.py`,
`src/msign.py`, `src/whitening.py`, `src/gauge.py`).

## 1. Idea

Vanilla Muon controls the operator norm of *each* weight update independently
(`Delta_W = -eta * msign(G)`). Compositional Muon instead controls the operator
norm of the *composed* update the loss actually sees — the QK product
`M = W_Q W_K^T` and the OV product `W_O W_V` — by whitening each factor's
gradient with its **partner's** inverse Gram root before the spectral sign, then
scaling by it again afterward.

```
C_K = (W_K^T W_K + lam I)^{1/2}                  # partner Gram root (damped)
Delta_Q = -(eta/2) * msign(G_Q C_K^{-1}) C_K^{-1}
Delta_K = -(eta/2) * msign(G_K C_Q^{-1}) C_Q^{-1}   # symmetric
```

`msign(A) = U V^T` for thin SVD `A = U S V^T` (Newton–Schulz, as in Muon). The
`eta/2` is the "half-split" budget: each factor gets half the trust region so the
*product* perturbation stays within the operator-norm budget.

## 2. Algorithm (half-split, full-whitening — the recommended default)

Per attention head (weights in math convention: `W_Q,W_K:(d_model,d_head)`,
`W_V:(d_model,d_v_head)`, `W_O:(d_v_head,d_model)`):

```
# 1. Momentum on RAW gradients (Muon SUM convention), per factor
m <- beta*m + G ; U = (beta*m + G if nesterov else m)        # one buffer per weight

# 2. Partner Gram inverse-root (per head)
C_K^{-1} = (W_K^T W_K + lam I)^{-1/2}      # QK: partner is K for Q, Q for K
C_Q^{-1} = (W_Q^T W_Q + lam I)^{-1/2}

# 3. Partner-whiten -> spectral sign -> partner-whiten
Delta_Q = msign(U_Q @ C_K^{-1}) @ C_K^{-1}
Delta_K = msign(U_K @ C_Q^{-1}) @ C_Q^{-1}

# 4. (optional) leg-norm restore, (optional) gauge/connection projection

# 5. Apply: decoupled weight decay + scaled update
W <- (1 - eta*wd) W
W <- W - (eta * budget * mp) * Delta        # budget=0.5 half-split, 1.0 joint
```

OV is analogous but the partner roots act on the *other* side
(`Delta_V = msign(C_O^{-1} G_V) ...`), and the recommended OV mode is **hybrid**:
V uses a per-head spectral sign, O uses a single per-matrix sign across all heads
(`G_O` is whitened per-head then stacked into one `(d_v, d_model)` sign).

Effective LR (no spectral shape-scale factor is applied):
`eta_eff = eta * budget * mp * (c^2 + lam)^{-1/2}`.

## 3. New hyperparameters (reference defaults)

| name | default | meaning |
|------|---------|---------|
| `mp` | `1.0` | CM learning-rate multiplier (blog swept 2–24) |
| `damping` (`lam`) | `1e-2` | Tikhonov reg added to the Gram before its root |
| `method` | `"half_split"` | `half_split` (budget 0.5, sign per factor) or `joint` (budget 1.0, one shared sign over stacked factors) |
| `isotropic` | `False` | `True` replaces the matrix Gram root with a per-head scalar `c=sqrt(\|\|W_h\|\|_F^2/d_head + lam)` |
| `hybrid` (OV only) | `True` | V per-head sign, O per-matrix sign |
| `whitening` | `"both"` | `both` / `pre` / `post` / `none` — partner-whitening ablation (OV legs) |
| `connection` | `"none"` | gauge fix: `none` / `frobenius` / `scale_aware` / `frobenius_scalar` / `scale_aware_scalar` |
| `momentum_reproject` | `False` | project momentum onto the horizontal bundle before the sign |
| `per_mat_renorm` | `False` | rescale each leg back to the Frobenius norm of its orthogonalized factor (never on QK joint) |
| `beta` | `0.95` | momentum coefficient (Muon convention, on raw grads) |
| `nesterov` | `False` | Nesterov momentum |
| `weight_decay` | `0.0` | decoupled weight decay |
| `eps` | `1e-7` | Frobenius-norm floor inside msign / Newton–Schulz |

Newton–Schulz for `msign`: 8-step **Polar Express** coefficient set, bf16 compute
(the circuit-muon default). Gram inverse-roots: coupled Newton–Schulz
(`coupled_inv_sqrt`, no eigh) for `connection="none"`; `eigh` (`eigh_inv_sqrt`,
reused by the Sylvester gauge solve) when a connection is active.

## 4. Integration into dion

dion optimizers are `torch.optim.Optimizer` subclasses that split params into
groups tagged by `algorithm` (`muon`/`dion`/`normuon`/`adamw`/`lion`), and already
support a per-head view via `num_heads` on a group. CM differs fundamentally:
its update couples **pairs** of weights (Q↔K, V↔O), so the optimizer must know the
pairing, not just process each param independently.

**Module**: `dion/compositional_muon.py` — a self-contained `CompositionalMuon`
optimizer plus the faithfully-ported math helpers (`msign`, partner Gram roots,
gauge connections, `qk_delta`/`ov_delta`). Mirrors `muon_reference.py`'s structure
(single-device math; DTensor handled by `full_tensor()` gather + re-shard, like
`MuonReference`) rather than the megabatch all-to-all path — the all-to-all base
assumes per-param-independent orthogonalization, which the paired partner-whitening
breaks. Exported from `dion/__init__.py`.

**Pairing API** (the part the blog leaves to the integrator): param groups tagged
`algorithm="cm_qk"` / `algorithm="cm_ov"`, whose `params` are listed **pairwise**
([Wq0,Wk0,Wq1,Wk1,...] for QK), carrying `head_dim`. Generic matrix params fall back
to `algorithm="muon"`, vectors/embeddings to `algorithm="adamw"` (same family
convention as `muon_reference`). Per-group CM knobs from §3 override the defaults.

## 5. GQA generalization (the blog's underspecified, load-bearing point)

The reference assumes MHA (`H_q == H_kv`, head `i` ↔ head `i`). Under GQA a shared
K/V head pairs with `G = H_q / H_kv` query heads. The faithful generalization
(reduces exactly to the reference for `G == 1`):

* **Per-query side** (Q, O): its partner is the *one* KV head of its group. Expand
  the KV-head inverse-Gram root over query heads — `C.repeat_interleave(G, dim=0)`
  (Llama `repeat_kv` grouping: query heads `[g·G:(g+1)·G]` share KV head `g`).
* **Shared side** (K, V): its partner is the *group* of `G` query heads. Aggregate
  the group's per-head Grams (`gram.view(H_kv, G, hd, hd).sum(1)`) before the root.
* **Budget**: the blog notes *"allocate K-side budget by ε/(2·num_query_heads_per_group)"* —
  the shared factor changes `G` products at once, so its step is `0.5/G` while the
  per-query factor keeps `0.5`. (`mp` multiplies both.)

## 6. Implementation choices (recommended path, per user direction)

1. **Variant surface** — recommended path only: `method="half_split"`,
   `whitening="both"`, OV `hybrid=True` (V per-head sign, O per-matrix sign),
   `connection="none"` (no gauge), `momentum_reproject=False`, `per_mat_renorm=False`,
   `isotropic=False`. (Joint / isotropic / gauge knobs left for a follow-up.)
2. **msign** — dion's existing 5-step `polar_express` (per user; integrates with
   the repo's Newton–Schulz), not the reference's 8-step set. Gram inverse-roots:
   faithful coupled Newton–Schulz (`coupled_inv_sqrt`, fp32, batched per head).
3. **Pairing API** — ordered pairwise `algorithm="cm_qk"` / `"cm_ov"` groups +
   `head_dim`; generic matrices fall back to `algorithm="muon"`, vectors/embeddings
   to `"adamw"` (drop-in single optimizer, like `muon_reference`).
4. **Distributed** — `full_tensor()` gather + re-shard per factor (FSDP2/DDP-correct;
   mirrors `muon_reference`). Compute is replicated across ranks, not yet
   compute-sharded; the megabatch all-to-all path (incompatible with paired
   partner-whitening as-is) is left as future work.
5. **GQA** — supported via §5; requires `H_q % H_kv == 0` and `head_dim` dividing
   both projection widths, else raises.
