"""Compositional Muon (CM): partner-whitened steepest descent for QK / OV.

Muon controls the operator norm of *each* weight update. Compositional Muon
controls the operator norm of the *composed* update the loss actually sees -- the
QK product ``M = W_Q W_K^T`` and the OV product ``W_O W_V`` -- by whitening each
factor's gradient with its partner's inverse Gram root before the spectral sign
and scaling by it again afterward::

    C_K = (W_K^T W_K + lam I)^{1/2}
    Delta_Q = -(eta/2) msign(G_Q C_K^{-1}) C_K^{-1}
    Delta_K = -(eta/2) msign(G_K C_Q^{-1}) C_Q^{-1}    # symmetric; OV analogous

``msign(A) = U V^T`` for the thin SVD ``A = U S V^T`` (computed with Newton-Schulz,
reusing dion's :func:`polar_express`). The ``eta/2`` is the half-split budget so the
*product* perturbation stays within the operator-norm trust region.

This module ports the recommended configuration of the reference implementation
(``method="half_split"``, full partner whitening, OV ``hybrid`` granularity, no
gauge connection) and generalizes it to grouped-query attention and to FSDP2 /
DDP via per-factor ``full_tensor()`` gather.

Compositional Muon by Tilde Research:
    https://blog.tilderesearch.com/blog/compositional-muon
    https://github.com/tilde-research/comp-muon-release

Grouped-query attention generalization (reduces exactly to the MHA reference when
there is one query head per KV head): a shared K/V head pairs with ``G = H_q/H_kv``
query heads, so the per-query factor expands the KV-head Gram root over query heads
(``repeat_interleave``) while the shared factor aggregates the group's Grams (sum)
and takes ``1/G`` of the step budget (the blog allocates the shared-side budget as
``eta/(2 * query_heads_per_group)``).
"""

import math
import warnings
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
from torch.optim.optimizer import Optimizer, ParamsT
from typing import List, Optional, Tuple

from .normuon import normuon_normalization_stacked
from .polar_express import polar_express
from .scalar_opts import adamw_update, lion_update

_CM_ALGORITHMS = ("cm_qk", "cm_ov", "muon", "normuon", "adamw", "lion")

# Coupled Newton-Schulz (CANS, arxiv 2506.10935) schedule for the batched inverse
# Gram root: 9 tuned (a, b) pairs then classic (1.5, -0.5) padding. Drives
# Y -> W^{1/2}, Z -> W^{-1/2} with both legs symmetrized each step.
_CANS_COEFFS: List[Tuple[float, float]] = [
    (5.182503604966906, -5.178098480082684),
    (2.586120737395915, -0.6479542005271643),
    (2.567364126726186, -0.6454968804392178),
    (2.520560084348265, -0.6393528082067044),
    (2.410759275435182, -0.6248683598710716),
    (2.1883348130094173, -0.5952022073798908),
    (1.8595760874873613, -0.5504490972723968),
    (1.589020160467417, -0.5126569802066718),
    (1.5051653981684994, -0.5007377068751799),
] + [(1.5, -0.5)] * 16


# ---------------------------------------------------------------------------
# Math helpers (all in math / un-transposed convention, fp32)
# ---------------------------------------------------------------------------


def _msign(x: Tensor, eps: float) -> Tensor:
    """Spectral sign ``U V^T`` via dion's Polar Express Newton-Schulz (batched)."""
    return polar_express(x, eps).to(torch.float32)


def _coupled_inv_sqrt(gram: Tensor, damping: float) -> Tensor:
    """``(gram + damping I)^{-1/2}`` via the coupled Newton-Schulz iteration.

    Accepts a batched stack of SPD matrices ``(H, n, n)`` (each normalized on its
    own). No eigendecomposition; runs in fp32.
    """
    n = gram.shape[-1]
    eye = torch.eye(n, device=gram.device, dtype=torch.float32)
    W = gram.to(torch.float32) + damping * eye
    scale = (1 - 1e-3) / W.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    Y = W * scale
    Z = eye.expand_as(W).contiguous() if W.ndim == 3 else eye
    fused = torch.baddbmm if W.ndim == 3 else torch.addmm
    for a, b in _CANS_COEFFS:
        M = Z @ Y
        Y_next = fused(Y, Y, M, beta=a, alpha=b)
        Z_next = fused(Z, M, Z, beta=a, alpha=b)
        Y = 0.5 * (Y_next + Y_next.mT)
        Z = 0.5 * (Z_next + Z_next.mT)
    return Z * torch.sqrt(scale)


def _to_heads_row(W: Tensor, num_heads: int, head_dim: int) -> Tensor:
    """``(d, num_heads * head_dim) -> (num_heads, d, head_dim)`` (heads on cols)."""
    d = W.shape[0]
    return W.view(d, num_heads, head_dim).transpose(0, 1).contiguous()


def _from_heads_row(W_h: Tensor, d: int, cols: int) -> Tensor:
    """Inverse of :func:`_to_heads_row`: ``(num_heads, d, head_dim) -> (d, cols)``."""
    return W_h.transpose(0, 1).contiguous().view(d, cols)


def _partner_roots(
    gram_per_query: Tensor, gram_shared: Tensor, group_size: int, damping: float
) -> Tuple[Tensor, Tensor]:
    """Inverse Gram roots for a GQA factor pair.

    ``gram_per_query`` are the ``H_q`` per-query-head Grams; ``gram_shared`` the
    ``H_kv`` shared-head Grams. Returns ``(C_shared_inv, C_perquery_inv_expanded)``:
    the shared-side root is the per-query partner aggregated over each group (sum
    of the group's Grams), and the per-query-side root is the shared partner's root
    repeated over the query heads of its group. For MHA (``group_size == 1``) both
    are plain per-head roots.
    """
    H_kv = gram_shared.shape[0]
    head_dim = gram_shared.shape[-1]
    gram_grouped = gram_per_query.view(H_kv, group_size, head_dim, head_dim).sum(1)
    C_shared_inv = _coupled_inv_sqrt(gram_grouped, damping)
    C_perquery_inv = _coupled_inv_sqrt(gram_shared, damping)
    C_perquery_inv_expanded = C_perquery_inv.repeat_interleave(group_size, dim=0)
    return C_shared_inv, C_perquery_inv_expanded


def qk_delta(
    W_Q: Tensor,
    W_K: Tensor,
    G_Q: Tensor,
    G_K: Tensor,
    head_dim: int,
    *,
    damping: float = 1e-2,
    eps: float = 1e-7,
) -> Tuple[Tensor, Tensor]:
    """Half-split, full-whitening QK direction (math convention, fp32).

    ``W_Q, W_K, G_Q, G_K`` are ``(d_model, H * head_dim)`` (Q has ``H_q`` heads, K
    has ``H_kv``, with ``H_q`` a multiple of ``H_kv`` for GQA). Returns the
    orthogonalized directions ``(delta_Q, delta_K)`` before any learning-rate /
    budget / weight-decay wrapping.
    """
    d, d_q = W_Q.shape
    d_k = W_K.shape[1]
    H_q, H_kv = d_q // head_dim, d_k // head_dim
    if H_q % H_kv != 0:
        raise ValueError(
            f"QK GQA requires query heads ({H_q}) divisible by KV heads ({H_kv})."
        )
    group_size = H_q // H_kv

    W_Q_h = _to_heads_row(W_Q, H_q, head_dim)
    W_K_h = _to_heads_row(W_K, H_kv, head_dim)
    G_Q_h = _to_heads_row(G_Q, H_q, head_dim)
    G_K_h = _to_heads_row(G_K, H_kv, head_dim)

    # Partner is the opposite factor: K is whitened by (grouped) Q, Q by expanded K.
    C_Q_inv, C_K_inv_exp = _partner_roots(
        W_Q_h.mT @ W_Q_h, W_K_h.mT @ W_K_h, group_size, damping
    )

    delta_Q_h = _msign(G_Q_h @ C_K_inv_exp, eps) @ C_K_inv_exp
    delta_K_h = _msign(G_K_h @ C_Q_inv, eps) @ C_Q_inv

    return (
        _from_heads_row(delta_Q_h, d, d_q),
        _from_heads_row(delta_K_h, d, d_k),
    )


def ov_delta(
    W_V: Tensor,
    W_O: Tensor,
    G_V: Tensor,
    G_O: Tensor,
    head_dim: int,
    *,
    damping: float = 1e-2,
    eps: float = 1e-7,
) -> Tuple[Tensor, Tensor]:
    """Half-split, hybrid OV direction (math convention, fp32).

    ``W_V, G_V`` are ``(d_model, H_kv * head_dim)``; ``W_O, G_O`` are
    ``(H_q * head_dim, d_model)``. V uses a per-head spectral sign, O a single
    per-matrix sign across all its heads (the ``hybrid`` granularity). Returns
    the orthogonalized directions ``(delta_V, delta_O)``.
    """
    d, d_v = W_V.shape
    d_o = W_O.shape[0]
    H_kv, H_q = d_v // head_dim, d_o // head_dim
    if H_q % H_kv != 0:
        raise ValueError(
            f"OV GQA requires O heads ({H_q}) divisible by V heads ({H_kv})."
        )
    group_size = H_q // H_kv

    W_V_h = _to_heads_row(W_V, H_kv, head_dim)
    W_O_h = W_O.view(H_q, head_dim, d)
    G_V_h = _to_heads_row(G_V, H_kv, head_dim)
    G_O_h = G_O.view(H_q, head_dim, d)

    # V is whitened by (grouped) O; O by expanded V.
    C_O_inv, C_V_inv_exp = _partner_roots(
        W_O_h @ W_O_h.mT, W_V_h.mT @ W_V_h, group_size, damping
    )

    # V: per-head sign with both-sided partner whitening.
    delta_V_h = _msign(G_V_h @ C_O_inv, eps) @ C_O_inv

    # O: per-matrix sign across all heads, partner-whitened per head.
    G_O_w = C_V_inv_exp @ G_O_h
    M_O = _msign(G_O_w.reshape(d_o, d), eps).view(H_q, head_dim, d)
    delta_O_h = C_V_inv_exp @ M_O

    return (
        _from_heads_row(delta_V_h, d, d_v),
        delta_O_h.reshape(d_o, d),
    )


# ---------------------------------------------------------------------------
# Distributed factor gather / re-shard
# ---------------------------------------------------------------------------


def _full(x: Tensor) -> Tensor:
    """Gather a (possibly sharded) tensor to a full replicated local tensor."""
    return x.full_tensor() if isinstance(x, DTensor) else x


def _reshard_like(local: Tensor, ref: Tensor) -> Tensor:
    """Re-distribute a replicated ``local`` tensor to the sharding of ``ref``."""
    if not isinstance(ref, DTensor):
        return local
    return DTensor.from_local(
        local, device_mesh=ref.device_mesh, placements=None, run_check=False
    ).redistribute(placements=ref.placements)


def _unwrap_subclass(t: Tensor) -> Tensor:
    """Unwrap a training-weight tensor subclass (e.g. torchao's MXFP8 wrapper) to
    its plain high-precision master tensor, mirroring what ``full_tensor()`` yields
    on the gather path. The master is the subclass's first ``__tensor_flatten__``
    inner tensor (a ``._data`` attribute in practice); plain tensors pass through.
    Newton-Schulz / Gram matmuls reject the wrapper, so the local path must unwrap.
    """
    if type(t) is Tensor or isinstance(t, DTensor):
        return t
    flatten = getattr(t, "__tensor_flatten__", None)
    if flatten is not None:
        try:
            names, _ = flatten()
        except Exception:
            names = ()
        for name in names:
            inner = getattr(t, name, None)
            if isinstance(inner, Tensor):
                return inner
    inner = getattr(t, "_data", None)
    return inner if isinstance(inner, Tensor) else t


def _head_local_shard(p: Tensor, head_dim: int) -> Optional[Tensor]:
    """Return the (unwrapped) local shard if ``p`` is a DTensor sharded on dim 0
    along head boundaries (and not otherwise), so per-head work needs no comm.

    Returns ``None`` (caller falls back to the gather path) when ``p`` is not a
    DTensor, is sharded on a dim other than 0, is sharded on more than one mesh
    dim, or its local row count is not a positive multiple of ``head_dim`` (an
    uneven / non-head-aligned shard).
    """
    if not isinstance(p, DTensor):
        return None
    shards = [pl for pl in p.placements if isinstance(pl, Shard)]
    if len(shards) != 1 or shards[0].dim != 0:
        return None
    local = _unwrap_subclass(p.to_local())
    if local.shape[0] == 0 or local.shape[0] % head_dim != 0:
        return None
    return local


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class CompositionalMuon(Optimizer):
    """Compositional Muon optimizer (FSDP2 / DDP / single-device).

    Parameters are split into groups by an ``algorithm`` tag:

    * ``"cm_qk"`` -- attention QK pairs. ``params`` are listed pairwise
      ``[W_Q0, W_K0, W_Q1, W_K1, ...]`` (``nn.Linear`` weights, i.e. ``q_proj.weight``
      of shape ``(H_q * head_dim, d_model)``). The group must set ``head_dim``.
    * ``"cm_ov"`` -- attention OV pairs, listed pairwise ``[W_V0, W_O0, ...]``
      (``v_proj.weight`` ``(H_kv * head_dim, d_model)`` then ``o_proj.weight``
      ``(d_model, H_q * head_dim)``). The group must set ``head_dim``.
    * ``"muon"`` / ``"normuon"`` -- generic 2D matrices, vanilla Muon or NorMuon
      (per-head when ``num_heads`` is set; NorMuon adds the per-neuron variance EMA).
    * ``"adamw"`` / ``"lion"`` -- element-wise fallback for vectors / embeddings.

    Grouped-query attention is supported: ``H_q`` need only be a multiple of
    ``H_kv``. Distributed weights are handled per factor: when a QK pair is sharded
    on heads (FSDP2 dim-0 shards along head boundaries, each rank holding whole
    KV-groups) the per-head CM math runs on the local shard with no communication;
    otherwise — and for the OV ``O`` factor, which is sharded on the hidden axis
    rather than heads — the factor is gathered (``full_tensor``), computed
    replicated, and the update re-sharded.

    Args:
        params: Parameters or param groups for the optimizer.
        lr: Base learning rate (``eta``). CM applies no spectral shape-scale factor;
            the effective rate is ``lr * budget * mp * (c^2 + lam)^{-1/2}``.
        mu: Momentum factor (on raw gradients, Muon convention).
        muon_beta2: Second beta for the ``"normuon"`` fallback's per-neuron variance EMA.
        betas: ``(beta1, beta2)`` for AdamW / Lion fallbacks.
        weight_decay: Decoupled weight decay.
        mp: CM learning-rate multiplier.
        damping: Tikhonov ``lam`` added to the Gram before its inverse root.
        nesterov: Use Nesterov momentum.
        epsilon: Small value for AdamW denominator / Newton-Schulz pre-norm floor.
        adjust_lr: LR adjustment for the ``"muon"`` / ``"normuon"`` fallback only
            (``"spectral_norm"`` / ``"rms_norm"`` / ``None``).

    Compositional Muon by Tilde Research:
        https://blog.tilderesearch.com/blog/compositional-muon
    Muon by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        mp: float = 1.0,
        damping: float = 1e-2,
        nesterov: bool = False,
        epsilon: float = 1e-8,
        adjust_lr: Optional[str] = "spectral_norm",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if muon_beta2 < 0.0:
            raise ValueError(f"Invalid muon_beta2: {muon_beta2}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. "
                "Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            mu=mu,
            muon_beta2=muon_beta2,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            mp=mp,
            damping=damping,
            nesterov=nesterov,
            epsilon=epsilon,
            adjust_lr=adjust_lr,
            algorithm="muon",
            head_dim=None,
            num_heads=None,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            algo = group["algorithm"]
            if algo not in _CM_ALGORITHMS:
                raise ValueError(
                    f"Unknown algorithm {algo!r}; choose from {_CM_ALGORITHMS}."
                )
            if algo in ("cm_qk", "cm_ov"):
                if group["head_dim"] is None:
                    raise ValueError(
                        f"algorithm={algo!r} requires 'head_dim' on the group."
                    )
                if len(group["params"]) % 2 != 0:
                    raise ValueError(
                        f"algorithm={algo!r} expects params listed pairwise; "
                        f"got an odd count ({len(group['params'])})."
                    )
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            f"{algo} requires 2D parameters, got {p.ndim}D."
                        )
            elif algo in ("muon", "normuon"):
                for p in group["params"]:
                    if p.ndim < 2:
                        raise ValueError(
                            f"{algo} requires matrix parameters (ndim >= 2)."
                        )

    def _momentum_direction(self, p: Tensor, group: dict) -> Tensor:
        """SUM-style momentum (Muon convention) on the raw grad; returns bf16 dir."""
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p.grad)
        m = state["momentum"]
        m.mul_(group["mu"]).add_(p.grad)
        u = m.mul(group["mu"]).add_(p.grad) if group["nesterov"] else m
        return u.to(torch.bfloat16)

    def _apply_update(
        self,
        p: Tensor,
        delta_nn: Tensor,
        base_lr: float,
        adjusted_lr: float,
        weight_decay: float,
    ) -> None:
        """Decoupled weight decay + re-sharded CM update on the weight in place."""
        update = _reshard_like(delta_nn.to(torch.bfloat16), p.data)
        p.data.mul_(1 - base_lr * weight_decay)
        p.data.sub_(update * adjusted_lr)

    def _cm_pair_step(self, group: dict, is_qk: bool) -> None:
        delta_fn = qk_delta if is_qk else ov_delta
        params = group["params"]
        head_dim = group["head_dim"]
        lr = group["lr"]
        wd = group["weight_decay"]
        budget = 0.5 * group["mp"]
        damping = group["damping"]

        for i in range(0, len(params), 2):
            p_a, p_b = params[i], params[i + 1]  # (Q, K) or (V, O)
            if p_a.grad is None and p_b.grad is None:
                continue
            if p_a.grad is None or p_b.grad is None:
                raise ValueError(
                    "Compositional Muon updates QK / OV factors jointly; both "
                    "weights in a pair must receive gradients."
                )

            # The shared factor (K for QK, V for OV) has the smaller head count and
            # changes group_size products at once, so it takes 1/group_size budget.
            # Head counts come from the head-packed dimension of each nn.Linear
            # weight: Q/K/V pack heads on dim 0 (out_features), but O packs them on
            # dim 1 (in_features = H_q * head_dim), so read O from shape[1].
            shared_heads = (
                p_b.shape[0] // head_dim if is_qk else p_a.shape[0] // head_dim
            )
            perquery_heads = (
                p_a.shape[0] // head_dim if is_qk else p_b.shape[1] // head_dim
            )
            group_size = perquery_heads // shared_heads
            if is_qk:
                lr_a, lr_b = lr * budget, lr * budget / group_size  # Q full, K shared
            else:
                lr_a, lr_b = lr * budget / group_size, lr * budget  # V shared, O full

            U_a = self._momentum_direction(p_a, group)
            U_b = self._momentum_direction(p_b, group)

            # No-comm fast path: when both QK factors are sharded on heads, each rank
            # holds whole KV-groups, so the per-head CM math runs on the local shard
            # with no gather. (OV's O factor is sharded on the hidden axis, not heads,
            # so it stays on the gather path.) The deltas are computed up front and
            # applied only if both succeed; any error falls back to the (validated)
            # gather path so an unexpected sharding can never break the step.
            if is_qk:
                deltas = None
                try:
                    deltas = self._cm_qk_local_deltas(
                        p_a, p_b, U_a, U_b, head_dim, group_size, damping
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    self._warn_local_fallback(exc)
                if deltas is not None:
                    self._apply_local(p_a, deltas[0], lr, lr_a, wd)
                    self._apply_local(p_b, deltas[1], lr, lr_b, wd)
                    continue

            # Gather full factors (replicated math), reshard the update.
            W_a = _full(p_a.data).mT.float()
            W_b = _full(p_b.data).mT.float()
            G_a = _full(U_a).mT.float()
            G_b = _full(U_b).mT.float()
            delta_a, delta_b = delta_fn(W_a, W_b, G_a, G_b, head_dim, damping=damping)
            self._apply_update(p_a, delta_a.mT, lr, lr_a, wd)
            self._apply_update(p_b, delta_b.mT, lr, lr_b, wd)

    def _warn_local_fallback(self, exc: Exception) -> None:
        if not getattr(self, "_local_fallback_warned", False):
            self._local_fallback_warned = True
            warnings.warn(
                "CompositionalMuon: the no-comm per-head QK path raised "
                f"({type(exc).__name__}: {exc}); falling back to the gather path "
                "for QK. Training is unaffected but the QK step is not "
                "communication-optimized.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _cm_qk_local_deltas(
        self,
        p_a: Tensor,
        p_b: Tensor,
        U_a: Tensor,
        U_b: Tensor,
        head_dim: int,
        group_size: int,
        damping: float,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Per-head QK directions computed on the local shard, no communication.

        Returns the local ``(delta_Q, delta_K)`` in nn.Linear convention when both
        factors are head-aligned shards whose local heads form whole KV-groups (so
        :func:`qk_delta` on the local heads is bit-identical to the gathered
        computation -- the QK math is per-head independent and group aggregation
        stays within the rank). Returns ``None`` (caller falls back) otherwise.
        """
        la = _head_local_shard(p_a.data, head_dim)
        lb = _head_local_shard(p_b.data, head_dim)
        if la is None or lb is None:
            return None
        local_q, local_kv = la.shape[0] // head_dim, lb.shape[0] // head_dim
        if (
            local_kv == 0
            or local_q % local_kv != 0
            or local_q // local_kv != group_size
        ):
            return None
        Ua = _unwrap_subclass(U_a.to_local() if isinstance(U_a, DTensor) else U_a)
        Ub = _unwrap_subclass(U_b.to_local() if isinstance(U_b, DTensor) else U_b)
        delta_q, delta_k = qk_delta(
            la.mT.float(),
            lb.mT.float(),
            Ua.mT.float(),
            Ub.mT.float(),
            head_dim,
            damping=damping,
        )
        return delta_q.mT, delta_k.mT

    def _apply_local(
        self,
        p: Tensor,
        delta_nn_local: Tensor,
        base_lr: float,
        adjusted_lr: float,
        weight_decay: float,
    ) -> None:
        """Decoupled weight decay + CM update applied to the local shard in place.

        Mutates the parameter's local tensor directly (the pattern dion's Muon /
        NorMuon megabatch path uses to write sharded updates), so no DTensor wrap
        or communication is needed. The master tensor behind a quantized weight
        wrapper is unwrapped so the in-place update lands on the high-precision
        weights (the wrapper recomputes its quantized copy each forward).
        """
        local = p.data.to_local() if isinstance(p.data, DTensor) else p.data
        local = _unwrap_subclass(local)
        update = delta_nn_local.to(local.dtype) * adjusted_lr
        local.mul_(1 - base_lr * weight_decay)
        local.sub_(update)

    def _ortho_fallback_step(self, group: dict, normuon: bool) -> None:
        """Vanilla Muon (or NorMuon when ``normuon``) fallback for non-CM matrices.

        Gathers each (possibly sharded) weight, orthogonalizes its momentum
        direction (per-head when ``num_heads`` is set), optionally applies
        NorMuon's per-neuron variance normalization, then applies the re-sharded
        update. NorMuon reuses dion's :func:`normuon_normalization_stacked` so the
        fallback matches the standalone optimizer.
        """
        lr = group["lr"]
        wd = group["weight_decay"]
        eps = group["epsilon"]
        num_heads = group["num_heads"]
        muon_beta2 = torch.tensor(group["muon_beta2"]) if normuon else None
        for p in group["params"]:
            if p.grad is None:
                continue
            U = self._momentum_direction(p, group)
            g_full = _full(U)
            if g_full.ndim > 2:
                g_full = g_full.view(g_full.size(0), -1)
            if num_heads is not None and num_heads > 1:
                head_dim = g_full.size(0) // num_heads
                u = _msign(g_full.view(num_heads, head_dim, -1).float(), eps)
                u = u.reshape(g_full.shape)
            else:
                u = _msign(g_full.float(), eps)
            if normuon:
                state = self.state[p]
                if "variance_neuron" not in state:
                    state["variance_neuron"] = torch.zeros_like(u[..., 0:1])
                v = state["variance_neuron"]
                # Per-row variance is independent of head grouping, so normalize
                # the full 2D update (one matrix -> singleton stack dim).
                u_norm, v_new = normuon_normalization_stacked(
                    u.unsqueeze(0), v.unsqueeze(0), muon_beta2
                )
                u = u_norm[0]
                v.copy_(v_new[0])
            adjust = group["adjust_lr"]
            if adjust == "spectral_norm":
                adjusted_lr = lr * math.sqrt(p.shape[-2] / p.shape[-1])
            elif adjust == "rms_norm":
                adjusted_lr = lr * 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))
            else:
                adjusted_lr = lr
            self._apply_update(p, u, lr, adjusted_lr, wd)

    def _scalar_step(self, group: dict) -> None:
        algo = group["algorithm"]
        lr = torch.tensor(group["lr"])
        beta1 = torch.tensor(group["beta1"])
        beta2 = torch.tensor(group["beta2"])
        wd = torch.tensor(group["weight_decay"])
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p)
                if algo == "adamw":
                    state["variance"] = torch.zeros_like(p)
                state["step"] = 0
            state["step"] += 1
            if algo == "adamw":
                adamw_update(
                    p.data,
                    p.grad,
                    state["momentum"],
                    state["variance"],
                    lr,
                    beta1,
                    beta2,
                    wd,
                    state["step"],
                    group["epsilon"],
                )
            else:
                lion_update(p.data, p.grad, state["momentum"], lr, beta1, beta2, wd)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            algo = group["algorithm"]
            if algo == "cm_qk":
                self._cm_pair_step(group, is_qk=True)
            elif algo == "cm_ov":
                self._cm_pair_step(group, is_qk=False)
            elif algo == "muon":
                self._ortho_fallback_step(group, normuon=False)
            elif algo == "normuon":
                self._ortho_fallback_step(group, normuon=True)
            else:
                self._scalar_step(group)

        return loss
