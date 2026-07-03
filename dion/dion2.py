import math
import torch
import torch.distributed as dist
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import (
    DistributedOrthoBase,
    megabatch_orthogonalize_async,
    muon_update_newton_schulz,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)
from .opt_utils import AsyncTask, to_local


class Dion2(DistributedOrthoBase):
    """
    Distributed Dion2 optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        fraction: Fraction of submatrix to orthogonalize per update (0 < fraction <= 1).
        ef_decay: Error-feedback decay factor applied to selected submatrix.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_gram_newton_schulz: Whether to use Gram Newton-Schulz for orthogonalization.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is ``func(input: Tensor, epsilon: float) -> Tensor``.
        verbose: Whether to print debug information during updates. If True, it prints whether rows or columns are selected for the submatrix selection process.
        selection_scope: On the FSDP2 row-sharded path, how the orthogonalized
            submatrix is selected. "global" (default): exact top-k on the
            assembled whole matrix -- layout-invariant/reproducible and better-
            converging. "local": per-shard top-k (union) -- cheaper comm but a
            sharding-dependent approximation that converges slightly worse; opt
            in when comm-bound at large scale. "global_capped": exact global
            top-``ceil(fraction*global)`` selection at ~local comm cost --
            all-gather the row norms only (~KBs), then pack each rank's winner
            rows into fixed-size chunks POOLED across the megabatch's matrices
            (an int32 count header rides in each chunk). A rank whose winners
            overflow its pooled chunk defers the overflow via error feedback;
            see ``capacity_factor``. Groups where the packing cannot save comm
            route to "global". No-op off the row-sharded path.
        capacity_factor: "global_capped" pooled-chunk slack. None (default) =
            auto per group, ``1 + 2*sqrt((1-1/world)/(per_rank*k))`` (covers
            ~2 sigma of winner-count fluctuation); a float >= 1.0 pins it.

    Dion2 optimizer by Ahn et al.: TBD
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        use_gram_newton_schulz: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        verbose: bool = False,
        triton_post_ortho: bool = False,
        selection_scope: str = "global",
        capacity_factor: Optional[float] = None,
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if selection_scope not in ("local", "global", "global_capped"):
            raise ValueError(
                f"selection_scope must be 'local', 'global', or 'global_capped', "
                f"got {selection_scope!r}"
            )
        if capacity_factor is not None and capacity_factor < 1.0:
            raise ValueError(
                f"capacity_factor must be None (auto) or >= 1.0, got {capacity_factor}"
            )
        if ef_decay < 0.0:
            raise ValueError(f"Invalid ef_decay: {ef_decay}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            ef_decay=ef_decay,
            fraction=float(fraction),
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="dion2",
            step=0,
            selection_scope=selection_scope,
            capacity_factor=capacity_factor,
        )
        super().__init__(
            params, distributed_mesh, "dion2", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )
        self.verbose = verbose
        if triton_post_ortho:
            from .dion2_triton import TRITON_AVAILABLE
            if not TRITON_AVAILABLE:
                raise ImportError(
                    "triton_post_ortho=True requires the 'triton' package, which is not installed. "
                    "Install it with: pip install dion[triton]  (or: pip install triton)"
                )
        self._triton_post_ortho = triton_post_ortho

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched Dion2 task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        # New optimizer step: reset the rank-local capped-deferral counters so
        # they accumulate exactly this step's packed megabatches.
        CAPPED_STATS.clear()
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Dion2 only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            update_args = dict(
                lr=torch.tensor(group["lr"]),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=group["fraction"],
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
                verbose=self.verbose,
                triton_post_ortho=self._triton_post_ortho,
                selection_scope=group["selection_scope"],
                capacity_factor=group["capacity_factor"],
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            num_heads = self._resolve_num_heads(group)
            if group.get("split_sizes") is not None:
                raise NotImplementedError(
                    "split_sizes is currently supported only by Muon and NorMuon."
                )

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, self._algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                if num_heads is not None:
                    params, gradients, momentums = self._prepare_head_split(
                        num_heads, params, gradients, momentums
                    )
                    megabatch_args = {**update_args, "process_group": None}
                    shard_dim = None
                else:
                    is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                        self._get_shard_info(params[0], group)
                    )
                    megabatch_args = update_args
                    if is_batch_sharded and not is_matrix_sharded:
                        megabatch_args = {**update_args, "process_group": None}
                    shard_dim = sharded_tensor_dim

                yield AsyncTask(
                    dion2_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


def dion2_update_megabatch_async(
    X: List[Tensor],  # All same-shape params (may be more than world_size)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    ef_decay: Tensor,  # Error-feedback factor (scalar tensor)
    fraction: float,  # Fraction of submatrix to orthogonalize (0 < fraction <= 1)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    verbose: bool = False,
    triton_post_ortho: bool = False,
    selection_scope: str = "global",  # "global" (exact whole-matrix top-k, default) or "local" (per-shard top-k; cheaper comm, sharding-variant)
    capacity_factor: Optional[float] = None,  # "global_capped" pooled-chunk slack; None = auto (z=2 model)
) -> Generator[None, None, None]:
    """
    Mega-batched Dion2 update: processes ALL same-shape parameters in one
    communication round instead of world_size-sized batches.

    ``selection_scope`` controls how the orthogonalized submatrix is chosen on
    the row-sharded path:

    - ``"global"`` (default): the full shard is communicated (like NorMuon), the
      top-k is taken on the assembled whole matrix, and Newton-Schulz runs on
      that submatrix. Comm is full-size but the selected set is the exact global
      top-k -- invariant to the sharding layout (reproducible across world
      sizes) and, in A/B tests, better-converging than "local" (which under-
      performed it by ~0.09 nat at matched steps on a 1.5B dense run).
    - ``"local"``: each rank picks its own top-k rows, and only those rows are
      communicated and orthogonalized, so comm and Newton-Schulz cost scale with
      ``fraction``. The selected set is the union of per-rank top-k -- a
      sharding-dependent approximation of the true top-k (world-size variant).
      Cheaper comm (the win grows with model size), but converges slightly
      worse; opt in when comm-bound at large scale.
    - ``"global_capped"``: exact global top-``ceil(fraction*global)`` selection
      at ~"local" comm cost, via packed transport. The slice L1 norms are
      all-gathered (one fp32 per slice, ~KBs) and every rank derives the same
      exact-count winner set (stable-sort tie-break). Each rank packs its
      winner rows into a fixed ``budget``-row chunk per destination, POOLED
      across the destination's ``per_rank`` matrices, with an int32 count
      header bit-cast into the chunk's first row -- the receiver parses the
      sender's plan instead of recomputing it. Winners that overflow the
      pooled budget are deferred: they skip error-feedback decay (only applied
      rows are decayed, by the masked post), so their momentum accumulates and
      they win later. Non-winners are never applied and never decayed -- a
      per-rank top-k fill would be trajectory-identical to "local". Groups
      whose budget would reach the full shard route to "global" instead.

    Off the row-sharded path (per-head, single-GPU, batch-sharded) each rank
    already holds whole matrices, so local and global selection coincide and
    ``selection_scope`` is a no-op.
    """
    N = len(X)
    assert N == len(G) == len(M)

    # Determine selection dimension based on sharding and tensor shape:
    # For sharded matrices, we align select_dim with shard_dim
    # For unsharded matrices (DDP or single-GPU), we select the shorter dimension
    ndim = X[0].ndim
    select_dim = None
    is_sharded = shard_dim is not None

    if is_sharded:
        shard_dim_neg = shard_dim if shard_dim < 0 else shard_dim - ndim
        if shard_dim_neg == -2:
            select_dim = -2  # Row-sharded
        elif shard_dim_neg == -1:
            select_dim = -1  # Column-sharded

    if select_dim is None:
        num_rows, num_cols = X[0].shape[-2:]
        select_dim = -2 if num_rows <= num_cols else -1

    if verbose:
        _print_selection_choice(X[0].shape, shard_dim, select_dim, ndim)

    # comm_dim for sharded communication: use select_dim (which equals normalized shard_dim)
    comm_dim = select_dim if is_sharded else None

    # On the sharded path X[0] must still be a DTensor, so .shape[comm_dim]
    # is the unsharded global size. The megabatch fn uses this to compute
    # the rank-consistent pad size for its alltoall. Catch the case where a
    # future refactor moves to_local(X) above this point and silently
    # collapses .shape to the local size.
    if comm_dim is not None:
        if not isinstance(X[0], DTensor):
            raise TypeError(
                "Sharded path requires X[0] to be a DTensor so .shape gives "
                f"the global size; got {type(X[0]).__name__}."
            )
        global_dim_size = X[0].shape[comm_dim]
    else:
        global_dim_size = None

    # --- "global_capped" packed-path decision (row-sharded 2D only) ---
    # Pooled chunk budget: per (rank -> dest) pair, B = ceil(c * per_rank * k)
    # rows shared across the dest's per_rank matrices. c defaults to the
    # z=2 fluctuation model (see dion2_capped_pack); if the budget reaches the
    # full shard, packing saves nothing and the group routes to exact "global".
    capped_packed = (
        selection_scope == "global_capped"
        and comm_dim is not None
        and process_group is not None
        and select_dim == -2
        and X[0].ndim == 2
    )
    if capped_packed:
        padded_local = (global_dim_size + world_size - 1) // world_size
        k = max(1, int(math.ceil(fraction * padded_local)))
        per_rank = (N + world_size - 1) // world_size
        if capacity_factor is None:
            c = 1.0 + 2.0 * math.sqrt((1.0 - 1.0 / world_size) / (per_rank * k))
        else:
            c = float(capacity_factor)
        budget = int(math.ceil(c * per_rank * k))
        # Route to exact "global" when packing cannot pay for itself -- the
        # packed chunk sends 1 + budget rows (count header included), so
        # break-even against sending the full shard is budget + 1 -- or when
        # the int32 count header cannot fit in the chunk's first row (2 bf16
        # slots per matrix).
        if budget + 1 >= per_rank * padded_local or 2 * per_rank > X[0].shape[-1]:
            capped_packed = False

    # Decide whether selection happens before communication ("local", and only
    # meaningful on the row-sharded path) or after the whole matrix is assembled
    # ("global"). Off the sharded path each rank holds whole matrices, so the two
    # are identical and we keep the cheaper pre-comm selection. "global_capped"
    # groups that fell out of the packed path (degenerate budget, col-sharded,
    # non-2D) take the exact global path.
    global_scope = comm_dim is not None and (
        selection_scope == "global"
        or (selection_scope == "global_capped" and not capped_packed)
    )

    if capped_packed:
        # --- "global_capped": winner-only packed transport ---
        # Accumulate momentum (single owner on this path), all-gather the slice
        # norms, pack the global winners into pooled fixed-size chunks, and
        # come back with full-size shards that are zero except at the applied
        # winner rows -- exactly the global path's masked-post format.
        norms = dion2_pre_accumulate_norms(
            G=to_local(G), M=to_local(M), select_dim=select_dim
        )
        U_ortho = yield from dion2_capped_packed_async(
            M_local=to_local(M),
            norms=norms,
            padded_local=padded_local,
            fraction=fraction,
            global_size=global_dim_size,
            device_rank=device_rank,
            world_size=world_size,
            process_group=process_group,
            per_rank=per_rank,
            budget=budget,
            kw=k * world_size,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        if adjust_lr is None:
            adjusted_lr = lr
        elif adjust_lr == "spectral_norm":
            adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
        elif adjust_lr == "rms_norm":
            adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
        else:
            raise ValueError(f"Unknown adjust_lr: {adjust_lr}")
        dion2_post_orthogonalize_masked(
            X=to_local(X),
            M=to_local(M),
            U=U_ortho,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            ef_decay=ef_decay,
            select_dim=select_dim,
        )
        return

    if global_scope:
        # --- Global selection: send the full shard, select after assembly ---
        # No pre-comm selection; momentum gets the gradient and the whole shard
        # is communicated. ``select_and_orthogonalize_func`` wraps the real
        # Newton-Schulz so that the top-k is taken on each assembled whole matrix
        # (and per-block / per-head, since it rides inside the NS callable).
        U_full = dion2_pre_accumulate(G=to_local(G), M=to_local(M))
        global_comm_dim_size = global_dim_size
        select_ns = _make_select_and_orthogonalize(
            newton_schulz_func, fraction, select_dim, global_select_size=global_dim_size
        )
        U_ortho = yield from megabatch_orthogonalize_async(
            U_full,
            comm_dim=comm_dim,
            device_rank=device_rank,
            world_size=world_size,
            process_group=process_group,
            newton_schulz_func=select_ns,
            flatten=flatten,
            epsilon=epsilon,
            global_comm_dim_size=global_comm_dim_size,
        )
        if adjust_lr is None:
            adjusted_lr = lr
        elif adjust_lr == "spectral_norm":
            adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
        elif adjust_lr == "rms_norm":
            adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
        else:
            raise ValueError(f"Unknown adjust_lr: {adjust_lr}")
        # U_ortho rows are exactly zero except at the globally-selected positions
        # this rank owns. Apply error-feedback decay to those rows of M and the
        # masked weight update, both keyed off the nonzero mask (no indices).
        dion2_post_orthogonalize_masked(
            X=to_local(X),
            M=to_local(M),
            U=U_ortho,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            ef_decay=ef_decay,
            select_dim=select_dim,
        )
        return

    # --- Local selection (opt-in, selection_scope="local"): per-shard top-k, communicate only the
    # selected rows. Under FSDP2 contiguous chunking every rank holds at most
    # ``padded_local = ceil(global / world_size)`` rows, so a uniform
    # ``k = ceil(fraction * padded_local)`` is the per-rank selected count. We
    # select up to ``k`` rows locally (short/empty shards select fewer -- see
    # dion2_pre_orthogonalize) and tell the megabatch to pad every shard to
    # exactly ``k`` via ``local_comm_size=k``, so the alltoall stays uniform
    # while comm and Newton-Schulz both shrink by ``fraction``. This holds for
    # uneven divisions too (the remainder/empty ranks just contribute zero-padded
    # rows), so there is no even-division special case. ``global_comm_dim_size``
    # keeps its true meaning (the unsharded size).
    if comm_dim is not None:
        padded_local = (global_dim_size + world_size - 1) // world_size
        k = max(1, int(math.ceil(fraction * padded_local)))
    else:
        k = None
    global_comm_dim_size = global_dim_size

    # Pre-orthogonalize: momentum update + submatrix selection
    U_selected, indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=ef_decay,
        select_dim=select_dim,
        k_override=k,
    )

    # Orthogonalize via shared megabatch communication
    U_ortho = yield from megabatch_orthogonalize_async(
        U_selected,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=flatten,
        epsilon=epsilon,
        global_comm_dim_size=global_comm_dim_size,
        local_comm_size=k,
    )

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr: {adjust_lr}")

    # Post-orthogonalize: apply update
    if triton_post_ortho:
        from .dion2_triton import dion2_post_orthogonalize_triton

        dion2_post_orthogonalize_triton(
            X=to_local(X),
            U=U_ortho,
            indices=indices_list,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            select_dim=select_dim,
        )
    else:
        dion2_post_orthogonalize(
            X=to_local(X),
            U=U_ortho,
            indices=indices_list,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            select_dim=select_dim,
        )


# Workaround for a torch.compile bug in PyTorch ≤2.11's inductor backend:
# the post-fusion loop reordering pass crashes when ForeachKernelSchedulerNode
# appears inside a FusedSchedulerNode.  Only triggered by recompilation across
# different tensor dimensionalities (e.g. 2D then 3D).
# https://github.com/pytorch/pytorch/issues/176591
# TODO: remove this decorator when pytorch/pytorch#176591 is fixed.
_inductor_workaround = (
    torch._inductor.config.patch(loop_ordering_after_fusion=False)
    if torch.__version__ < "2.13"
    else lambda fn: fn
)


@_inductor_workaround
@torch.compile(fullgraph=True)
def dion2_pre_accumulate_norms(
    G: List[Tensor], M: List[Tensor], select_dim: int
) -> Tensor:
    """
    Phase A of the "global_capped" scope: update momentum with the gradient and
    return the stacked slice L1 norms ``[N, local_size]`` (fp32). The caller
    all-gathers these tiny norms across ranks (an eager collective that cannot
    live inside this compiled graph; see ``dion2_capped_select_async``). This
    is the ONLY momentum accumulation on the capped path -- the packed
    transport never calls ``dion2_pre_orthogonalize``, so the gathered norms
    always reflect the post-accumulation momentum.
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]
    torch._foreach_add_(M, G)
    norm_dim = -1 if select_dim == -2 else -2
    if M[0].size(select_dim) == 0:
        return torch.zeros((len(M), 0), dtype=torch.float32, device=M[0].device)
    return torch.stack(M, dim=0).norm(p=1, dim=norm_dim).float()


# Deferral instrumentation for the "global_capped" packed path. RANK-LOCAL
# semantics: cleared once per optimizer step (at ortho-task creation) and
# ACCUMULATED across that step's packed megabatches (shape groups), as GPU
# scalar tensors -- reading them costs no sync until the consumer calls
# .item(). Empty dict => no packed group ran this step (do not reuse stale
# values). "winner_rows" counts THIS rank's winners and can legitimately be
# zero; aggregate across ranks before forming a deferral ratio.
CAPPED_STATS: dict = {}


def dion2_capped_select_async(
    norms: Tensor,  # [N, local_size] fp32 from dion2_pre_accumulate_norms
    padded_local: int,
    fraction: float,
    global_size: int,
    device_rank: int,
    world_size: int,
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, Tuple[Tensor, Tensor]]:
    """
    "global_capped" winner selection: all-gather the slice L1 norms (one fp32
    per slice -- ~KBs vs MBs for the rows) and return, for this rank,

      - ``winner`` bool ``[N, local_size]``: membership in the exact global
        top-``k_total`` set, ``k_total = ceil(fraction * global_size)``. The
        set is EXACT-COUNT: a stable descending argsort over the gathered
        norms breaks ties deterministically by flat (rank, row) position, so
        ties cannot inflate the winner set past the budget (the receiver-side
        buffer bound in the packed transport relies on this).
      - ``thresh`` ``[N, 1]``: the k_total-th largest norm per matrix, used to
        scale-normalize cross-matrix deferral priorities in the pack step.

    ``k_total <= global_size`` always, so the -1 shard padding can never win.
    Rank consistency: every rank sorts the same bit-identical gathered tensor;
    the receiver clip in ``dion2_capped_assemble`` guards the (pathological)
    divergent case. Async generator in the megabatch style: yields at the
    collective, returns the pair.
    """
    N, local_size = norms.shape
    if local_size > padded_local:
        # F.pad with a negative amount would silently TRIM norms; fail loudly
        # instead (mirrors the analogous guard in megabatch_orthogonalize_async).
        raise ValueError(
            f"Local norm count {local_size} exceeds padded size {padded_local}. "
            "This should not happen with FSDP2 contiguous sharding."
        )
    if local_size < padded_local:
        norms_pad = torch.nn.functional.pad(
            norms, (0, padded_local - local_size), value=-1.0
        )
    else:
        norms_pad = norms
    gathered = torch.empty(
        world_size * N * padded_local, dtype=torch.float32, device=norms.device
    )
    work = dist.all_gather_into_tensor(
        gathered, norms_pad.reshape(-1).contiguous(),
        group=process_group, async_op=True,
    )
    yield
    work.wait()

    # [world, N, padded_local] -> flat per-matrix [N, world*padded_local] with
    # rank-major flat position (r * padded_local + row).
    flat = gathered.view(world_size, N, padded_local).permute(1, 0, 2).reshape(N, -1)
    k_total = min(max(1, int(math.ceil(fraction * global_size))), global_size)
    order = torch.argsort(flat, dim=-1, descending=True, stable=True)
    winner_flat = torch.zeros_like(flat, dtype=torch.bool)
    winner_flat.scatter_(1, order[:, :k_total], True)
    thresh = flat.gather(1, order[:, k_total - 1 : k_total])
    own = winner_flat.view(N, world_size, padded_local)[:, device_rank, :local_size]
    return own.contiguous(), thresh


@_inductor_workaround
@torch.compile(fullgraph=True)
def dion2_capped_pack(
    M: List[Tensor],  # N tensors [local, cols] (momentum, already accumulated)
    norms: Tensor,  # [N, local] fp32
    winner: Tensor,  # [N, local] bool, from dion2_capped_select_async
    thresh: Tensor,  # [N, 1] fp32
    world_size: int,
    per_rank: int,
    budget: int,  # pooled row budget B per (rank -> dest) chunk
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sender side of the packed "global_capped" transport. Matrices are assigned
    to destination ranks in blocks of ``per_rank`` (megabatch order); for each
    destination this rank packs its winner rows of that destination's matrices
    back-to-back into a fixed ``budget``-row chunk -- slots are POOLED across
    the per_rank matrices, so one matrix's overflow uses another's spare room.

    If a destination's total winners exceed ``budget``, the lowest-priority
    rows are deferred (error feedback: they keep full momentum and win later).
    Priority is the SCALE-NORMALIZED ``norm / thresh`` -- raw norms are not
    comparable across matrices, and a large-scale matrix must not evict a
    small-scale matrix's winners from the shared chunk.

    Returns ``(payload [world, budget, cols] bf16, counts [world, per_rank]
    int32, src_index [world, budget] long, deferred_rows scalar)``. Empty
    slots gather a dedicated zero row (never read by the receiver, which
    parses only ``counts`` rows per segment). ``src_index`` maps each slot to
    its flat local row (sentinel = the zero row) and is kept by the sender to
    scatter the orthogonalized rows home; the packing is stable and
    position-preserving -- the reverse path relies on it.
    """
    N = len(M)
    local = M[0].size(-2)
    cols = M[0].size(-1)
    n_pad = world_size * per_rank
    device = M[0].device

    M_stacked = torch.stack(M, dim=0)  # [N, local, cols]
    if n_pad > N:
        # Virtual matrices padding the group to world_size * per_rank: no
        # winners, zero rows.
        M_stacked = torch.cat(
            [M_stacked, torch.zeros(n_pad - N, local, cols, dtype=M_stacked.dtype, device=device)]
        )
        winner = torch.cat(
            [winner, torch.zeros(n_pad - N, local, dtype=torch.bool, device=device)]
        )
        norms = torch.cat(
            [norms, torch.zeros(n_pad - N, local, dtype=norms.dtype, device=device)]
        )
        thresh = torch.cat(
            [thresh, torch.ones(n_pad - N, 1, dtype=thresh.dtype, device=device)]
        )

    prio = norms / thresh.clamp_min(1e-12)
    score = torch.where(winner, prio, torch.full_like(prio, float("-inf")))
    score_d = score.view(world_size, per_rank * local)
    b_eff = min(budget, per_rank * local)
    # Stable argsort (not topk): on tied normalized priorities the flat
    # (matrix, row) order is the deterministic secondary key, so WHICH row
    # defers on overflow is reproducible across runs/devices -- matching the
    # stable tie contract of the winner selection itself.
    order = torch.argsort(score_d, dim=-1, descending=True, stable=True)
    idxs = order[:, :b_eff]
    vals = torch.gather(score_d, 1, idxs)
    kept_flat = torch.zeros_like(score_d, dtype=torch.bool)
    kept_flat.scatter_(1, idxs, vals.isfinite())  # winners only; spare slots stay empty
    kept = kept_flat.view(world_size, per_rank, local)

    counts = kept.sum(dim=-1)  # [world, per_rank]
    seg_border = counts.cumsum(dim=1) - counts  # exclusive, within chunk
    # Slot of each kept row: matrix segment border + within-matrix ordinal
    # (row order preserved -- stable packing).
    ordinal = kept.cumsum(dim=-1) - 1
    slot = seg_border.unsqueeze(-1) + ordinal
    slot_safe = torch.where(kept, slot, torch.full_like(slot, budget))  # trash col

    fid = torch.arange(n_pad * local, device=device).view(world_size, per_rank, local)
    sentinel = n_pad * local  # index of the appended zero row
    src_index = torch.full(
        (world_size, budget + 1), sentinel, dtype=torch.long, device=device
    )
    src_index.scatter_(
        1, slot_safe.reshape(world_size, -1), fid.reshape(world_size, -1)
    )
    src_index = src_index[:, :budget]

    M_flat = torch.cat(
        [M_stacked.reshape(n_pad * local, cols),
         torch.zeros(1, cols, dtype=M_stacked.dtype, device=device)]
    )
    payload = M_flat[src_index].to(torch.bfloat16)  # [world, budget, cols]
    deferred = winner.sum() - kept.sum()
    return payload, counts.to(torch.int32), src_index, deferred


@torch.compile(fullgraph=True)
def dion2_capped_assemble(
    payload: Tensor,  # [world, budget, cols] bf16 (received chunks, no header)
    counts: Tensor,  # [world, per_rank] int32 (parsed headers)
    kw: int,  # static NS row count per matrix (k * world_size)
) -> Tuple[Tensor, Tensor]:
    """
    Receiver side: assemble each of this rank's ``per_rank`` matrices from the
    variable-length segments of all senders' chunks into a static, zero-padded
    ``[per_rank, kw, cols]`` Newton-Schulz input. Rows land in (sender rank,
    within-segment) order; NS is permutation-equivariant and the reverse path
    inverts the same mapping, so the order is only required to be stable.

    Arrivals beyond ``kw`` for a matrix (impossible with exact-count winners;
    possible only under a divergent sender) are clipped to a trash slot rather
    than corrupting the buffer. Returns ``(ns_input, ns_index)`` where
    ``ns_index [per_rank, kw]`` maps NS rows back to flat payload positions
    (sentinel = zero row) for ``dion2_capped_repack``.
    """
    world_size, budget, cols = payload.shape
    per_rank = counts.size(1)
    device = payload.device
    cnt = counts.to(torch.long)

    seg_border = cnt.cumsum(dim=1) - cnt  # [world, per_rank] within sender chunk
    dst_border = cnt.cumsum(dim=0) - cnt  # [world, per_rank] within NS rows
    max_seg = min(budget, kw)
    j = torch.arange(max_seg, device=device)
    valid = j < cnt.unsqueeze(-1)  # [world, per_rank, max_seg]
    src_pos = (
        torch.arange(world_size, device=device).view(-1, 1, 1) * budget
        + seg_border.unsqueeze(-1)
        + j
    )
    src_pos = torch.where(valid, src_pos, torch.full_like(src_pos, world_size * budget))
    dst_pos = dst_border.unsqueeze(-1) + j
    dst_pos = torch.where(
        valid & (dst_pos < kw), dst_pos, torch.full_like(dst_pos, kw)  # receiver clip
    )

    ns_index = torch.full(
        (per_rank, kw + 1), world_size * budget, dtype=torch.long, device=device
    )
    ns_index.scatter_(
        1,
        dst_pos.permute(1, 0, 2).reshape(per_rank, -1),
        src_pos.permute(1, 0, 2).reshape(per_rank, -1),
    )
    ns_index = ns_index[:, :kw]

    pay_flat = torch.cat(
        [payload.reshape(world_size * budget, cols),
         torch.zeros(1, cols, dtype=payload.dtype, device=device)]
    )
    ns_input = pay_flat[ns_index]  # [per_rank, kw, cols], zero-padded tail
    return ns_input, ns_index


@torch.compile(fullgraph=True)
def dion2_capped_repack(
    ns_out: Tensor,  # [per_rank, kw, cols] orthogonalized
    ns_index: Tensor,  # [per_rank, kw] from dion2_capped_assemble
    world_size: int,
    budget: int,
) -> Tensor:
    """Inverse of assemble: scatter NS rows back to their payload positions.
    Padding/sentinel rows collapse onto a trash row; payload slots that carried
    no real row stay exactly zero (so the sender scatters zeros = no update)."""
    per_rank, kw, cols = ns_out.shape
    out = torch.zeros(
        world_size * budget + 1, cols, dtype=ns_out.dtype, device=ns_out.device
    )
    out.scatter_(0, ns_index.reshape(-1, 1).expand(-1, cols), ns_out.reshape(-1, cols))
    return out[: world_size * budget].view(world_size, budget, cols)


@torch.compile(fullgraph=True)
def dion2_capped_unpack(
    recv: Tensor,  # [world, budget, cols] orthogonalized rows, sender's layout
    src_index: Tensor,  # [world, budget] from dion2_capped_pack
    n_params: int,
    local: int,
    total_rows: int,  # n_pad * local (sentinel index)
) -> List[Tensor]:
    """Sender-side finish: scatter the returned rows to their home (matrix,
    row) positions. Output is a list of full-size ``[local, cols]`` shards that
    are exactly zero except at this rank's applied winner rows -- the format
    ``dion2_post_orthogonalize_masked`` consumes (same as the global scope)."""
    world_size, budget, cols = recv.shape
    U_flat = torch.zeros(
        total_rows + 1, cols, dtype=recv.dtype, device=recv.device
    )
    U_flat.scatter_(0, src_index.reshape(-1, 1).expand(-1, cols), recv.reshape(-1, cols))
    return list(U_flat[: n_params * local].view(n_params, local, cols).unbind(0))


def dion2_capped_packed_async(
    M_local: List[Tensor],
    norms: Tensor,
    padded_local: int,
    fraction: float,
    global_size: int,
    device_rank: int,
    world_size: int,
    process_group: Optional[ProcessGroup],
    per_rank: int,
    budget: int,
    kw: int,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Generator[None, None, List[Tensor]]:
    """
    "global_capped" packed transport: winner selection (norms-only all-gather)
    -> pooled pack -> fixed-size a2a with an int32 count header bit-cast into
    each chunk's first row -> assemble + Newton-Schulz -> repack -> reverse a2a
    -> scatter home. All collective sizes are construction-time constants; the
    data-dependence lives entirely in GPU-side gathers/scatters (no host
    syncs). The header makes the sender's packing plan the single source of
    truth -- the receiver never recomputes winner counts.
    """
    winner, thresh = yield from dion2_capped_select_async(
        norms, padded_local, fraction, global_size,
        device_rank, world_size, process_group,
    )
    payload, counts, src_index, deferred = dion2_capped_pack(
        M_local, norms, winner, thresh, world_size, per_rank, budget
    )
    CAPPED_STATS["deferred_rows"] = CAPPED_STATS.get("deferred_rows", 0) + deferred
    CAPPED_STATS["winner_rows"] = CAPPED_STATS.get("winner_rows", 0) + winner.sum()

    # Header row: int32 counts bit-cast into bf16 (2 bf16 per int32). Counts
    # are NOT representable as bf16 *values* (exact only to 256); the bit-cast
    # round-trips exactly.
    cols = payload.size(-1)
    header = torch.zeros(
        world_size, 1, cols, dtype=torch.bfloat16, device=payload.device
    )
    header[:, 0, : per_rank * 2] = counts.contiguous().view(torch.bfloat16)
    chunks = torch.cat([header, payload], dim=1)  # [world, 1 + budget, cols]

    send = [c.contiguous() for c in chunks.unbind(0)]
    recv = [torch.empty_like(c) for c in send]
    work = dist.all_to_all(recv, send, group=process_group, async_op=True)
    yield
    work.wait()

    recv_t = torch.stack(recv)
    counts_recv = recv_t[:, 0, : per_rank * 2].contiguous().view(torch.int32)
    ns_input, ns_index = dion2_capped_assemble(
        recv_t[:, 1:].contiguous(), counts_recv, kw
    )
    ns_out = muon_update_newton_schulz(
        ns_input, newton_schulz_func=newton_schulz_func,
        flatten=flatten, epsilon=epsilon,
    )
    back = dion2_capped_repack(ns_out, ns_index, world_size, budget)

    send2 = [c.contiguous() for c in back.unbind(0)]
    recv2 = [torch.empty_like(c) for c in send2]
    work = dist.all_to_all(recv2, send2, group=process_group, async_op=True)
    yield
    work.wait()

    local = M_local[0].size(-2)
    n_pad = world_size * per_rank
    return dion2_capped_unpack(
        torch.stack(recv2), src_index, len(M_local), local, n_pad * local
    )


@_inductor_workaround
@torch.compile(fullgraph=True)
def dion2_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    fraction: Tensor,
    ef_decay: Tensor,
    select_dim: int,
    k_override: Optional[int] = None,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    More specifically, it does the following steps:
        - updates the momentum with gradient
        - computes the top-k indices (according to L1 norm) to determine submatrices
        - (other norms can be used such as L2 norm)
        - does in-place error-feedback decay on the selected submatrices
        - output submatrices and indices
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().

    ``k_override`` forces the number of selected slices (used by the row-sharded
    "local" path so every rank selects the same count, derived from the global
    size, instead of ``ceil(fraction * local_size)``). A rank whose local shard
    is shorter than ``k_override`` selects all of its rows; ``topk`` is clamped
    to the available count. The pad up to ``k_override`` is not done here -- the
    downstream megabatch pads U to ``local_comm_size=k_override`` for a uniform
    alltoall, and indices stay at the real selected count.

    (The "global_capped" scope does not pass through here at all -- it uses
    the packed transport, ``dion2_capped_packed_async``, and the masked post.)
    """
    dtype = M[0].dtype

    # norm_dim is the dimension we compute norm over
    # select_dim is the dimension we select submatrix from
    num_select = M[0].size(select_dim)
    norm_dim = -1 if select_dim == -2 else -2
    # k is the requested selected count; k_topk is what topk can actually take
    # from this shard (clamped to its real rows). When the shard is shorter than
    # k_override -- possible on the last/remainder rank under uneven FSDP2
    # chunking -- we select all its rows here. U_selected then carries only the
    # real k_topk rows; the pad up to k happens downstream in
    # megabatch_orthogonalize_async (local_comm_size=k) purely so the alltoall
    # sees a uniform per-rank size. indices deliberately stay at k_topk: the
    # megabatch narrows its result back to k_topk before it is scattered, so the
    # padded rows never reach post_orthogonalize.
    if k_override is not None:
        k = k_override
    else:
        k = max(1, int(math.ceil(fraction * num_select)))
    k_topk = min(k, num_select)

    # Update momentum: M = M + G
    G = [g.to(dtype=dtype) for g in G]
    torch._foreach_add_(M, G)

    # Empty local shard along select_dim: FSDP2 contiguous chunking leaves this
    # rank with a size-0 shard when the param's sharded dim is smaller than
    # world_size (or doesn't divide evenly to fill all ranks). There is nothing
    # to select here, and topk(k>=1) over a size-0 dimension raises "k not in
    # range for dimension". Short-circuit with empty submatrices (downstream
    # megabatch_orthogonalize_async pads these to padded_local_size; the real
    # orthogonalization runs on the gathered global tensor) and empty index
    # tensors (post-orthogonalize scatter_add over an empty index is a no-op on
    # this rank). num_select is a static int at trace time, so this branch is
    # torch.compile-safe.
    if num_select == 0:
        U_selected = [m.to(dtype=torch.bfloat16) for m in M]
        indices_list = [torch.empty(0, dtype=torch.long, device=M[0].device) for _ in M]
        return U_selected, indices_list

    M_stacked = torch.stack(M, dim=0)
    slice_norms = M_stacked.norm(p=1, dim=norm_dim)

    # Batched topk: indices shape (batch_size, k_topk). k_topk <= num_select is
    # guaranteed, so this never raises even on a short remainder shard.
    _, indices = torch.topk(slice_norms, k_topk, dim=-1, sorted=False)

    # Extract the selected rows/columns from each momentum tensor.
    # `indices` has shape (..., k) where k is the number of selected slices.
    # `gather` requires the index tensor to have the same number of dimensions
    # as the source, so we expand the indices to cover the non-selected dimension.
    if select_dim == -2:
        # Selecting rows: expand indices from (..., k) to (..., k, num_cols)
        num_cols = M[0].size(-1)
        indices_expanded = indices.unsqueeze(-1).expand(*indices.shape, num_cols)
        selected_stacked = torch.gather(M_stacked, dim=-2, index=indices_expanded)
    else:
        # Selecting cols: expand indices from (..., k) to (..., num_rows, k)
        num_rows = M[0].size(-2)
        indices_expanded = indices.unsqueeze(-2).expand(
            *indices.shape[:-1], num_rows, indices.shape[-1]
        )
        selected_stacked = torch.gather(M_stacked, dim=-1, index=indices_expanded)

    # Apply error feedback decay to selected slices in the original M tensors.
    # We reuse the already-gathered slices and write them back (scaled) using
    # scatter_, which places values into positions specified by the index
    # tensor. The .to(dtype) guards bf16 momentum: scatter_ requires src dtype
    # to match M exactly, and it keeps a single fp32-multiply-then-round if
    # ef_decay ever becomes a dimensioned tensor (a 0-dim fp32 scalar loses
    # type promotion to bf16, but a dimensioned one would win it).
    indices_list = list(indices.unbind(dim=0))
    ef_src_list = list((selected_stacked * ef_decay).to(dtype).unbind(dim=0))
    for m, idx, ef_src in zip(M, indices_list, ef_src_list):
        if select_dim == -2:
            idx_exp = idx.unsqueeze(-1).expand(*idx.shape, m.size(-1))
        else:
            idx_exp = idx.unsqueeze(-2).expand(*idx.shape[:-1], m.size(-2), idx.shape[-1])
        m.scatter_(dim=select_dim, index=idx_exp, src=ef_src)

    # Convert to bf16 and unstack for communication
    U_selected = list(selected_stacked.to(dtype=torch.bfloat16).unbind(dim=0))

    return U_selected, indices_list


@torch.compile(fullgraph=True)
def dion2_pre_accumulate(G: List[Tensor], M: List[Tensor]) -> List[Tensor]:
    """
    Global-scope pre-orthogonalize: update momentum with the gradient and return
    the whole shard in bf16 for communication. No selection happens here -- the
    top-k is taken after the full matrix is assembled, inside the wrapped
    Newton-Schulz function. Error-feedback decay is deferred to the masked post
    step. Inputs/outputs are regular Tensors, not DTensor.
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]
    torch._foreach_add_(M, G)
    return [m.to(dtype=torch.bfloat16) for m in M]


def _make_select_and_orthogonalize(
    newton_schulz_func: Callable,
    fraction: float,
    select_dim: int,
    global_select_size: Optional[int] = None,
) -> Callable:
    """
    Wrap a Newton-Schulz function so it (1) selects the top-k slices of each
    assembled whole matrix by L1 norm along ``select_dim``, (2) orthogonalizes
    only that submatrix, and (3) scatters the result back into a full-size,
    otherwise-zero tensor. Used by the "global" selection scope: because the
    wrapper is invoked on whole matrices (and per head, since it rides inside
    the per-head split), the selection is exact and invariant to the sharding
    layout. The returned full-size tensor has exactly
    zero rows/cols everywhere except the selected positions, which the masked
    post step keys off.

    ``global_select_size`` is the true unsharded size along ``select_dim``. On
    the row-sharded path the matrix handed to this wrapper is zero-padded to
    ``ceil(global / world_size) * world_size`` rows, which exceeds the true
    global size whenever it is not divisible by ``world_size``. Deriving ``k``
    from ``X.size(select_dim)`` would then select ``ceil(fraction * padded)``
    slices -- more than the true whole-matrix top-k, and a count that depends on
    ``world_size`` -- silently breaking the exact/reproducible-across-world-sizes
    guarantee. So ``k`` is computed from ``global_select_size`` when provided;
    the padded rows are exactly zero and never rank into the top-k, so selecting
    over the padded matrix still picks exactly the real global top-k. Falls back
    to ``X.size(select_dim)`` when not given (e.g. an already-whole matrix).
    """

    def _select_ns(X: Tensor, epsilon=None) -> Tensor:
        num_select = (
            X.size(select_dim) if global_select_size is None else global_select_size
        )
        norm_dim = -1 if select_dim == -2 else -2
        # k derives from the true global size but never exceeds the (padded)
        # matrix handed in, so topk is always valid.
        k = min(max(1, int(math.ceil(fraction * num_select))), X.size(select_dim))
        slice_norms = X.norm(p=1, dim=norm_dim)
        _, indices = torch.topk(slice_norms, k, dim=-1, sorted=False)
        if select_dim == -2:
            idx_exp = indices.unsqueeze(-1).expand(*indices.shape, X.size(-1))
        else:
            idx_exp = indices.unsqueeze(-2).expand(
                *indices.shape[:-1], X.size(-2), indices.shape[-1]
            )
        sub = torch.gather(X, dim=select_dim, index=idx_exp)
        ortho = newton_schulz_func(sub, epsilon=epsilon)
        out = torch.zeros_like(X)
        out.scatter_(dim=select_dim, index=idx_exp, src=ortho.to(out.dtype))
        return out

    return _select_ns


@torch.compile(fullgraph=True)
def dion2_post_orthogonalize_masked(
    X: List[Tensor],
    M: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    ef_decay: Tensor,
    select_dim: int,
):
    """
    Global-scope post-orthogonalize. ``U`` holds full-size orthogonalized shards
    that are exactly zero except at the globally-selected slices this rank owns.
    Derive the selected mask from the nonzero slices (orthonormal rows/cols have
    unit norm; non-selected are exactly 0), then apply error-feedback decay to
    the selected slices of ``M`` and the masked weight update -- no indices
    needed. Inputs/outputs are regular Tensors, not DTensor.
    """
    norm_dim = -1 if select_dim == -2 else -2
    dtype = X[0].dtype

    # Weight decay on all weights (matches the unmasked dion2_post_orthogonalize)
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    one = torch.ones((), dtype=M[0].dtype, device=M[0].device)
    ef = ef_decay.to(M[0].dtype)
    neg_lr = -adjusted_lr
    for x, m, u in zip(X, M, U):
        # Boolean mask over the selected dim: True where the orthogonalized
        # slice is nonzero (i.e. a selected slice). Keepdim so it broadcasts
        # back over the full slice for the in-place updates below.
        sel = u.to(torch.float32).abs().sum(dim=norm_dim, keepdim=True) > 0
        # Error-feedback decay on the selected slices of momentum: multiply
        # selected slices by ef_decay, leave the rest unchanged.
        m.mul_(torch.where(sel, ef, one))
        # Masked weight update X += -adjusted_lr * U. U is exactly zero off the
        # selected slices, so a plain add only touches them (equivalent to the
        # scatter_add over indices used by the local path).
        x.add_((neg_lr * u).to(dtype))


# NOTE: if this function starts failing with an InductorError on recompilation
# across tensor ranks, apply the same _inductor_workaround used on
# dion2_pre_orthogonalize above.  See pytorch/pytorch#176591.
@torch.compile(fullgraph=True)
def dion2_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    indices: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    select_dim: int,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Convert U to match parameter dtype
    dtype = X[0].dtype
    U = [u.to(dtype=dtype) for u in U]
    # Apply weight update
    neg_lr = -adjusted_lr
    U_scaled = [neg_lr * u for u in U]
    # Apply the orthogonalized update to only the selected rows/columns.
    # scatter_add_ accumulates values into positions specified by the index tensor:
    #   x[..., idx_exp[..., i, j], j] += u_scaled[..., i, j]  (for select_dim == -2)
    # where i ranges over the k selected rows and j over all columns.
    for x, u_scaled, idx in zip(X, U_scaled, indices):
        if select_dim == -2:
            idx_exp = idx.unsqueeze(-1).expand_as(u_scaled)
        else:
            idx_exp = idx.unsqueeze(-2).expand_as(u_scaled)
        x.scatter_add_(dim=select_dim, index=idx_exp, src=u_scaled)


@torch.compile(fullgraph=True)
def dion2_post_orthogonalize_fused(
    X: List[Tensor],
    U: List[Tensor],
    indices: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    select_dim: int,
):
    """
    Single-rounding weight decay + weight update after orthogonalization.

    Computes the new value of the selected rows/columns as
    ``(1 - base_lr*weight_decay)*x - adjusted_lr*u`` in float32 and writes it
    once, matching the single-rounding numerics of the fused Triton kernel
    (dion2_post_orthogonalize_triton). Unselected entries get the weight decay
    in place, also a single rounding. This differs from dion2_post_orthogonalize,
    which writes the weight-decayed weight and then accumulates the update in a
    second pass, rounding the selected slices twice.

    Only the selected slices are gathered into float32, so the extra work over
    the in-place weight decay is small. Uses only ``__torch_dispatch__``-routed
    ops (no raw data_ptr writes), so it is safe for traceable wrapper subclasses
    such as the MXFP8 training weight wrapper, for which the Triton kernel cannot
    be used. Inputs should be lists of regular Tensor, not DTensor.
    """
    a = 1 - base_lr * weight_decay
    neg_lr = -adjusted_lr
    for x, u, idx in zip(X, U, indices):
        if select_dim == -2:
            idx_exp = idx.unsqueeze(-1).expand_as(u)
        else:
            idx_exp = idx.unsqueeze(-2).expand_as(u)
        # Fused single-rounding value for the selected slices, computed in float32
        # from the original weight before any in-place modification.
        x_sel = a * torch.gather(x, select_dim, idx_exp).float() + neg_lr * u.float()
        # Weight decay for the unselected entries (single rounding); the selected
        # slices are overwritten with the fused value below.
        x.mul_(a)
        x.scatter_(dim=select_dim, index=idx_exp, src=x_sel.to(x.dtype))


# A helper function to print selection choice for each matrix
# It only prints once `verbose` is set True
_printed_configs: set = set()


def _print_selection_choice(
    shape: torch.Size,
    shard_dim: Optional[int],
    select_dim: int,
    ndim: int,
):
    config_key = (tuple(shape), shard_dim, select_dim)
    if config_key not in _printed_configs:
        _printed_configs.add(config_key)

        num_rows, num_cols = shape[-2:]
        select_info = "rows" if select_dim == -2 else "columns"
        norm_info = "row norms" if select_dim == -2 else "col norms"

        if shard_dim is None:
            mode = "DDP/Single-GPU"
            shorter = "rows" if num_rows <= num_cols else "cols"
            reason = f"shorter dim = {shorter} ({min(num_rows, num_cols)})"
        else:
            # Normalize shard_dim for display
            normalized = shard_dim if shard_dim < 0 else shard_dim - ndim
            if normalized == -2:
                mode = "FSDP"
                reason = f"row-sharded (shard_dim={shard_dim}→-2)"
            elif normalized == -1:
                mode = "FSDP"
                reason = f"col-sharded (shard_dim={shard_dim}→-1)"
            else:
                mode = "FSDP batch-sharded"
                shorter = "rows" if num_rows <= num_cols else "cols"
                reason = f"shard_dim={shard_dim} (batch), shorter = {shorter}"

        print(
            f"[Dion2] Shape {tuple(shape)}: {mode}, {reason} → "
            f"select top-α {select_info} by {norm_info}"
        )
