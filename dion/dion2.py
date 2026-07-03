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
            in when comm-bound at large scale. "global_capped": global selection
            at local comm cost -- all-gather the row norms only (~KBs), compute
            the same global top-``k*world_size`` threshold everywhere, and send
            each rank's globally-selected rows through the fixed ``k``-slot
            pipes; a rank owning more winners than slots defers the overflow
            (error feedback re-selects them on a later step), one owning fewer
            sends zeros in the spare slots so only winners produce updates.
            No-op off the row-sharded path.

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
        capped_slack: float = 1.0,
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if selection_scope not in ("local", "global", "global_capped", "global_exact"):
            raise ValueError(
                f"selection_scope must be 'local', 'global', 'global_capped', or "
                f"'global_exact', got {selection_scope!r}"
            )
        if capped_slack < 1.0:
            raise ValueError(
                f"capped_slack must be >= 1.0 (slots = ceil(slack*k) >= the winner "
                f"budget k), got {capped_slack}"
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
            capped_slack=float(capped_slack),
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
                capped_slack=group["capped_slack"],
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
    capped_slack: float = 1.0,  # "global_capped" only: comm slots = ceil(slack*k); >1 defers fewer winners for a slack*-comm premium
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
    - ``"global_capped"``: global selection at local comm cost. The slice L1
      norms are all-gathered (one fp32 per slice, ~KBs), every rank computes the
      same global top-``k*world_size`` threshold, and each rank sends its
      globally-selected slices ("winners") through the same fixed ``k``-slot
      pipes as "local". Overflow winners (a rank owning more winners than slots)
      are deferred -- they skip error-feedback decay, so their momentum keeps
      accumulating and they win a slot on a later step. Spare slots travel as
      zeros: non-winners are never applied and never decayed, which is what
      distinguishes this scope from "local" (a per-rank top-``k`` fill would be
      trajectory-identical to "local", since a rank's winners are always its
      top-norm slices). Note the selection budget is top-``k*world_size``, not
      top-``ceil(fraction*global)``: the two coincide when ``world_size``
      divides the sharded dim and the budget is slightly larger otherwise.

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

    # Decide whether selection happens before communication ("local", and only
    # meaningful on the row-sharded path) or after the whole matrix is assembled
    # ("global"). Off the sharded path each rank holds whole matrices, so the two
    # are identical and we keep the cheaper pre-comm selection.
    global_scope = selection_scope == "global" and comm_dim is not None

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
        # "global_capped" slack: allocate more comm slots (k_comm) than the
        # winner budget k so fewer winners overflow (are deferred). The winner
        # THRESHOLD below still uses k (top-k*world_size), so the selected set is
        # unchanged -- only per-rank capacity grows; the extra slots carry
        # zeroed non-winners (winners-only masking) or, once a rank's winners
        # exceed k, more of its real winners. Clamped to padded_local (the full
        # shard, == "global" comm cost). capped_slack=1.0 (default) => k_comm==k,
        # so "local"/"global_capped" default behavior is bit-identical.
        if selection_scope == "global_capped":
            k_comm = min(padded_local, max(k, int(math.ceil(capped_slack * k))))
        else:
            k_comm = k
    else:
        k = None
        k_comm = None
    global_comm_dim_size = global_dim_size

    # "global_capped"/"global_exact": accumulate momentum + all-gather the slice
    # norms (one fp32 per slice, ~KBs), and hand the global top-(k*world_size)
    # threshold to pre_orthogonalize, which applies the capacity rule (winners-
    # only updates, zeros in spare slots). "global_exact" additionally sets the
    # comm-slot count k_comm to the max per-(rank, matrix) winner count this
    # step, so EVERY winner fits with no deferral -- exact global selection at
    # (near-)local comm cost, one scalar host sync for the dynamic k_comm.
    thresh = None
    if (
        selection_scope in ("global_capped", "global_exact")
        and comm_dim is not None
        and process_group is not None
    ):
        norms = dion2_pre_accumulate_norms(
            G=to_local(G), M=to_local(M), select_dim=select_dim
        )
        thresh, kmax = yield from dion2_capped_threshold_async(
            norms, padded_local, k, world_size, process_group
        )
        if selection_scope == "global_exact":
            kmax_i = max(1, int(kmax.item()))
            # Quantize the dynamic slot count into a few buckets: pre_orthogonalize
            # is torch.compile'd and SPECIALIZES on k_comm (an int arg), so a
            # per-step-varying value thrashes the compile cache. Round UP so
            # k_comm >= kmax -> every winner still fits -> selection stays exact,
            # at <= q extra zero-slots of comm (q ~ padded_local/16).
            q = max(1, padded_local // 16)
            k_comm = min(padded_local, ((kmax_i + q - 1) // q) * q)

    # Pre-orthogonalize: momentum update + submatrix selection
    U_selected, indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=ef_decay,
        select_dim=select_dim,
        k_override=k_comm,
        thresh=thresh,
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
        local_comm_size=k_comm,
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
    live inside this compiled graph) and feeds the resulting global threshold
    back into ``dion2_pre_orthogonalize`` via ``thresh``. This function is the
    single owner of momentum accumulation on the capped path: passing ``thresh``
    tells ``dion2_pre_orthogonalize`` NOT to re-accumulate the gradient, so the
    gathered norms always reflect the post-accumulation momentum.
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]
    torch._foreach_add_(M, G)
    norm_dim = -1 if select_dim == -2 else -2
    if M[0].size(select_dim) == 0:
        return torch.zeros((len(M), 0), dtype=torch.float32, device=M[0].device)
    return torch.stack(M, dim=0).norm(p=1, dim=norm_dim).float()


def dion2_capped_threshold_async(
    norms: Tensor,  # [N, local_size] fp32 from dion2_pre_accumulate_norms
    padded_local: int,
    k: int,
    world_size: int,
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, Tensor]:
    """
    "global_capped" selection threshold: all-gather the slice L1 norms (one fp32
    per slice -- ~KBs vs MBs for the rows) and return the per-matrix global
    top-``k*world_size`` threshold ``[N, 1]``, computed identically on every
    rank. ``dion2_pre_orthogonalize`` compares its local norms against this
    value to apply the capacity rule (winners-only updates -- see its
    docstring).

    Rank consistency is by construction: every rank computes the threshold from
    the same bit-identical gathered tensor, and the threshold is a *value* (the
    ``k_total``-th largest norm), so it does not depend on ``topk`` tie-breaking
    order. Async generator in the megabatch style: yields at the collective,
    returns the threshold.
    """
    N, local_size = norms.shape
    # Pad short/empty shards to the rank-uniform padded_local. L1 norms are
    # nonnegative, so -1 padding can never enter the winner set unless
    # k*world_size exceeds the real row count (tiny-matrix degenerate case,
    # where "everything is a winner" is the correct answer anyway).
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

    # [world, N, padded_local] -> per-matrix global threshold = k_total-th value
    all_norms = gathered.view(world_size, N, padded_local)
    k_total = min(k * world_size, world_size * padded_local)
    vals, _ = torch.topk(
        all_norms.permute(1, 0, 2).reshape(N, -1), k_total, dim=-1
    )
    thresh = vals[:, -1:]
    # Also return the max per-(rank, matrix) winner count -- the "global_exact"
    # scope uses this as its comm-slot count so EVERY winner fits with no
    # deferral (exact global top-(k*world) selection). Padding rows are -1 and
    # thresh >= 0 except in the degenerate everything-wins case (thresh == -1,
    # where counting the pads correctly yields padded_local = the full shard).
    # A device scalar -- the caller .item()s it only on the exact path.
    kmax = (all_norms >= thresh.view(1, N, 1)).sum(dim=-1).max()
    return thresh, kmax


@_inductor_workaround
@torch.compile(fullgraph=True)
def dion2_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    fraction: Tensor,
    ef_decay: Tensor,
    select_dim: int,
    k_override: Optional[int] = None,
    thresh: Optional[Tensor] = None,
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

    ``thresh`` (the "global_capped" scope) is the per-matrix global
    top-``k*world_size`` norm threshold ``[N, 1]`` from
    ``dion2_capped_threshold_async``. Passing it means the momentum was ALREADY
    accumulated by ``dion2_pre_accumulate_norms`` (the single accumulation
    owner on the capped path -- the gathered norms must reflect the
    post-accumulation momentum), so the ``M += G`` here is skipped. The top-k
    gather itself is unchanged: a rank's winners (slices at/above ``thresh``)
    are by construction its top-norm slices, so they already lead the plain
    norm-order selection. What changes is what happens to the non-winner
    "filler" slices occupying spare slots: they are masked to zero in the
    communicated ``U_selected`` (zero slices pass through Newton-Schulz as
    exact zeros, so they produce no weight update) and they skip error-feedback
    decay, letting their momentum accumulate until they cross the global
    threshold on a later step. Without this masking, capped selection would be
    trajectory-identical to "local" scope.
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

    # Update momentum: M = M + G. Skipped when `thresh` is given ("global_
    # capped"): dion2_pre_accumulate_norms is the single accumulation owner on
    # that path, so the gathered norms reflect the post-accumulation momentum.
    if thresh is None:
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

    # "global_capped" capacity rule: only slices at/above the global threshold
    # ("winners") are applied. Winners lead the norm-order selection above by
    # construction, so `indices` needs no change; non-winner "fillers" in spare
    # slots are masked to zero for communication (zero slices pass through
    # Newton-Schulz as exact zeros -> no weight update) and keep their momentum
    # un-decayed so it accumulates until they win. `>=` marks hard ties at the
    # threshold as winners, which can inflate the winner set past k*world_size
    # -- measure-zero for continuous fp32 norms and bounded by the per-rank k
    # cap, so benign.
    if thresh is not None:
        sel_norms = torch.gather(slice_norms, dim=-1, index=indices)  # [N, k_topk]
        winner = sel_norms >= thresh
        winner_exp = winner.unsqueeze(-1) if select_dim == -2 else winner.unsqueeze(-2)
        ef_factor = torch.where(winner_exp, ef_decay, torch.ones_like(ef_decay))
        U_stacked = selected_stacked * winner_exp.to(selected_stacked.dtype)
    else:
        ef_factor = ef_decay
        U_stacked = selected_stacked

    # Apply error feedback decay to selected slices in the original M tensors
    # (capped scope: winners only). We reuse the already-gathered slices and
    # write them back (scaled) using scatter_, which places values into
    # positions specified by the index tensor.
    indices_list = list(indices.unbind(dim=0))
    # Cast the error-feedback write-back to M's dtype: ef_factor derives from the
    # scalar ef_decay (float32), so selected_stacked(bf16) * ef_factor promotes to
    # float32; scatter_ into a bf16 M requires matching dtype. fp32-then-cast is
    # also numerically preferable to a bf16 multiply. (No-op when M is float32.)
    ef_src_list = list((selected_stacked * ef_factor).to(dtype).unbind(dim=0))
    for m, idx, ef_src in zip(M, indices_list, ef_src_list):
        if select_dim == -2:
            idx_exp = idx.unsqueeze(-1).expand(*idx.shape, m.size(-1))
        else:
            idx_exp = idx.unsqueeze(-2).expand(*idx.shape[:-1], m.size(-2), idx.shape[-1])
        m.scatter_(dim=select_dim, index=idx_exp, src=ef_src)

    # Convert to bf16 and unstack for communication
    U_selected = list(U_stacked.to(dtype=torch.bfloat16).unbind(dim=0))

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
