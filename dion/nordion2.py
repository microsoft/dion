import math
import torch
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
from .dion2 import (
    dion2_pre_orthogonalize,
    dion2_post_orthogonalize,
    dion2_pre_accumulate,
    _make_select_and_orthogonalize,
)
from .normuon import normuon_normalization_stacked, _normuon_normalization_core


class NorDion2(DistributedOrthoBase):
    """
    Distributed NorDion2 optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For NorDion2, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        fraction: Fraction of submatrix to orthogonalize per update (0 < fraction <= 1).
        mu: Momentum factor for NorDion2 algorithm.
        muon_beta2: Second beta parameter for NorDion2 algorithm's adaptive updates.
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
        selection_scope: On the FSDP2 row-sharded path, how the orthogonalized
            submatrix is selected. "global" (default): exact top-k on the
            assembled whole matrix -- layout-invariant/reproducible and better-
            converging. "local": per-shard top-k (union) -- cheaper comm but a
            sharding-dependent approximation that converges slightly worse; opt
            in when comm-bound at large scale. No-op off the row-sharded path.

    NorDion2 optimizer applying Dion2 update to NorMuon
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        fraction: float = 0.25,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        use_gram_newton_schulz: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        triton_post_ortho: bool = False,
        selection_scope: str = "global",
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if selection_scope not in ("local", "global"):
            raise ValueError(
                f"selection_scope must be 'local' or 'global', got {selection_scope!r}"
            )
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if muon_beta2 < 0.0:
            raise ValueError(f"Invalid muon_beta2: {muon_beta2}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            fraction=float(fraction),
            mu=mu,
            muon_beta2=muon_beta2,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="nordion2",
            step=0,
            selection_scope=selection_scope,
        )
        super().__init__(
            params, distributed_mesh, "nordion2", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )
        if triton_post_ortho:
            from .dion2_triton import TRITON_AVAILABLE
            if not TRITON_AVAILABLE:
                raise ImportError(
                    "triton_post_ortho=True requires the 'triton' package, which is not installed. "
                    "Install it with: pip install dion[triton]  (or: pip install triton)"
                )
        self._triton_post_ortho = triton_post_ortho

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == self._algo_name and "variance_neuron" not in state:
            # V stored in param dtype (bf16); upcast to fp32 for compute, truncated back on write
            state["variance_neuron"] = torch.zeros_like(param[..., 0:1])
        return state

    def _get_shard_info(self, param: Tensor, group: dict):
        result = super()._get_shard_info(param, group)
        _, is_matrix_sharded, sharded_tensor_dim = result
        if is_matrix_sharded and sharded_tensor_dim == param.ndim - 1:
            raise NotImplementedError(
                "NorDion2 currently does not support parameters sharded along the last dimension. "
                "Please avoid shards at dim -1."
            )
        return result

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched NorDion2 task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "NorDion2 optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            update_args = dict(
                lr=torch.tensor(group["lr"]),
                fraction=group["fraction"],
                momentum=torch.tensor(group["mu"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
                triton_post_ortho=self._triton_post_ortho,
                selection_scope=group["selection_scope"],
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
                variances_neuron = [s["variance_neuron"] for s in states]

                if num_heads is not None:
                    params, gradients, momentums, variances_neuron = self._prepare_head_split(
                        num_heads, params, gradients, momentums, variances_neuron
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
                    nordion2_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances_neuron,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


def nordion2_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    fraction: float,
    momentum: Tensor,
    muon_beta2: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    triton_post_ortho: bool = False,
    selection_scope: str = "global",
) -> Generator[None, None, None]:
    """
    Mega-batched NorDion2 update: processes ALL same-shape parameters in one
    communication round instead of world_size-sized batches.

    ``selection_scope`` mirrors Dion2: "global" (default) sends the full shard
    and selects the exact top-k on the assembled whole matrix (full comm,
    layout-invariant, better-converging); "local" selects each rank's top-k rows
    before communication (cheaper comm, but a sharding-variant approximation that
    converges slightly worse). See ``dion2_update_megabatch_async`` for details.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V)

    # Select along rows so selected slices preserve full rows for per-neuron normalization
    select_dim = -2
    is_sharded = shard_dim is not None

    # comm_dim for sharded communication: use select_dim
    comm_dim = select_dim if is_sharded else None
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
        U_full = dion2_pre_accumulate(G=to_local(G), M=to_local(M))
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
            global_comm_dim_size=global_dim_size,
        )
        if adjust_lr is None:
            adjusted_lr = lr
        elif adjust_lr == "spectral_norm":
            adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
        elif adjust_lr == "rms_norm":
            adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
        else:
            raise ValueError(f"Unknown adjust_lr: {adjust_lr}")
        # U_ortho rows are exactly zero except at globally-selected positions
        # this rank owns. The masked post applies error-feedback decay to those
        # rows of M, runs NorMuon per-neuron normalization on the selected rows
        # (updating the local variance buffer V in place), and applies the
        # masked weight update -- all keyed off the nonzero mask, no indices.
        nordion2_post_orthogonalize_masked(
            X=to_local(X),
            M=to_local(M),
            V=to_local(V),
            U=U_ortho,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            ef_decay=momentum,
            muon_beta2=muon_beta2,
            select_dim=select_dim,
        )
        return

    # --- Local selection (opt-in, selection_scope="local"): per-shard top-k, communicate only the
    # selected rows. Uniform k = ceil(fraction * ceil(global / world_size)) is
    # the per-rank selected count under FSDP2 contiguous chunking; the megabatch
    # pads every shard to exactly k (short/empty shards zero-pad), so the
    # alltoall stays uniform for uneven divisions too, with no special case. See
    # dion2 for the full rationale.
    if comm_dim is not None:
        padded_local = (global_dim_size + world_size - 1) // world_size
        k = max(1, int(math.ceil(fraction * padded_local)))
    else:
        k = None
    global_comm_dim_size = global_dim_size

    # Update momentum and compute the inputs for orthogonalization
    # Dion2 pre-orthogonalizes differs from NorMuon by applying damping before updating momentum
    U_selected, indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=momentum,
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

    # Update variance neuron buffer for the selected rows and normalize the
    # orthonormalized update. The gather of the selected variance rows, the
    # NorMuon normalization (fp32 compute, V stored bf16), and the scatter of
    # the updated rows back into the full buffer are fused into one compiled
    # graph so the per-param Python scatter loop and the eager gather/normalize
    # graph boundary collapse into a single launch per shape group.
    V_local = to_local(V)
    U_stacked = torch.stack(U_ortho)
    V_stacked = torch.stack(V_local)
    indices = torch.stack(indices_list, dim=0)

    U_stacked, V_stacked = nordion2_normalize_selected_stacked(
        U_stacked, V_stacked, indices, muon_beta2
    )
    U_normed = [U_stacked[i] for i in range(N)]
    torch._foreach_copy_(V_local, list(V_stacked.unbind(0)))

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
    # Cast U to match X's dtype for scatter_add_ (requires matching dtypes).
    X_local = to_local(X)

    if triton_post_ortho:
        from .dion2_triton import dion2_post_orthogonalize_triton

        dion2_post_orthogonalize_triton(
            X=X_local,
            U=U_normed,
            indices=indices_list,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            select_dim=select_dim,
        )
    else:
        dion2_post_orthogonalize(
            X=X_local,
            U=U_normed,
            indices=indices_list,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            select_dim=select_dim,
        )


@torch.compile(fullgraph=True)
def nordion2_normalize_selected_stacked(
    U: Tensor,  # [N, k, cols]  orthogonalized selected rows
    V_full: Tensor,  # [N, rows, 1]  full per-neuron variance buffer (param dtype)
    indices: Tensor,  # [N, k]  selected row indices
    muon_beta2: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Gather the selected variance rows, run NorMuon normalization on them, and
    scatter the updated rows back into the full variance buffer, all in one
    compiled graph. Compute is done in fp32 (V stored in param dtype); the
    returned V_full has the same dtype as the input.
    Returns (normalized_U, updated_V_full).
    """
    idx = indices.unsqueeze(-1)
    V_sel = torch.gather(V_full, dim=-2, index=idx).float()
    U_normed, V_sel_new = normuon_normalization_stacked(U, V_sel, muon_beta2)
    V_full = V_full.scatter(dim=-2, index=idx, src=V_sel_new.to(V_full.dtype))
    return U_normed, V_full


@torch.compile(fullgraph=True)
def nordion2_post_orthogonalize_masked(
    X: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    ef_decay: Tensor,
    muon_beta2: Tensor,
    select_dim: int,
):
    """
    Global-scope NorDion2 post-orthogonalize. ``U`` holds full-size orthogonalized
    shards, exactly zero except at the globally-selected rows this rank owns.

    Selection is along rows (``select_dim == -2``), so a selected row is nonzero
    over all columns. We:
      1. derive the selected-row mask from the nonzero rows of U,
      2. apply error-feedback decay to those rows of M,
      3. run NorMuon per-neuron normalization over the full matrix (equivalent to
         the selected submatrix alone, since orthonormal rows have unit norm and
         non-selected rows are exactly zero). The per-neuron variance EMA is
         written back to V only on selected rows.
      4. apply the masked weight update X += -adjusted_lr * U_normed.
    Inputs/outputs are regular Tensors, not DTensor.
    """
    assert select_dim == -2
    dtype = X[0].dtype

    # Weight decay on all weights
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    one = torch.ones((), dtype=M[0].dtype, device=M[0].device)
    ef = ef_decay.to(M[0].dtype)
    neg_lr = -adjusted_lr
    for x, m, v, u in zip(X, M, V, U):
        # [rows, 1] mask: True for selected (nonzero) rows.
        sel = u.to(torch.float32).abs().sum(dim=-1, keepdim=True) > 0
        # Error-feedback decay on the selected rows of momentum.
        m.mul_(torch.where(sel, ef, one))
        # NorMuon per-neuron normalization over the full matrix. V is updated
        # only on selected rows (gated by the mask).
        u_normed, v_new = _normuon_normalization_core(u, v.float(), muon_beta2)
        v.copy_(torch.where(sel, v_new.to(v.dtype), v))
        # Masked weight update: zero rows are no-ops.
        x.add_((neg_lr * u_normed).to(dtype))
