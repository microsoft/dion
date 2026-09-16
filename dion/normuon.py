import math
import operator

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
    compute_split_lr_scales,
    local_split_row_scales,
)
from .opt_utils import AsyncTask, as_scalar_tensor, to_local
from .muon import muon_update_pre_orthogonalize, muon_update_post_orthogonalize


# Returned by _layout_version for a marker that is present but not a usable
# integer. Distinct from None (absent) and from any real version number.
_INVALID_LAYOUT_VERSION = -1


def _layout_version(state: dict):
    """Read the variance layout marker, or None when it is absent.

    The marker is stored as a plain Python int, the same way torch's own
    AdamW stores a non-capturable ``step``: it needs no device round-trip and
    survives torch.save and DCP. Some checkpoint backends still hand small
    scalars back as a 0-d tensor or a numpy integer, so accept any exact
    integer and funnel everything else to a sentinel the caller rejects,
    rather than failing an identity check on the type.
    """
    version = state.get("variance_neuron_layout_version")
    if version is None:
        return None
    if isinstance(version, Tensor):
        version = version.item() if version.numel() == 1 else _INVALID_LAYOUT_VERSION
    if isinstance(version, bool):
        return _INVALID_LAYOUT_VERSION
    try:
        return operator.index(version)
    except TypeError:
        return _INVALID_LAYOUT_VERSION


def neuron_variance_buffer(param: Tensor, flatten: bool) -> Tensor:
    """One variance per matrix row, retaining parameter rank and DTensor layout."""
    if flatten and param.ndim > 2:
        rows = param
        # Supported row shards remain row shards; no flatten across a shard.
        for dim in range(1, param.ndim):
            rows = rows.narrow(dim, 0, 1)
        return torch.zeros_like(rows)
    return torch.zeros_like(param[..., :1])


class NorMuon(DistributedOrthoBase):
    """
    Distributed NorMuon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For NorMuon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for NorMuon algorithm.
        muon_beta2: Second beta parameter for NorMuon algorithm's adaptive updates.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor. Pass a Tensor to carry it as a persistent
            device tensor the kernels read live, so filling it in place drives a
            CUDA-graph-captured step (see dion.cuda_graph); a float is baked at capture.
        cautious_wd: Whether to apply weight decay only where update and parameter signs align.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Each 3D+ parameter is a single [out, prod(rest)] matrix for
                both orthogonalization and neuron normalization. Sharded
                parameters must use dim 0; column shards are unsupported.
                Old flattened-convolution optimizer checkpoints are rejected;
                retain model weights and construct a fresh optimizer instead.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_gram_newton_schulz: Whether to use Gram Newton-Schulz for orthogonalization.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is ``func(input: Tensor, epsilon: float) -> Tensor``.

    Param groups may also set the ``split_sizes`` option to orthogonalize row
    blocks of a fused 2D weight independently (e.g. a fused QKV projection as
    separate Q, K, and V blocks, which may have unequal sizes under GQA).
    Newton-Schulz, the learning-rate adjustment, and the NorMuon norm rescale
    run per block, matching the update that separate per-block parameters
    would receive, while the model keeps the single wide GEMM. Under FSDP the
    norm rescale is still shard-local (the existing distributed approximation),
    but the learning-rate adjustment is exact per block. See the README for
    details.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    NorMuon optimizer: https://arxiv.org/abs/2510.05491
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: Union[float, Tensor] = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        newton_schulz_func: Optional[Callable] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
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
            mu=mu,
            muon_beta2=muon_beta2,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="normuon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(
            params, distributed_mesh, "normuon", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_state(self, param: Tensor, algo: str, group: dict) -> dict:
        if algo == self._algo_name and group["flatten"] and param.ndim > 2:
            # Validate before narrow() can redistribute an unsupported column shard.
            self._get_shard_info(param, group)
        state = super()._get_or_initialize_state(param, algo, group)
        if algo == self._algo_name and "variance_neuron" not in state:
            state["variance_neuron"] = neuron_variance_buffer(param, group["flatten"])
            if group["flatten"] and param.ndim > 2:
                # Plain int, not a device tensor: unlike ``step_dev`` this is
                # never read by a kernel and never advances under graph replay,
                # so it needs no device round-trip. It is written only when the
                # flattened layout is used, so an unflattened state keeps
                # exactly the key set it had before this marker existed.
                state["variance_neuron_layout_version"] = 1
        if algo == self._algo_name and param.ndim > 2:
            self._validate_variance_state(param, state, group["flatten"])
        return state

    @staticmethod
    def _validate_variance_state(param: Tensor, state: dict, flatten: bool) -> None:
        flattened = flatten and param.ndim > 2
        version = _layout_version(state)
        expected = ((param.shape[0],) + (1,) * (param.ndim - 1)
                    if flattened else tuple(param.shape[:-1]) + (1,))
        variance = state.get("variance_neuron")
        if (flattened and version != 1) or (
            not flattened and version is not None
        ) or (variance is not None and tuple(variance.shape) != expected) or (
            flattened and variance is None
        ):
            raise ValueError(
                "Incompatible NorMuon variance_neuron state for "
                f"shape {tuple(param.shape)}, flatten={flatten}: expected "
                f"shape {expected} and {'layout version 1' if flattened else 'unflattened layout'}. "
                "Old flattened-convolution checkpoints used different statistics "
                "and cannot be migrated exactly. Load model weights only and "
                "construct a fresh optimizer; do not change flatten on live state."
            )

    def load_state_dict(self, state_dict):
        # Check before Optimizer.load_state_dict replaces any live state/groups.
        # A version is necessary: old and new shapes coincide for some 1x1 kernels.
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        for saved, current in zip(saved_groups, self.param_groups):
            if len(saved["params"]) != len(current["params"]):
                raise ValueError("loaded state dict has a mismatched parameter group size")
            # A checkpoint written by other tooling, or by an older version of
            # this optimizer, may omit keys that only exist because they are in
            # ``defaults``. Fall back to the live group, which is what
            # Optimizer.load_state_dict leaves in place for a key the saved
            # group does not carry.
            if saved.get("algorithm", current.get("algorithm")) != self._algo_name:
                continue
            flatten = saved.get("flatten", current.get("flatten", False))
            shard_group = {**saved, "flatten": flatten}
            for key, param in zip(saved["params"], current["params"]):
                self._get_shard_info(param, shard_group)
                self._validate_variance_state(
                    param, state_dict["state"].get(key, {}), flatten
                )
        return super().load_state_dict(state_dict)

    def _get_shard_info(self, param: Tensor, group: dict):
        result = super()._get_shard_info(param, group)
        _, is_matrix_sharded, sharded_tensor_dim = result
        if (is_matrix_sharded and group["flatten"] and param.ndim > 2
                and sharded_tensor_dim != 0):
            raise NotImplementedError(
                "NorMuon with flatten=True requires 3D+ parameters to be sharded "
                f"at dim 0, not dim {sharded_tensor_dim}: other dimensions split "
                "a neuron's columns across ranks."
            )
        if is_matrix_sharded and sharded_tensor_dim == param.ndim - 1:
            raise NotImplementedError(
                "NorMuon currently does not support parameters sharded along the last dimension. "
                "Please avoid shards at dim -1."
            )
        return result

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched NorMuon task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        # Catch a group whose ``flatten`` was changed after its state was built,
        # before yielding any task, so a later invalid group cannot leave
        # earlier parameters half-updated. Grad-less parameters are included.
        #
        # Only the layout marker is compared here. Everything else the full
        # validator checks is immutable once the state exists: the buffer shape
        # is fixed at creation, and a parameter's placements cannot change
        # between steps. Those are enforced where state is created
        # (_get_or_initialize_state, reached from __init__ and add_param_group)
        # and where it is replaced (load_state_dict). Re-deriving
        # _get_shard_info for every parameter on every step cost a DTensor
        # process-group lookup per parameter and could never report anything
        # the construction-time check had not already rejected.
        for group in param_groups:
            flatten = bool(group["flatten"])
            for p in group["params"]:
                if p.ndim > 2:
                    state = self.state.get(p)
                    if state and ("variance_neuron_layout_version" in state) != flatten:
                        # Disagrees with the layout the state was built for;
                        # defer to the full validator for the error message.
                        self._validate_variance_state(p, state, flatten)
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "NorMuon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            update_args = dict(
                lr=group["lr"],
                momentum=torch.tensor(group["mu"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                weight_decay=as_scalar_tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
                cautious_wd=group["cautious_wd"],
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            num_heads = self._resolve_num_heads(group)
            split_sizes = self._resolve_split_sizes(group)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, self._algo_name, group) for p in params]
                momentums = [s["momentum"] for s in states]
                variances_neuron = [s["variance_neuron"] for s in states]

                split_args = {}
                if split_sizes is not None:
                    self._validate_split_shape(split_sizes, params)
                    split_args = dict(
                        split_sizes=split_sizes,
                        split_scales=compute_split_lr_scales(
                            split_sizes, params[0].shape, group["adjust_lr"]
                        ),
                    )

                if num_heads is not None:
                    params, gradients, momentums, variances_neuron = (
                        self._prepare_head_split(
                            num_heads, params, gradients, momentums, variances_neuron
                        )
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
                    normuon_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances_neuron,
                        shard_dim=shard_dim,
                        **split_args,
                        **megabatch_args,
                    )
                )


def normuon_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    momentum: Tensor,
    muon_beta2: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    cautious_wd: bool = False,
    split_sizes: Optional[Tuple[int, ...]] = None,
    split_scales: Optional[Tuple[float, ...]] = None,
) -> Generator[None, None, None]:
    """
    Mega-batched NorMuon update: processes ALL same-shape parameters in one
    communication round instead of world_size-sized batches.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V)

    # Pre-orthogonalize: update momentum
    U = muon_update_pre_orthogonalize(
        G=to_local(G), M=to_local(M), momentum=momentum, nesterov=nesterov,
    )

    # Convert shard_dim to negative for comm_dim
    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None

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
        global_comm_dim_size = X[0].shape[comm_dim]
    else:
        global_comm_dim_size = None

    # Orthogonalize via shared megabatch communication. With split_sizes, each
    # row block is orthogonalized independently. split_scales -- which converts
    # the whole-matrix adjusted_lr applied below into the per-block adjustment
    # that separate parameters would receive -- is deliberately NOT applied
    # here: the normalization below divides U by sqrt(V) and rescales it back
    # to the Frobenius norm of its own input, so a per-block factor applied
    # before it cancels in the division and re-enters only through that norm.
    # That reproduces the factor only when the rescale is per block, which it
    # is not on the sharded path, where a shard spanning a block boundary
    # blends the two blocks' scales into one factor and silently gives both
    # the wrong learning rate. The scales are applied after normalization
    # instead, per row block of this rank's shard.
    # Request the stacked [N, *shape] result directly: the normalization below
    # immediately re-stacks the orthogonalized update, so taking the list and
    # re-stacking it would be an unbind-then-restack round-trip (N selects + a
    # stacking copy per shape group). return_stacked hands back the tensor the
    # megabatch already has assembled.
    U_stacked = yield from megabatch_orthogonalize_async(
        U,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=flatten,
        epsilon=epsilon,
        global_comm_dim_size=global_comm_dim_size,
        split_sizes=split_sizes,
        split_scales=None,
        return_stacked=True,
    )

    # NorMuon normalization using stacked tensors for fewer kernel launches.
    # With split_sizes, the Frobenius-norm-preserving rescale runs per row
    # block, matching separate per-block parameters. Only a row shard breaks
    # that: U then holds a contiguous row range rather than whole matrices, so
    # the rescale stays per-shard there (the existing distributed
    # approximation) and mixes blocks where a shard straddles a block
    # boundary. Keyed on comm_dim == -2 for the same reason as the row offset
    # below -- any other comm_dim leaves dim -2 whole, so the row blocks are
    # intact and per-block normalization is the right thing there too. Only
    # the norm-preserving rescale is ever approximated; the learning rate
    # itself is exact per block, applied below.
    norm_split_sizes = (
        None if (comm_dim == -2 and process_group is not None) else split_sizes
    )
    V_local = to_local(V)
    V_stacked = torch.stack(V_local)
    flatten_rows = flatten and U_stacked.ndim > 3
    if flatten_rows:
        U_shape, V_shape = U_stacked.shape, V_stacked.shape
        # Explicit columns also work on empty output-channel shards, where
        # reshape(N, 0, -1) is ambiguous. Retain the existing shard-local rescale.
        U_stacked = U_stacked.reshape(U_shape[0], U_shape[1], math.prod(U_shape[2:]))
        V_stacked = V_stacked.reshape(V_shape[0], V_shape[1], 1)
    U_stacked, V_stacked = normuon_normalization_stacked(
        U_stacked, V_stacked, muon_beta2, split_sizes=norm_split_sizes
    )
    if flatten_rows:
        U_stacked = U_stacked.reshape(U_shape)
        V_stacked = V_stacked.reshape(V_shape)
    # Write the updated variance buffers back into the persistent per-param
    # state in a single multi-tensor kernel instead of N separate copy_
    # launches, and unbind U in one dispatch instead of N selects. Both are
    # numerically identical to the per-element loops; this is purely a
    # host-dispatch reduction (the V writeback alone was ~20% of step CPU time).
    torch._foreach_copy_(V_local, V_stacked.unbind(0))

    # Apply the per-block learning-rate adjustment now that normalization can
    # no longer cancel it. When the rows themselves are what is sharded, this
    # rank holds rows [row_offset, row_offset + local_rows) of the fused matrix;
    # split_sizes requires dim 0 to be divisible by the world size
    # (megabatch_base raises otherwise), so every rank holds the same number of
    # rows and the offset is exact. Any other comm_dim leaves dim -2 whole, so
    # the offset is 0 -- the condition is on comm_dim == -2 rather than on
    # "sharded at all" so that this does not silently depend on NorMuon
    # rejecting last-dim shards elsewhere. Blocks whose scale is 1.0 are
    # skipped, so the common case costs one narrow + mul_ per block that
    # intersects this shard, and nothing at all when adjust_lr is None.
    if split_scales is not None:
        if comm_dim == -2 and process_group is not None:
            local_rows = global_comm_dim_size // world_size
            row_offset = device_rank * local_rows
            # A raise, not an assert: -O strips asserts, and this is the one
            # check standing between a wrong row offset and every block
            # silently running at the wrong learning rate. It is a host-side
            # integer compare on a path that already does per-block narrows.
            if U_stacked.size(-2) != local_rows:
                raise RuntimeError(
                    f"expected {local_rows} local rows on the sharded split "
                    f"path, got {U_stacked.size(-2)}; the row offset derived "
                    f"from device_rank would not match this rank's shard."
                )
        else:
            row_offset = 0
        for start, end, scale in local_split_row_scales(
            split_sizes, split_scales, row_offset, U_stacked.size(-2)
        ):
            U_stacked.narrow(-2, start, end - start).mul_(scale)

    U = list(U_stacked.unbind(0))

    # Compute scaled learning rate
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Post-orthogonalize: apply update
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        cautious_wd=cautious_wd,
    )


@torch.compile(fullgraph=True)
def normuon_normalization_stacked(
    U: Tensor,  # [N, rows, cols]
    V: Tensor,  # [N, rows, 1]  (variance neuron buffer)
    muon_beta2: Tensor,
    split_sizes: Optional[Tuple[int, ...]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    NorMuon normalization on stacked 3D tensors for minimal kernel launches.
    Equivalent to normuon_normalization but operates on a single stacked tensor
    instead of a list, reducing per-element kernel overhead.
    With ``split_sizes``, each row block is normalized independently so the
    Frobenius-norm-preserving rescale matches separate per-block parameters.
    Returns (normalized_U, updated_V).
    """
    if split_sizes is not None:
        results = [
            _normuon_normalization_core(u, v, muon_beta2)
            for u, v in zip(
                U.split(list(split_sizes), dim=-2),
                V.split(list(split_sizes), dim=-2),
            )
        ]
        normalized_U = torch.cat([r[0] for r in results], dim=-2)
        V = torch.cat([r[1] for r in results], dim=-2)
        return normalized_U, V

    return _normuon_normalization_core(U, V, muon_beta2)


def _normuon_normalization_core(
    U: Tensor,
    V: Tensor,
    muon_beta2: Tensor,
) -> Tuple[Tensor, Tensor]:
    V_dtype = V.dtype
    U = U.to(dtype=V_dtype)

    # Frobenius norm per matrix: [N, 1, 1]
    norm_U = U.norm(p=2, dim=(-2, -1), keepdim=True)

    # Neuron-wise variance: mean of squares along last dim -> [N, rows, 1]
    neuron_norms = (U * U).mean(dim=-1, keepdim=True)

    # Update variance buffer (EMA)
    V = torch.lerp(V, neuron_norms, 1 - muon_beta2)

    # Normalize
    denom = V.sqrt() + 1e-8
    normalized_U = U / denom

    # Rescale to preserve Frobenius norm
    norm_U_new = normalized_U.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    normalized_U = normalized_U * (norm_U / norm_U_new)

    return normalized_U, V
