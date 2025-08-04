import math
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from dataclasses import dataclass
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Replicate, Shard
from torch.distributed.tensor import randn as dtensor_randn
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Any, Dict, List, Tuple, Optional, Union

from .opt_utils import to_local
from .scalar_opts import adamw_update, lion_update

try:
    from torch.distributed.tensor.placement_types import _StridedShard
except ImportError:
    _StridedShard = None


@dataclass
class _DionParamConfig:
    """
    Per-parameter configuration for Dion optimizer.
    """

    # Dimensions of the tensor that is sharded
    outer_shard_tensor_dim: Optional[int] = None
    inner_shard_tensor_dim: Optional[int] = None

    # Dimensions of the device mesh that the tensor is sharded over
    outer_shard_mesh_dim: Optional[int] = None
    inner_shard_mesh_dim: Optional[int] = None

    # Use transposed version of the algorithm
    is_transposed: bool = False

    # Whether to all-reduce compressed P and R instead of full gradient
    # This should always be False for 1D tensors
    compressed_all_reduce = False

    # Sharding configurations for the Q matrix
    Q_sharded_placements: Optional[Tuple[Placement]] = None
    Q_inner_unsharded_placements: Optional[Tuple[Placement]] = None


@dataclass
class DionMixedPrecisionConfig:
    """
    Configuration for mixed precision in Dion optimizer.
    None means that optimizer states will use the same dtype as each parameter.
    """

    # Momentum state for all algorithms
    momentum_dtype: Optional[torch.dtype] = None
    # Dion Q matrix
    Q_dtype: Optional[torch.dtype] = None
    # Adam variance state
    variance_dtype: Optional[torch.dtype] = None
    # TODO look into separate dtypes for communication operations


class Dion(Optimizer):
    """
    Distributed Dion Optimizer.
    https://arxiv.org/abs/2504.05295

    Args:
        params: Parameters for the optimizer.
        replicate_mesh: DeviceMesh or ProcessGroup for replicated data parallelism.
            Use DeviceMesh for hybrid sharded FSDP and ProcessGroup for DistributedDataParallel.
        outer_shard_mesh: Parameter sharding DeviceMesh, replicated during orthogonalization.
            This is the FS dimension in the paper.
        inner_shard_mesh: Parameter sharding DeviceMesh, sharded during orthogonalization.
            This is the TP dimension in the paper.
        replicate_mesh_grad_sync: If True, optimizer handles data-parallel gradient sync.
            If False, the optimizer expects gradients to be already synchronized.
        rank_fraction: r/d fraction for low-rank approximation. Used to compute the low-rank dimension.
            This may be specified per param-group to have different rank fractions.
        rank_multiple_of: Round up the low-rank dimension to a multiple of this number.
            This may be useful to ensure even sharding.
        lr: Base learning rate. For Dion, this will be scaled based on the matrix dimensions.
            For non-Dion algorithms, this is the actual learning rate and no additional scaling is done.
        qr_method: Method for computing QR decomposition during orthogonalization.
            Options are "rcqr" (Randomized Cholesky QR), "cqr" (Cholesky QR), and "qr" (standard QR).
        cqr_warmup_steps: Number of steps before enabling CQR. Ignored if qr_method is not "cqr".
        rcqr_oversample: Random sketch oversampling factor for RCQR. Ignored if qr_method is not "rcqr".

    Note: We assume parameters are all DTensor or all regular Tensors. All sharded tensors are assumed
    to be uniformly sharded - that is, each device along the sharding axis has identical size shards.
    The only distributed scenarios supported are:
        - DTensor + DeviceMesh: sharding with FSDP2 fully_shard() and/or TP parallelize_module().
        - regular Tensor + ProcessGroup: No sharding allowed. DDP may be used.
    FSDP1 (FullyShardedDataParallel wrapper class) is not supported.
    """

    def __init__(
        self,
        params: ParamsT,
        replicate_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        outer_shard_mesh: Optional[DeviceMesh] = None,
        inner_shard_mesh: Optional[DeviceMesh] = None,
        replicate_mesh_grad_sync: bool = True,
        rank_fraction: float = 1.0,
        rank_multiple_of: int = 1,
        lr: float = 0.01,
        mu: float = 0.95,  # Momentum for Dion
        betas: Tuple[float, float] = (0.9, 0.95),  # Betas for AdamW and Lion
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        power_iters: int = 1,  # Number of power iterations for low-rank approximation
        qr_method: str = "rcqr",  # Method for computing QR decomposition
        cqr_warmup_steps: int = 150,  # Warmup steps before enabling CQR
        rcqr_oversample: float = 1.25,  # Random sketch matrix oversampling for RCQR
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rank_fraction <= 0 or rank_fraction > 1:
            raise ValueError(f"Invalid rank fraction: {rank_fraction}")
        if rank_multiple_of <= 0:
            raise ValueError(f"Invalid rank multiple of: {rank_multiple_of}")
        if power_iters <= 0:
            raise ValueError(f"Invalid power iterations: {power_iters}")
        if qr_method not in ("qr", "cqr", "rcqr"):
            raise ValueError(f"Unknown QR method: {qr_method}")

        # Check device mesh
        if replicate_mesh is not None:
            if not isinstance(replicate_mesh, (DeviceMesh, ProcessGroup)):
                raise TypeError(
                    f"Replicate mesh must be a DeviceMesh or ProcessGroup, but got {type(replicate_mesh)}."
                )
        if outer_shard_mesh is not None:
            if not isinstance(outer_shard_mesh, DeviceMesh):
                raise TypeError(
                    f"Outer shard mesh must be a DeviceMesh, but got {type(outer_shard_mesh)}."
                )
            if outer_shard_mesh.ndim != 1:
                raise ValueError(
                    f"Outer shard mesh must be 1D, but got {outer_shard_mesh.ndim}D. Try using a 1D sub-mesh."
                )
            if outer_shard_mesh == replicate_mesh:
                raise ValueError(
                    "Outer shard mesh must be different from replicate mesh."
                )
        if inner_shard_mesh is not None:
            if not isinstance(inner_shard_mesh, DeviceMesh):
                raise TypeError(
                    f"Inner shard mesh must be a DeviceMesh, but got {type(inner_shard_mesh)}."
                )
            if inner_shard_mesh.ndim != 1:
                raise ValueError(
                    f"Inner shard mesh must be 1D, but got {inner_shard_mesh.ndim}D. Try using a 1D sub-mesh."
                )
            if inner_shard_mesh == replicate_mesh:
                raise ValueError(
                    "Inner shard mesh must be different from replicate mesh."
                )
            if inner_shard_mesh == outer_shard_mesh:
                raise ValueError("Outer and inner shard meshes must be different.")

        # Default arguments for each param group
        defaults = dict(
            rank_fraction=rank_fraction,
            rank_multiple_of=rank_multiple_of,
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            oversample=rcqr_oversample,
            algorithm="dion",
            step=0,
        )
        super().__init__(params, defaults)

        self._power_iters = power_iters
        self._qr_method = qr_method
        self._cqr_warmup_steps = cqr_warmup_steps
        self._rng = None

        # This is intentionally not in self.state so it doesn't get checkpointed
        # State here may change upon resharding a checkpoint, so we recompute it
        self._param_config: Dict[Tensor, _DionParamConfig] = {}

        self._replicate_mesh = replicate_mesh
        self._outer_shard_mesh = outer_shard_mesh
        self._inner_shard_mesh = inner_shard_mesh
        self._replicate_mesh_grad_sync = replicate_mesh_grad_sync

        # Get global ranks for outer and inner shard meshes
        if self._outer_shard_mesh is not None:
            self._outer_shard_ranks = dist.get_process_group_ranks(
                self._outer_shard_mesh.get_group()
            )
        else:
            self._outer_shard_ranks = None
        if self._inner_shard_mesh is not None:
            self._inner_shard_ranks = dist.get_process_group_ranks(
                self._inner_shard_mesh.get_group()
            )
        else:
            self._inner_shard_ranks = None

        # Mixed precision
        if mixed_precision_config is None:
            mixed_precision_config = DionMixedPrecisionConfig()
        self._mixed_precision_config = mixed_precision_config
        # TODO check what happens when loading state dict with different precision

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            algo = group["algorithm"]

            # Increment step
            group["step"] += 1
            step = group["step"]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            oversample = torch.tensor(group["oversample"])

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                param_config = self._get_dion_param_config(param)

                # Dion is intended to be used without gradient sync over the replicated data-parallel world
                # For each parameter, we either all-reduce its gradient or compressed PQ states
                # Except when replicate_mesh_grad_sync is False, in which case we assume that
                # gradients are already synchronized before the optimizer step is called.
                if (
                    self._replicate_mesh_grad_sync
                    and not param_config.compressed_all_reduce
                ):
                    gradient = all_reduce(param.grad, self._replicate_mesh)
                else:
                    gradient = param.grad

                # Call the corresponding update function
                if algo == "dion":
                    if not state:
                        self._init_opt_state_dion(
                            param,
                            state,
                            group["rank_fraction"],
                            group["rank_multiple_of"],
                        )

                    Q = state["Q"]
                    compressed_all_reduce = (
                        self._replicate_mesh_grad_sync
                        and param_config.compressed_all_reduce
                    )

                    # Unshard Q along the inner sharding dimension
                    if param_config.Q_inner_unsharded_placements is not None:
                        assert isinstance(Q, DTensor)
                        Q = Q.redistribute(
                            placements=param_config.Q_inner_unsharded_placements,
                            async_op=True,
                        )

                    if not isinstance(param, DTensor) and self._rng is None:
                        # Lazy initialization of RNG for non-DTensor parameters
                        # All DP ranks must have identical seed, so we just set to 0 here
                        self._rng = torch.Generator(device=param.device)
                        self._rng.manual_seed(0)

                    qr_method = self._qr_method
                    if qr_method == "cqr" and step <= self._cqr_warmup_steps:
                        # Enable CQR only after warmup, so matrices are well-conditioned
                        qr_method = "rcqr"

                    Q_new = dion_update(
                        X=param,
                        G=gradient,
                        M=state["momentum"],
                        Q=Q,
                        lr=lr,
                        mu=mu,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                        transpose=param_config.is_transposed,
                        power_iters=self._power_iters,
                        qr_method=qr_method,
                        oversample=oversample,
                        compressed_all_reduce=compressed_all_reduce,
                        replicate_mesh=self._replicate_mesh,
                        inner_shard_mesh_dim=param_config.inner_shard_mesh_dim,
                        rng=self._rng,
                    )

                    # Shard new Q along the inner sharding dimension
                    if param_config.Q_sharded_placements is not None:
                        assert isinstance(Q_new, DTensor)
                        Q_new = Q_new.redistribute(
                            placements=param_config.Q_sharded_placements,
                        )
                    state["Q"] = Q_new

                elif algo == "adamw":
                    if not state:
                        self._init_opt_state_adam(param, state)

                    # Sharded DTensor params can be updated element-wise
                    adamw_update(
                        X=to_local(param),
                        G=to_local(gradient),
                        M=to_local(state["momentum"]),
                        V=to_local(state["variance"]),
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=epsilon,
                    )

                elif algo == "lion":
                    if not state:
                        self._init_opt_state_momentum(param, state)

                    # Sharded DTensor params can be updated element-wise
                    lion_update(
                        X=to_local(param),
                        G=to_local(gradient),
                        M=to_local(state["momentum"]),
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                    )

                else:
                    raise ValueError(f"Unknown algorithm: {algo}")

        return loss

    @torch.no_grad()
    def synchronize_for_checkpoint(self):
        """
        Synchronize the internal optimizer states across the replicated mesh.

        Dion uses compressed gradient synchronization with decoupled momentum, which
        results in optimizer states diverging across the replicated data-parallel mesh.
        To ensure consistency of distributed checkpoints, we must manually synchronize
        the optimizer states before saving a checkpoint. If replicate_mesh is None or
        replicate_mesh_grad_sync is False, this function is a no-op.
        """

        if self._replicate_mesh is None or not self._replicate_mesh_grad_sync:
            # Nothing to do
            return

        # Get all tensors in optimizer states
        state_tensors: List[Tensor] = []
        for state in self.state.values():
            assert isinstance(state, dict)
            for val in state.values():
                if isinstance(val, Tensor):
                    state_tensors.append(val)

        # All-reduce each tensor in the optimizer state
        for tensor in state_tensors:
            tensor = to_local(tensor)
            result = all_reduce(tensor, self._replicate_mesh)
            tensor.copy_(result)

    def _get_dion_param_config(self, x: Tensor) -> _DionParamConfig:
        """
        Get the Dion-specific parameter configuration for a given tensor.
        If the configuration is not already initialized, it will be created.
        Lazy initialization is necessary because PyTorch allows new parameters
        to be added to the optimizer after it has been created.
        """
        if x in self._param_config:
            return self._param_config[x]

        if x.ndim > 2:
            raise NotImplementedError(
                f"Tensors with more than 2 dimensions are not supported. Got {x.ndim}D tensor."
            )

        # Check for allowed DeviceMesh and DTensor combinations
        # We only allow DTensor + DeviceMesh or regular Tensor + ProcessGroup
        using_device_mesh = (
            isinstance(self._replicate_mesh, DeviceMesh)
            or isinstance(self._outer_shard_mesh, DeviceMesh)
            or isinstance(self._inner_shard_mesh, DeviceMesh)
        )
        using_process_group = isinstance(self._replicate_mesh, ProcessGroup)
        if using_device_mesh and not isinstance(x, DTensor):
            raise TypeError("When using DeviceMesh, all parameters must be DTensor.")
        if using_process_group and isinstance(x, DTensor):
            raise TypeError(
                "When using DTensor parameters, the data parallel group must be specified by a DeviceMesh instead of ProcessGroup."
            )

        # State is initialized for both matrix and scalar parameters
        config = _DionParamConfig()

        # By default, we transpose matrices so that dim0 >= dim1
        # This can change depending on sharding
        if x.ndim == 2:
            m, n = x.shape
            config.is_transposed = m < n

        # Detect sharding dimensions for DTensor
        if isinstance(x, DTensor) and x.ndim == 2:
            device_mesh = x.device_mesh
            placements = x.placements
            assert len(placements) == device_mesh.ndim

            dim_map = [None for _ in range(x.ndim)]

            for mesh_dim, placement in enumerate(placements):
                # StridedShard not allowed
                if _StridedShard is not None and isinstance(placement, _StridedShard):
                    raise NotImplementedError(
                        f"StridedShard is not supported. Ensure that FSDP and TP shard different dimensions of each matrix."
                    )

                # Skip non-sharded device mesh dimensions
                if not placement.is_shard():
                    continue
                tensor_dim = placement.dim

                # Check for double sharding on same tensor dimension
                if dim_map[tensor_dim] is not None:
                    raise RuntimeError(
                        f"Got double-sharded DTensor for tensor dimension {placement.dim}."
                    )
                dim_map[tensor_dim] = mesh_dim

                # Get global ranks corresponding to this mesh dimension
                mesh_dim_ranks = dist.get_process_group_ranks(
                    device_mesh.get_group(mesh_dim)
                )

                # Check if it matches the outer or inner shard ranks
                outer_sharded, inner_sharded = False, False
                if mesh_dim_ranks == self._outer_shard_ranks:
                    config.outer_shard_tensor_dim = tensor_dim
                    config.outer_shard_mesh_dim = mesh_dim
                    outer_sharded = True
                if mesh_dim_ranks == self._inner_shard_ranks:
                    config.inner_shard_tensor_dim = tensor_dim
                    config.inner_shard_mesh_dim = mesh_dim
                    inner_sharded = True

                # Check for double sharding on same mesh dimension
                if outer_sharded and inner_sharded:
                    raise RuntimeError(
                        "Cannot have outer and inner sharding over the same process group."
                    )

                # Check for sharding on unrecognized mesh dimension
                # Ignore edge case for single GPU "sharding" = Replicate()
                # Make sure to check that size(mesh_dim) > 1
                if (
                    device_mesh.size(mesh_dim) > 1
                    and not outer_sharded
                    and not inner_sharded
                ):
                    raise RuntimeError(
                        f"Got DTensor sharded on unrecognized {mesh_dim=}, which does not match outer_shard_mesh or inner_shard_mesh."
                    )

            # Set transpose so that orthogonalization happens over the inner sharding dimension
            # Standard Dion orthogonalizes over tensor dimension 0
            if config.inner_shard_tensor_dim == 0 or config.outer_shard_tensor_dim == 1:
                config.is_transposed = False
            # Transposed Dion orthogonalizes over tensor dimension 1
            if config.outer_shard_tensor_dim == 0 or config.inner_shard_tensor_dim == 1:
                config.is_transposed = True

        self._param_config[x] = config
        return config

    def _init_opt_state_momentum(self, param: Tensor, state: Dict[str, Any]):
        # Create the momentum buffer
        # If param is DTensor, this will also be a DTensor
        state["momentum"] = torch.zeros_like(
            param, dtype=self._mixed_precision_config.momentum_dtype
        )

    def _init_opt_state_adam(self, param: Tensor, state: Dict[str, Any]):
        self._init_opt_state_momentum(param, state)
        state["variance"] = torch.zeros_like(
            param, dtype=self._mixed_precision_config.variance_dtype
        )

    def _init_opt_state_dion(
        self,
        param: Tensor,
        state: Dict[str, Any],
        rank_fraction: float,
        rank_multiple_of: int,
    ):
        """
        Initialize the optimizer state for Dion.
        This includes the momentum buffer and the Q matrix.

        The low-rank factor `r` is computed as `rank_fraction` * min(m, n),
        and rounded up to the next multiple of `rank_multiple_of`.
        """
        if param.ndim != 2:
            raise ValueError(
                f"Expected Dion parameters to be 2D matrix, but got {param.ndim}D. "
                f"For scalar parameters, set 'algorithm' to 'lion' or 'adamw' when creating param group."
            )

        param_config = self._get_dion_param_config(param)
        self._init_opt_state_momentum(param, state)

        # Compute the low-rank factor r
        m, n = param.shape
        r = rank_fraction * min(m, n)
        r = rank_multiple_of * math.ceil(r / rank_multiple_of)
        r = min(r, m, n)
        Q_shape = (m, r) if param_config.is_transposed else (n, r)

        # Set compressed_all_reduce based on if it saves communication cost
        # Otherwise we will all-reduce the gradient matrix instead
        if rank_fraction < 1 and (m + n) * r < m * n:
            param_config.compressed_all_reduce = True

        # Get dtype for Q
        if self._mixed_precision_config.Q_dtype is not None:
            Q_dtype = self._mixed_precision_config.Q_dtype
        else:
            Q_dtype = param.dtype

        if isinstance(param, DTensor):
            # Directly construct Q as DTensor
            # Shard(0) on outer sharding mesh and Shard(1) on inner sharding mesh
            placements = [Replicate() for _ in range(param.device_mesh.ndim)]
            if param_config.outer_shard_mesh_dim is not None:
                placements[param_config.outer_shard_mesh_dim] = Shard(0)
            if param_config.inner_shard_mesh_dim is not None:
                placements[param_config.inner_shard_mesh_dim] = Shard(1)
            param_config.Q_sharded_placements = tuple(placements)

            # Q is unsharded along the inner sharding dimension only
            if param_config.inner_shard_mesh_dim is not None:
                placements[param_config.inner_shard_mesh_dim] = Replicate()
                param_config.Q_inner_unsharded_placements = tuple(placements)
            else:
                # No inner sharding, so placements are the same as Q_sharded_placements
                param_config.Q_inner_unsharded_placements = None

            # DTensor RNG should automatically produce identical results across DP replicas
            Q = dtensor_randn(
                Q_shape,
                device_mesh=param.device_mesh,
                dtype=Q_dtype,
                placements=param_config.Q_sharded_placements,
            )

        else:
            # Make sure all DP ranks have the same Q
            Q = torch.randn(Q_shape, device=param.device, dtype=Q_dtype)
            self._replicate_mesh_broadcast(Q)

        state["Q"] = Q

    def _replicate_mesh_broadcast(self, tensor: Tensor):
        """
        Broadcast a tensor from rank 0 over the replicated data-parallel world.
        Tensor is modified in place.
        """
        if self._replicate_mesh is None:
            # No data parallelism used, do nothing
            pass
        elif isinstance(self._replicate_mesh, DeviceMesh):
            for group in self._replicate_mesh.get_all_groups():
                dist.broadcast(tensor, group=group, group_src=0)
        elif isinstance(self._replicate_mesh, ProcessGroup):
            dist.broadcast(tensor, group=self._replicate_mesh, group_src=0)
        else:
            raise TypeError(
                "Data parallel mesh must be either a DeviceMesh or ProcessGroup."
            )


@torch.compile()
def dion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    Q: Tensor,  # Q matrix for power iteration
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    transpose: bool,
    power_iters: int,
    qr_method: str,  # Method for computing QR decomposition
    oversample: float = 1.25,
    compressed_all_reduce: bool = True,
    replicate_mesh: Union[DeviceMesh, ProcessGroup, None] = None,
    inner_shard_mesh_dim: Optional[int] = None,  # for DTensor only
    rng: Optional[torch.Generator] = None,  # for regular tensor only
) -> Tensor:
    """
    Dion optimizer algorithm.
    """
    # Match dtype of Q and M
    Q_init_dtype = Q.dtype
    Q = Q.to(M.dtype)

    # Add new gradient to momentum
    M.add_(G)

    # Compute low-rank approximation of M = P @ Q^T
    # M, Q, P, R should all have the same dtype
    P, R = power_iteration(
        M.T if transpose else M,
        Q,
        power_iters=power_iters,
        qr_method=qr_method,
        oversample=oversample,
        compressed_all_reduce=compressed_all_reduce,
        replicate_mesh=replicate_mesh,
        inner_shard_mesh_dim=inner_shard_mesh_dim,
        rng=rng,
    )
    P, R = fix_all_zero_or_nan(P, R, Q, M)

    # Error feedback
    # M = M - (1 - mu) * (P @ R.T)
    if not transpose:
        M.add_(P @ R.T, alpha=-(1 - mu))
    else:
        M.add_(R @ P.T, alpha=-(1 - mu))

    # Column normalize R to get new Q
    # Do this in float32 for numerical stability
    # For sharded matrices, DTensor will automatically sync the full-tensor norm
    R = R.to(dtype=torch.float32)
    R_norm = R.norm(dim=0, keepdim=True) + epsilon
    Q = (R / R_norm).to(P.dtype)

    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Compute update scale factor
    fan_out = X.size(0)
    fan_in = X.size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight update
    # X = X - scaled_lr * (P @ Q.T)
    if not transpose:
        X.add_(P @ Q.T, alpha=-scaled_lr)
    else:
        X.add_(Q @ P.T, alpha=-scaled_lr)

    # Return new Q for next iteration
    return Q.to(Q_init_dtype)


def power_iteration(
    B: Tensor,
    Q_init: Tensor,
    power_iters: int,
    qr_method: str,  # Method for computing QR decomposition
    oversample: float,  # Oversampling factor for RCQR
    compressed_all_reduce: bool,  # Whether to all-reduce low-rank P and Q
    replicate_mesh: Union[DeviceMesh, ProcessGroup, None] = None,
    inner_shard_mesh_dim: Optional[int] = None,  # for DTensor only
    rng: Optional[torch.Generator] = None,  # for regular tensor only
) -> Tuple[Tensor, Tensor]:
    """
    Returns a low-rank approximation of B by power iteration.
    Compute P and Q such that (approximately) B = P @ Q^T.
    """
    assert (
        B.dtype == Q_init.dtype
    ), f"Expected inputs to have the same dtype, but got {B.dtype} and {Q_init.dtype}."

    Q = Q_init

    for _ in range(power_iters):
        P = B @ Q
        if compressed_all_reduce:
            P = all_reduce(P, replicate_mesh)

        if isinstance(P, DTensor):
            P = distributed_orthogonalize(
                P,
                qr_method=qr_method,
                oversample=oversample,
                shard_mesh_dim=inner_shard_mesh_dim,
            )
        else:
            P = orthogonalize(P, qr_method=qr_method, oversample=oversample, rng=rng)

        Q = B.T @ P
        if compressed_all_reduce:
            Q = all_reduce(Q, replicate_mesh)

    return P, Q


def orthogonalize(
    P: Tensor,
    qr_method: str = "rcqr",
    oversample: float = 1.25,
    rng: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Orthogonalize the input matrix using the specified method.
        - "qr": Householder QR (torch.linalg.qr)
        - "cqr": Cholesky QR
        - "rcqr": Randomized Cholesky QR
    """
    assert qr_method in ("qr", "cqr", "rcqr"), f"Unknown method: {qr_method}"
    assert not isinstance(P, DTensor), "Use distributed_orthogonalize() instead"

    m, n = P.shape
    original_dtype = P.dtype

    # Cholesky QR (may not be numerically stable) unless matrices are well-conditioned
    if qr_method == "cqr":
        P_32 = P.to(dtype=torch.float32)  # multiply in float32 for numerical stability
        R, info = torch.linalg.cholesky_ex(P_32.T @ P_32, upper=True)
        if info == 0:
            Q = torch.linalg.solve_triangular(R, P_32, upper=True, left=False)
        else:
            qr_method = "rcqr"  # Fallback to randomized QR

    # Standard QR is faster than RCQR if matrix is square or wide
    if qr_method == "qr" or (qr_method == "rcqr" and m <= n):
        Q, _ = torch.linalg.qr(P.to(dtype=torch.float32))

    # Randomized Cholesky QR
    if qr_method == "rcqr" and m > n:
        # Compute size k and round up to next multiple of 128
        m, n = P.shape
        k = math.ceil(oversample * n / 128.0) * 128
        std = math.sqrt(1.0 / k)

        # Generate random sketching matrix of shape (k, m)
        # Must use same RNG seed on all DP ranks
        S = torch.empty((k, m), device=P.device, dtype=P.dtype).normal_(
            std=std, generator=rng
        )
        SP = S @ P

        # Calculate right triangular matrix R using standard QR, and solve for Q
        _, R = torch.linalg.qr(SP.to(dtype=torch.float32), mode="r")
        Q = torch.linalg.solve_triangular(
            R, P.to(dtype=torch.float32), upper=True, left=False
        )

        # Apply another iteration of Cholesky QR to better orthogonalize Q
        QQ = Q.T @ Q
        R, _ = torch.linalg.cholesky_ex(QQ, upper=True)
        Q = torch.linalg.solve_triangular(R, Q, upper=True, left=False)

    return Q.to(dtype=original_dtype)


def distributed_orthogonalize(
    P: DTensor,
    qr_method: str = "rcqr",
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
) -> DTensor:
    """
    Orthogonalize the input matrix using the specified method.
        - "qr": Householder QR (torch.linalg.qr)
        - "cqr": Cholesky QR
        - "rcqr": Randomized Cholesky QR

    P has shape (m, r) and can be sharded on shard_mesh_dim. If sharded, it is
    assumed to have Shard(0) placement (row-sharded along size-m tensor dimension).
    The orthogonalized output will have the same shape and sharding as the input P.

    Methods "rcqr" and "cqr" can work directly with sharded matrices.
    The "qr" method will unshard the full matrix before computing QR.
    The "cqr" method is faster than "rcqr" but is not numerically stable.
    """
    assert qr_method in ("qr", "cqr", "rcqr"), f"Unknown method: {qr_method}"
    assert isinstance(P, DTensor), "Use orthogonalize() for regular tensors"

    # Get desired placements for output
    m, r = P.shape
    original_dtype = P.dtype
    placements = [Replicate() for _ in range(P.device_mesh.ndim)]
    if shard_mesh_dim is not None:
        placements[shard_mesh_dim] = Shard(0)  # Shard(0) = rows

    # Cholesky QR (may not be numerically stable) unless matrices are well-conditioned
    if qr_method == "cqr":
        P_32 = P.to(dtype=torch.float32)  # multiply in float32 for numerical stability
        PP: DTensor = P_32.T @ P_32
        PP_full = PP.full_tensor()
        R_full, info = torch.linalg.cholesky_ex(PP_full, upper=True)

        if info == 0:
            P_local = P_32.redistribute(placements=placements).to_local()
            Q_local = torch.linalg.solve_triangular(
                R_full, P_local, upper=True, left=False
            )
            Q = DTensor.from_local(
                Q_local.to(dtype=original_dtype),  # cast back to original dtype
                device_mesh=P.device_mesh,
                placements=placements,
            )
        else:
            qr_method = "rcqr"  # Fallback to randomized QR

    # Standard QR is faster than RCQR if matrix is square or wide
    if qr_method == "qr" or (qr_method == "rcqr" and m <= r):
        # Fully unshard for QR
        P_full = P.full_tensor().to(dtype=torch.float32)
        Q_full, _ = torch.linalg.qr(P_full)
        Q = DTensor.from_local(
            Q_full.to(dtype=original_dtype),  # cast back to original dtype
            device_mesh=P.device_mesh,
        ).redistribute(placements=placements)

    # Randomized Cholesky QR
    if qr_method == "rcqr" and m > r:
        # Apply random sketch to input matrix
        S = generate_random_sketch_dtensor(
            P, oversample=oversample, shard_mesh_dim=shard_mesh_dim
        )
        SP: DTensor = S @ P

        # Calculate right triangular matrix R using standard QR, and solve for Q
        P_local = (
            P.redistribute(placements=placements).to_local().to(dtype=torch.float32)
        )
        SP_full = SP.full_tensor().to(dtype=torch.float32)
        _, R_full = torch.linalg.qr(SP_full, mode="r")
        Q_local = torch.linalg.solve_triangular(R_full, P_local, upper=True, left=False)
        Q = DTensor.from_local(
            Q_local,  # dtype is float32
            device_mesh=P.device_mesh,
            placements=placements,
        )

        # Apply another iteration of Cholesky QR to better orthogonalize Q
        QQ: DTensor = Q.T @ Q
        QQ_full = QQ.full_tensor()
        R_full, _ = torch.linalg.cholesky_ex(QQ_full, upper=True)
        Q_local = torch.linalg.solve_triangular(R_full, Q_local, upper=True, left=False)
        Q = DTensor.from_local(
            Q_local.to(dtype=original_dtype),  # cast back to original dtype
            device_mesh=P.device_mesh,
            placements=placements,
        )

    assert Q.dtype == original_dtype
    return Q


def generate_random_sketch_dtensor(
    P: DTensor,
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
) -> DTensor:
    """
    Generate a random sketching matrix S of shape (oversample * r, m).
    If shard_mesh_dim is not None, S will be sharded on that mesh dimension.
    S is always Shard(1) which is column-sharded (along size-m tensor dimension).
    """
    # Compute size k and round up to next multiple of 128
    m, r = P.shape
    k = math.ceil(oversample * r / 128.0) * 128
    std = math.sqrt(1.0 / k)

    # Generate random sketching matrix of shape (k, m)
    S_placements = [Replicate() for _ in range(P.device_mesh.ndim)]
    if shard_mesh_dim is not None:
        S_placements[shard_mesh_dim] = Shard(1)  # Shard(1) = columns

    # DTensor RNG should automatically produce identical results across DP replicas
    S = dtensor_randn(
        (k, m),
        device_mesh=P.device_mesh,
        dtype=P.dtype,
        placements=S_placements,
    )
    S *= std

    return S


def fix_all_zero_or_nan(
    P: Tensor, Q: Tensor, Q_init: Tensor, B: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    If input is all zero, P and Q will be nan or all zero.
    We want to return the conditional expressions:

        if is_all_zero:
            P = torch.zeros_like(P)
            Q = Q_init
        else:
            P = P
            Q = Q

    Here this is implemented without data-dependent control flow.
    To avoid additional communication, we handle sharded tensors independently.
    """
    B_local = to_local(B)
    is_all_zero = (B_local == 0).all()
    not_all_zero = ~is_all_zero
    P = P.nan_to_num() * not_all_zero
    Q = Q.nan_to_num() * not_all_zero + Q_init * is_all_zero
    return P, Q


def all_reduce(
    tensor: Tensor,
    reduce_mesh: Union[DeviceMesh, ProcessGroup, None],
    reduce_op: str = "avg",
) -> Tensor:
    """
    Generic all-reduce operation to support both DTensor and regular Tensor.
    """
    if isinstance(tensor, DTensor):
        # First redistribute any Partial to Replicate
        if any(p.is_partial() for p in tensor.placements):
            tensor = tensor.redistribute(
                placements=[
                    Replicate() if p.is_partial() else p for p in tensor.placements
                ]
            )

    if reduce_mesh is None:
        # No reduce mesh used, do nothing
        return tensor

    if isinstance(tensor, DTensor):
        assert isinstance(reduce_mesh, DeviceMesh)
        # All-reduce local shard over the device mesh
        tensor_local = funcol.all_reduce(
            tensor.to_local(),
            reduceOp=reduce_op,
            group=reduce_mesh,
        )
        # Convert back to DTensor
        tensor = DTensor.from_local(
            tensor_local,
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
        )

    else:
        tensor = funcol.all_reduce(tensor, reduceOp=reduce_op, group=reduce_mesh)

    return tensor
