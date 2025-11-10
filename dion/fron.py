import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union


from .newton_schulz_triton import newton_schulz_triton, zeropower_via_newtonschulz5
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach, lion_update_foreach

# Reuse Muon's helper functions
from .muon import (
    muon_update_newton_schulz,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)


def _full_dtype_and_shape(p: Tensor) -> Tuple[torch.Size, torch.dtype, torch.device]:
    if isinstance(p, DTensor):
        shape = p.size()  # global shape
        dev = p.to_local().device
        return shape, p.dtype, dev
    return p.size(), p.dtype, p.device


class Fron(Optimizer):
    """
    Distributed Fractional Orthonormalization optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Fron, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        fraction: Fraction of rows/columns to orthogonalize per update (0 < fraction <= 1).
        ef_decay: Error-feedback decay factor for Fron algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        adjust_lr: How to adjust the learning rate for Fron updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Fron updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.
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
        newton_schulz_func: Optional[Callable] = None,
    ):
        # Chenk hyperparameter
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if ef_decay < 0.0:
            raise ValueError(f"Invalid error-feedback decay (ef_decay): {ef_decay}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            ef_decay=ef_decay,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="fron",
            step=0,
            fraction=fraction,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Group by optimizers
        fron_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "fron":
                fron_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        fron_tasks = self._create_fron_tasks(fron_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(fron_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_fron_state_layer(self, param: Tensor) -> dict:
        """
        Layer-sharded momentum state for FRON:
        - 'momentum_full' lives only on the owner rank (owner is implicitly device_rank).
        """
        st = self.state[param]
        if "momentum_full" not in st:
            st["momentum_full"] = None
        return st

    def _get_or_initialize_fron_state_local(self, param: Tensor) -> dict:
        """
        Local-shard momentum state for FRON:
        - Each rank keeps 'momentum_local' matching its local shard shape.
        """
        st = self.state[param]
        if "momentum_local" not in st:
            if isinstance(param, DTensor):
                lp = param.to_local()
                st["momentum_local"] = torch.zeros(
                    lp.shape, dtype=param.dtype, device=lp.device
                )
            else:
                st["momentum_local"] = torch.zeros_like(param)
        return st

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _pad_states(self, states: List[dict], n: int) -> List[dict]:
        """
        Pad states to length n. Real entries get is_pad=False; padded entries get is_pad=True.
        """
        out = list(states)
        # Mark existing entries explicitly as not padded
        for st in out:
            if "is_pad" not in st:
                st["is_pad"] = False
        # Append padded placeholders
        while len(out) < n:
            out.append({"momentum_full": None, "is_pad": True})
        return out

    def _create_fron_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "fron",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Fron optimizer only supports matrix parameters."

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            # Wrap hyperparameters as tensors for torch.compile
            fron_args = dict(
                lr=torch.tensor(group["lr"]),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=torch.tensor(group["fraction"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
            )

            # Create batches of parameters of size self._world_size
            for batch_params in create_param_batches(
                params, batch_size=self._world_size
            ):
                grads = [p.grad for p in batch_params]

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(batch_params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, pl)
                        for i, pl in enumerate(batch_params[0].placements)
                        if pl.is_shard() and batch_params[0].device_mesh.size(i) > 1
                    ]

                    # If we don't flatten 3D matrices, we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    if not group["flatten"]:
                        matrix_dims = {
                            batch_params[0].ndim - 1,
                            batch_params[0].ndim - 2,
                        }
                        is_batch_sharded = any(
                            pl.dim not in matrix_dims for _, pl in shard_placements
                        )
                        shard_placements = [
                            (i, pl)
                            for i, pl in shard_placements
                            if pl.dim in matrix_dims
                        ]

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that non-flattened 3D tensors can have additional sharded batch dimensions
                    # Flattened 3D tensors are limited to one sharded dimension out of all dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Fron does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and batch_params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if is_batch_sharded and not is_matrix_sharded:

                    # For this case, we use local momentum per shard
                    for x, g in zip(batch_params, grads):
                        st = self._get_or_initialize_fron_state_local(x)

                        # Create task for non-communicating local update
                        yield AsyncTask(
                            fron_update_local_async(
                                X=[x],
                                G=[g],
                                STATE=st,
                                **fron_args,
                            )
                        )
                    continue

                # Otherwise we use layer-sharded momentum and owner mapping
                states = [
                    self._get_or_initialize_fron_state_layer(p) for p in batch_params
                ]

                # Create task for communicating batch update
                yield AsyncTask(
                    fron_update_batch_async(
                        X=pad_batch(batch_params, self._world_size),
                        G=pad_batch(grads, self._world_size),
                        STATES=self._pad_states(states, self._world_size),
                        shard_dim=sharded_tensor_dim,
                        **fron_args,
                    )
                )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        # Check whether algo_name matches "lion"
        if algo_name != "lion":
            raise RuntimeError(f"lion is applied to {algo_name} groups")

        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        # Check whether algo_name matches "adamw"
        if algo_name != "adamw":
            raise RuntimeError(f"adamw is applied to {algo_name} groups")

        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def fron_update_local_async(
    X: List[Tensor],
    G: List[Tensor],
    STATE: dict,  # Should put local momentum state here
    lr: Tensor,
    ef_decay: Tensor,
    fraction: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    flatten: bool,
    adjust_lr: Optional[str],
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    assert len(X) == len(G) == 1
    x = X[0]
    g = to_local(G)[0]  # local shard grad
    M = STATE["momentum_local"]  # local shard momentum

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, x.shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, x.shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Error feedback on local shard and orthonormalize fraction
    M.add_(g.to(dtype=M.dtype))
    O_local = fractional_orthonormalize_update(
        M_full=M,
        fraction=float(fraction),
        ef_decay=ef_decay,
        flatten=flatten,
        epsilon=epsilon,
        newton_schulz_func=newton_schulz_func,
    )

    # Apply update locally
    fron_update_post_orthogonalize(
        X=to_local([x]),
        U=[O_local],
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )
    yield


def fron_update_batch_async(
    X: List[Tensor],  # DTensors or local Tensors
    G: List[Tensor],  # local shards (regular Tensors)
    STATES: List[dict],  # layer-sharded optimizer states (each has 'momentum_full')
    lr: Tensor,
    ef_decay: Tensor,
    fraction: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    FRON layer-sharded path:
      - Matrix-dim sharded: all_to_all shards <-> full, compute once on owner, all_to_all back.
      - Replicated/unsharded: compute once on owner (index=device_rank), all_gather dense updates.
      - Single-GPU (batch size=1): compute once on owner, apply locally.
    """

    # Ownership-by-index: owner of batch index j is rank j.
    assert 0 <= device_rank < world_size
    assert len(X) == len(STATES) == world_size  # Guaranteed by padding

    # convert gradients to local shards
    G_local = to_local(G)
    frac = float(fraction)

    # Compute adjusted lr from global shape (matrix dims last)
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Matrix-dimension sharded path
    if shard_dim is not None:
        assert len(X) == world_size, "Batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert X[0].size(shard_dim) % world_size == 0, (
            f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} "
            f"is not divisible by world size {world_size}."
        )

        # Shards -> single full gradient on owner (bf16 comm)
        G_bf16 = [g.to(dtype=torch.bfloat16) for g in G_local]
        recv_shards = [torch.empty_like(g) for g in G_bf16]
        work = dist.all_to_all(recv_shards, G_bf16, group=process_group, async_op=True)
        yield
        work.wait()

        full_grad_bf16 = torch.cat(recv_shards, dim=shard_dim)

        # Ownership-by-index contract:
        # For layer-sharded path, the "owner" of batch index j is rank j (device_rank).
        owner_state = STATES[device_rank]
        owner_is_pad = owner_state.get("is_pad", False)

        if owner_is_pad:
            # For pads, do NOT allocate momentum_full or run NS.
            # Just return zero shards.
            send_shards = [torch.zeros_like(g) for g in G_bf16]  # bf16 payloads
            U = [torch.empty_like(g) for g in G_bf16]
            work = dist.all_to_all(U, send_shards, group=process_group, async_op=True)
            yield
            work.wait()
        else:
            # For non-pads allocate momentum_full, accumulate, NS, split, all_to_all
            if owner_state["momentum_full"] is None:
                full_shape, param_dtype, param_device = _full_dtype_and_shape(
                    X[device_rank]
                )
                owner_state["momentum_full"] = torch.zeros(
                    full_shape, dtype=param_dtype, device=param_device
                )
            M_full = owner_state["momentum_full"]
            full_grad = full_grad_bf16.to(dtype=M_full.dtype)
            M_full.add_(full_grad)

            O_full = fractional_orthonormalize_update(
                M_full=M_full,
                fraction=frac,
                ef_decay=ef_decay,
                flatten=flatten,
                epsilon=epsilon,
                newton_schulz_func=newton_schulz_func,
            )
            send_shards = [
                t.contiguous().to(torch.bfloat16)
                for t in torch.tensor_split(O_full, world_size, dim=shard_dim)
            ]
            U = [torch.empty_like(g) for g in G_bf16]
            work = dist.all_to_all(U, send_shards, group=process_group, async_op=True)
            yield
            work.wait()

    # Replicated / unsharded
    else:
        # Owner index is device_rank
        x_owner = X[device_rank]
        g_owner = G_local[device_rank]
        st_owner = STATES[device_rank]
        owner_is_pad = st_owner.get("is_pad", False)

        if owner_is_pad:
            # For pads, do NOT allocate momentum_full or run NS.
            if process_group is not None and world_size > 1:
                zero_full_bf16 = torch.zeros_like(
                    g_owner, dtype=torch.bfloat16
                ).contiguous()
                U = [torch.empty_like(zero_full_bf16) for _ in range(world_size)]
                work = dist.all_gather(
                    U, zero_full_bf16, group=process_group, async_op=True
                )
                yield
                work.wait()
            else:
                # Single-GPU: produce a local zero update
                U = [torch.zeros_like(x_owner)]
        else:
            # For non-pads allocate momentum_full, accumulate, NS, and gather
            if st_owner["momentum_full"] is None:
                full_shape, param_dtype, param_device = _full_dtype_and_shape(x_owner)
                st_owner["momentum_full"] = torch.zeros(
                    full_shape, dtype=param_dtype, device=param_device
                )
            M_full = st_owner["momentum_full"]
            M_full.add_(g_owner.to(dtype=M_full.dtype))

            O_full = fractional_orthonormalize_update(
                M_full=M_full,
                fraction=frac,
                ef_decay=ef_decay,
                flatten=flatten,
                epsilon=epsilon,
                newton_schulz_func=newton_schulz_func,
            )

            if process_group is not None and world_size > 1:
                O_full_bf16 = O_full.to(dtype=torch.bfloat16).contiguous()
                U = [torch.empty_like(O_full_bf16) for _ in range(world_size)]
                work = dist.all_gather(
                    U, O_full_bf16, group=process_group, async_op=True
                )
                yield
                work.wait()
            else:
                # Single-GPU: produce a local update
                U = [O_full]

    # Ensure foreach dtypes match parameter shards for the update
    X_local = to_local(X)
    U = [u.to(dtype=xi.dtype) for u, xi in zip(U, X_local)]

    fron_update_post_orthogonalize(
        X=X_local,
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def make_work_view(M: Tensor) -> Tuple[Tensor, bool]:
    I, J = M.size(-2), M.size(-1)
    if I < J:
        return M.mT, True
    return M, False


def fractional_orthonormalize_update(
    M_full: Tensor,
    fraction: float,
    ef_decay: Tensor,
    flatten: bool,
    epsilon: Tensor,
    newton_schulz_func: Callable,
) -> Tensor:
    M_work, transposed = make_work_view(M_full)
    I, J = M_work.size(-2), M_work.size(-1)
    if fraction >= 1.0:
        ortho_update = muon_update_newton_schulz(
            M_work, newton_schulz_func, flatten=flatten, epsilon=epsilon
        )
        M_work.mul_(ef_decay)
    else:
        k = int(math.ceil(fraction * J))
        ortho_update = topk_and_orthonormalize(
            M_work,
            ef_decay=ef_decay,
            k=k,
            flatten=flatten,
            epsilon=epsilon,
            newton_schulz_func=newton_schulz_func,
        )
    return ortho_update.mT.contiguous() if transposed else ortho_update


def topk_and_orthonormalize(
    M_work: Tensor,
    ef_decay: Tensor,
    k: int,
    flatten: bool,
    epsilon: Tensor,
    newton_schulz_func,
) -> Tensor:
    """ """
    # Compute once
    alpha = M_work.abs().sum(dim=-2)  # [J]
    K = torch.topk(alpha, k, sorted=False).indices  # [k]
    # Select and orthonormalize
    M_sel = torch.index_select(M_work, dim=-1, index=K)  # [I, k]
    O_sel = muon_update_newton_schulz(
        M_sel, newton_schulz_func, flatten=flatten, epsilon=epsilon
    )
    # In-place error-feedback decay only on selected columns:
    M_work[..., K] *= ef_decay
    # Construct the full update matrix
    O_full = torch.zeros_like(M_work, dtype=O_sel.dtype)
    O_full.index_copy_(dim=-1, index=K, source=O_sel)
    return O_full


def fron_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def adamw_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


def lion_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
) -> Generator[None, None, None]:
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
    yield
