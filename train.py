import argparse
import math
import os
import shutil
import time
import tempfile
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import wandb
import yaml

from dataclasses import dataclass
from pathlib import Path
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import Optional

from models.gpt_model import GPT, GPTConfig, parallelize_gpt_model
from models.gpt_utils import DistributedDataLoader
from dion import Dion, DionMixedPrecisionConfig
from dion import DionReference
from dion import DionSimple
from dion import Muon
from dion import MuonReference


@dataclass
class Hyperparameters:
    # Data directory
    data_dir: str = "data/fineweb10B"

    # Training config
    batch_size: int = 8 * 64  # global batch size (across devices)
    device_batch_size: int = 64  # per-device batch size
    sequence_length: int = 1024  # tokens per sequence
    num_iterations: int = 5000
    warmup_ratio: float = 0.01
    warmdown_ratio: float = 0.2

    # Model config
    model_dim: int = 768
    n_layer: int = 12
    n_head: int = 6

    # Evaluation and logging
    val_loss_every: int = 125
    val_tokens: int = 10485760
    checkpoint_freq: int = 0
    checkpoint_dir: str = None
    wandb_project_name: str = "dion-test"

    # Optimizer
    optimizer: str = "dion"
    scalar_opt: str = "lion"

    # Main optimizer hyperparameters
    lr: float = 0.02
    mu: float = 0.95
    weight_decay: float = 0.01
    rank_fraction: float = 0.125

    # Optimizer specific hyperparameters
    qr_method: str = "rcqr"
    cqr_warmup: float = 0.05
    rcqr_oversample: float = 1.25
    replicate_mesh_grad_sync: bool = False
    mixed_precision: bool = False
    adjust_lr: str = "spectral_norm"  # for Muon only


# Helper function to only print on global rank 0
MASTER_PROCESS = True


def print0(*args):
    if MASTER_PROCESS:
        print(*args)


def parse_cli_args():
    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML file whose keys match train.py flags "
        "(CLI values always override the YAML).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory that contains fineweb_train_*.bin and fineweb_val_*.bin",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to load and save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps, 0 to disable",
    )

    # ---------- optimizer ----------
    parser.add_argument(
        "--optimizer", type=str, default=None, help="Choice of optimizer algorithm"
    )
    parser.add_argument(
        "--scalar_opt", type=str, help="Optimizer for scalar parameters", default=None
    )
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate")
    parser.add_argument(
        "--adjust_lr",
        type=str,
        default=None,
        help="Adjust learning rate method for Muon",
    )
    parser.add_argument(
        "--inv_rank_fraction",
        type=int,
        default=None,
        help="1/r rank fraction for Dion",
    )
    parser.add_argument(
        "--qr_method", type=str, default=None, choices=["qr", "cqr", "rcqr"]
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision for Dion"
    )

    # ---------- model ----------
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)

    # ---------- training hyperparameters ----------
    parser.add_argument(
        "--num_iterations", type=int, default=None, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Global batch size"
    )
    parser.add_argument("--device_batch_size", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--warmdown_ratio", type=float, default=None)

    # ---------- wandb logging ----------
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--wandb_project_name", type=str, default=None, help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_job_name",
        type=str,
        default=None,
        help="Append custom text to wandb job name",
    )

    # ---------- distributed training ----------
    parser.add_argument(
        "--dp_size", type=int, default=None, help="Data Parallel size (no sharding)"
    )
    parser.add_argument(
        "--fs_size", type=int, default=None, help="Fully Sharded Data Parallel size"
    )
    parser.add_argument(
        "--tp_size", type=int, default=None, help="Tensor Parallel size"
    )
    parser.add_argument(
        "--replicate_mesh_grad_sync",
        action="store_true",
        help="Do data-parallel gradient sync inside Dion optimizer",
    )
    parser.add_argument(
        "--fast_fsdp",
        action="store_true",
        help="Optimizer FSDP for speed instead of memory efficiency",
    )

    # ---------- debugging ----------
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile for model"
    )
    parser.add_argument(
        "--no_triton", action="store_true", help="Disable Triton kernels"
    )

    cli_args = parser.parse_args()
    if cli_args.config:
        # Read YAML → dict
        cfg_path = Path(cli_args.config)
        with cfg_path.open("r") as f:
            yaml_cfg = yaml.safe_load(f)

        # Copy any key the user did NOT supply on the CLI
        for k, v in yaml_cfg.items():
            if getattr(cli_args, k, None) is None:
                setattr(cli_args, k, v)

        # We need to manually handle store_true flags
        for flag in (
            "mixed_precision",
            "replicate_mesh_grad_sync",
            "fast_fsdp",
            "no_wandb",
            "no_compile",
            "no_triton",
            "debug",
        ):
            if yaml_cfg.get(flag, False):
                setattr(cli_args, flag, True)

    return cli_args


def override_args_from_cli(
    hp: Hyperparameters, cli_args: argparse.Namespace
) -> Hyperparameters:
    for key, value in vars(cli_args).items():
        if value is not None:
            if hasattr(hp, key):
                print0(f"Setting hyperparameter {key}={value}")
                setattr(hp, key, value)
    return hp


def init_distributed(dp_size, fs_size, tp_size) -> Optional[DeviceMesh]:
    """
    Initialize DeviceMesh or ProcessGroup for distributed training.
    If all mesh dimensions are None, we default to using DDP.
    """
    assert torch.cuda.is_available(), "CUDA must be available"
    assert torch.distributed.is_available(), "Distributed must be available"

    # Check that environment variables are set
    assert all(
        var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    ), "This script must be launched using the 'torchrun' command."
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set global master process flag
    global MASTER_PROCESS
    MASTER_PROCESS = rank == 0

    mesh_dims = (dp_size, fs_size, tp_size)
    if all(d is None for d in mesh_dims):
        # If no mesh dimensions given, initialize process group for DDP
        device_mesh = None
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)

        print0("=" * 80)
        print0("Distributed training initialized with DDP")
        print0(f"World size: {world_size}")

    else:
        # Use device mesh for distributed training
        # All mesh dimensions must be specified
        assert all(
            d is not None for d in mesh_dims
        ), f"All mesh dimensions (dp_size, fs_size, tp_size) must be specified, but got ({dp_size}, {fs_size}, {tp_size})"

        # Check if we have the right number of GPUs
        total_gpus = dp_size * fs_size * tp_size
        assert world_size == total_gpus, (
            f"World size {world_size} does not match expected size {total_gpus} "
            f"(DP {dp_size}, FS {fs_size}, TP {tp_size})"
        )
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(dp_size, fs_size, tp_size),
            mesh_dim_names=("dp", "fs", "tp"),
        )

        print0("=" * 80)
        print0("Distributed training initialized with DeviceMesh")
        print0(f"World size: {world_size}")
        print0(f"DP size: {dp_size}")
        print0(f"FS size: {fs_size}")
        print0(f"TP size: {tp_size}")
        print0(device_mesh)

    return device_mesh


def init_optimizer(
    model: GPT,
    device_mesh: Optional[DeviceMesh],
    ddp_model: Optional[DDP],
    hp: Hyperparameters,
    cli_args: argparse.Namespace,
):
    # Check that we have a valid scalar optimizer
    if hp.scalar_opt not in ["adamw", "lion"]:
        raise ValueError(f"Unrecognized scalar optimizer: {hp.scalar_opt}")

    # Separate the model's parameters based on their types
    matrix_params = list(model.transformer.h.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())

    # Matrix params use optimizer default settings
    param_groups = [dict(params=matrix_params)]

    # Add additional param groups with the necessary configurations for scalar params
    param_groups.append(
        dict(
            params=embedding_params,
            algorithm=hp.scalar_opt,
            lr=hp.lr,  # no LR adjustment for embedding parameters
            betas=(0.95, 0.98),
            weight_decay=0,  # no weight decay for embedding parameters
        )
    )
    param_groups.append(
        dict(
            params=lm_head_params,
            algorithm=hp.scalar_opt,
            lr=hp.lr / math.sqrt(hp.model_dim),  # scale LR for lm_head
            betas=(0.95, 0.98),
            weight_decay=0,  # no weight decay for lm_head parameters
        )
    )

    # Create the main optimizer
    if device_mesh is not None:
        replicate_mesh = device_mesh["dp"]
        outer_shard_mesh = device_mesh["fs"]
        inner_shard_mesh = device_mesh["tp"] if device_mesh["tp"].size() > 1 else None
    else:
        assert ddp_model is not None
        replicate_mesh = ddp_model.process_group
        outer_shard_mesh = None
        inner_shard_mesh = None

    if hp.mixed_precision:
        dion_mixed_precision_config = DionMixedPrecisionConfig(
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            Q_dtype=torch.bfloat16,
        )
    else:
        dion_mixed_precision_config = None

    if hp.optimizer == "dion":
        print0(f"Dion rank fraction: {hp.rank_fraction}")
        print0(f"Dion mixed precision: {hp.mixed_precision}")
        print0(f"Compressed data-parallel gradient sync: {hp.replicate_mesh_grad_sync}")
        opt = Dion(
            param_groups,
            replicate_mesh=replicate_mesh,
            outer_shard_mesh=outer_shard_mesh,
            inner_shard_mesh=inner_shard_mesh,
            replicate_mesh_grad_sync=hp.replicate_mesh_grad_sync,
            rank_fraction=hp.rank_fraction,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            qr_method=hp.qr_method,
            cqr_warmup_steps=round(hp.cqr_warmup * hp.num_iterations),
            rcqr_oversample=hp.rcqr_oversample,
            mixed_precision_config=dion_mixed_precision_config,
        )

    elif hp.optimizer == "dion_reference":
        print0(f"Dion rank fraction: {hp.rank_fraction}")
        print0(f"Dion QR method: {hp.qr_method}")
        print0(f"Dion mixed precision: {hp.mixed_precision}")
        print0(f"Compressed data-parallel gradient sync: {hp.replicate_mesh_grad_sync}")
        opt = DionReference(
            param_groups,
            replicate_mesh=replicate_mesh,
            outer_shard_mesh=outer_shard_mesh,
            inner_shard_mesh=inner_shard_mesh,
            replicate_mesh_grad_sync=hp.replicate_mesh_grad_sync,
            rank_fraction=hp.rank_fraction,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            qr_method=hp.qr_method,
            cqr_warmup_steps=round(hp.cqr_warmup * hp.num_iterations),
            rcqr_oversample=hp.rcqr_oversample,
            mixed_precision_config=dion_mixed_precision_config,
        )

    elif hp.optimizer == "muon":
        if device_mesh is not None:
            # Ensure that we have a supported device mesh configuration for Muon
            if inner_shard_mesh is not None and inner_shard_mesh.size() > 1:
                raise ValueError("Tensor parallel is not supported by Muon.")
            distributed_mesh = (
                outer_shard_mesh if outer_shard_mesh.size() > 1 else replicate_mesh
            )
            comm_method = "all-to-all" if outer_shard_mesh.size() > 1 else "all-gather"
        else:
            assert ddp_model is not None
            distributed_mesh = ddp_model.process_group  # using ProcessGroup for DDP
            comm_method = "all-gather"
        print0(f"Muon LR adjust method: {hp.adjust_lr}")
        print0(f"Triton Newton-Schulz kernels: {not cli_args.no_triton}")
        print0(f"Distributed Muon using: {comm_method}")
        opt = Muon(
            param_groups,
            distributed_mesh=distributed_mesh,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            nesterov=True,
            adjust_lr=hp.adjust_lr,
            use_triton=(not cli_args.no_triton),
        )

    elif hp.optimizer == "dion_simple":
        assert device_mesh is None, f"{hp.optimizer} does not support device mesh"
        print0(f"Dion rank fraction: {hp.rank_fraction}")
        opt = DionSimple(
            param_groups,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            rank=round(hp.rank_fraction * hp.model_dim),
        )

    elif hp.optimizer == "muon_reference":
        print0(f"Muon LR adjust method: {hp.adjust_lr}")
        opt = MuonReference(
            param_groups,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            nesterov=True,
            adjust_lr=hp.adjust_lr,
        )

    elif hp.optimizer == "adamw":
        print0("Using AdamW for all params, scalar optimizer will be ignored")
        print0("Setting all param groups to use unscaled base learning rate")
        for group in param_groups:
            group["lr"] = hp.lr
            group["betas"] = (0.9, 0.95)  # AdamW default betas
        opt = torch.optim.AdamW(
            param_groups,
            lr=hp.lr,
            betas=(0.9, 0.95),
            weight_decay=hp.weight_decay,
        )

    else:
        raise ValueError(f"Unsupported optimizer: {hp.optimizer}")

    # Check replicate_mesh_grad_sync and optimizer combination
    if hp.replicate_mesh_grad_sync and hp.optimizer not in ("dion", "dion_reference"):
        # Results will be wrong if replicate_mesh_grad_sync is set for non-Dion optimizer
        raise ValueError("replicate_mesh_grad_sync is set for non-Dion optimizer")
    if not hp.replicate_mesh_grad_sync and hp.optimizer in ("dion", "dion_reference"):
        # Using Dion without replicate_mesh_grad_sync means we won't get communication savings
        print0("Warning: not using replicate_mesh_grad_sync for Dion optimizer")

    return opt


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DistributedDataLoader,
        val_loader: DistributedDataLoader,
        wandb_id: Optional[str] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_id = wandb_id
        self.step = None
        self.DEFAULT_NAME = "checkpoint"

    def _get_state_dict(self) -> dict:
        # Use get_state_dict() instead of directly calling model.state_dict() etc.
        # This standardizes state dict for model and optimizer regardless of sharding
        model_state, opt_state = get_state_dict(self.model, self.optimizer)
        state_dict = {
            "model": model_state,
            "optimizer": opt_state,
            "train_loader": self.train_loader.state_dict(),
            "val_loader": self.val_loader.state_dict(),
            "step": self.step,
            "wandb_id": self.wandb_id,
        }
        return state_dict

    def save(self, name: Optional[str] = None, step: Optional[int] = None):
        """
        Save the checkpoint to the path "self.checkpoint_dir/name/".
        The distributed checkpoint is a directory with sharded files.
        It must reside on a shared filesystem accessible by all processes.
        """
        assert self.checkpoint_dir, "Checkpoint directory must be specified"
        self.step = step
        name = name or self.DEFAULT_NAME
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        print0(f"Saving checkpoint to {checkpoint_path}")

        # Save to a temporary subdirectory first
        tmpdir = None
        if dist.get_rank() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir=self.checkpoint_dir)

        # Broadcast tmpdir from rank 0 to all ranks
        obj_list = [tmpdir]
        dist.broadcast_object_list(obj_list, src=0)
        tmpdir = obj_list[0]

        # Save the checkpoint
        state_dict = self._get_state_dict()
        dcp.save(state_dict, checkpoint_id=tmpdir)
        dist.barrier()

        if dist.get_rank() == 0:
            # Delete any existing checkpoint with the same name
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            elif os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path, ignore_errors=True)
            # Move the checkpoint to the final location
            shutil.move(tmpdir, checkpoint_path)
        dist.barrier()

    def load(self, name: Optional[str] = None, allow_missing: bool = False):
        """
        Load the checkpoint from the path "self.checkpoint_dir/name/".
        """
        assert self.checkpoint_dir, "Checkpoint directory must be specified"
        name = name or self.DEFAULT_NAME
        checkpoint_path = os.path.join(self.checkpoint_dir, name)

        if not os.path.isdir(checkpoint_path):
            if allow_missing:
                print0(f"Checkpoint {checkpoint_path} does not exist, skipping load")
                return
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")

        print0(f"Loading checkpoint from {checkpoint_path}")
        state_dict = self._get_state_dict()
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

        # Load model and optimizer state dicts
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )

        # Load train and validation dataloader states
        self.train_loader.load_state_dict(state_dict["train_loader"])
        self.val_loader.load_state_dict(state_dict["val_loader"])

        self.step = state_dict["step"]
        self.wandb_id = state_dict["wandb_id"]
        dist.barrier()


def main():
    torch._dynamo.config.cache_size_limit = 100
    # --- Parse command line arguments and set hyperparams ---
    cli_args = parse_cli_args()
    hp = Hyperparameters()
    hp = override_args_from_cli(hp, cli_args)

    if cli_args.inv_rank_fraction:
        hp.rank_fraction = 1.0 / cli_args.inv_rank_fraction

    if hp.checkpoint_freq > 0:
        if not hp.checkpoint_dir:
            raise ValueError("Must specify --checkpoint_dir to save checkpoints")

    # --- Distributed training initialization ---
    device_mesh = init_distributed(
        dp_size=cli_args.dp_size,
        fs_size=cli_args.fs_size,
        tp_size=cli_args.tp_size,
    )
    print0("=" * 80)

    # --- DataLoader Setup ---
    if device_mesh is not None:
        # Combine replicated and sharded data parallel meshes for data loading
        data_parallel_mesh = device_mesh["dp", "fs"]._flatten()
        data_parallel_size = data_parallel_mesh.size()
        data_parallel_rank = data_parallel_mesh.get_local_rank()
    else:
        # We are using DDP with one global process group
        data_parallel_mesh = None
        data_parallel_size = dist.get_world_size()
        data_parallel_rank = dist.get_rank()

    if cli_args.debug:
        # in debug mode, make batch size very small
        hp.batch_size = 2 * data_parallel_size
        hp.device_batch_size = 1

    # Calculate validation steps
    tokens_in_global_batch = (
        hp.device_batch_size * hp.sequence_length * data_parallel_size
    )
    assert hp.val_tokens % tokens_in_global_batch == 0, "Invalid val_tokens"
    val_steps = hp.val_tokens // tokens_in_global_batch

    if cli_args.debug:
        # train for just a few steps
        hp.num_iterations = 20
        val_steps = min(val_steps, 2)

    # Calculate gradient accumulation steps
    sequences_in_global_batch = hp.device_batch_size * data_parallel_size
    assert hp.batch_size % sequences_in_global_batch == 0, "Invalid batch_size"
    grad_accum_steps = hp.batch_size // sequences_in_global_batch
    assert grad_accum_steps >= 1, "Invalid grad_accum_steps"

    print0(f"Global batch size: {hp.batch_size} sequences")
    print0(f"Per-device batch size: {hp.device_batch_size} sequences")
    print0(f"Sequence length: {hp.sequence_length} tokens")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0("=" * 80)

    train_glob = os.path.join(hp.data_dir, "fineweb_train_*.bin")
    val_glob = os.path.join(hp.data_dir, "fineweb_val_*.bin")

    print0(f"Training data: {train_glob}")
    print0(f"Validation data: {val_glob}")

    # Each data parallel rank gets different data
    # TP ranks must all use identical data
    train_loader = DistributedDataLoader(
        train_glob,
        hp.device_batch_size,
        hp.sequence_length,
        data_parallel_rank,
        data_parallel_size,
    )
    val_loader = DistributedDataLoader(
        val_glob,
        hp.device_batch_size,
        hp.sequence_length,
        data_parallel_rank,
        data_parallel_size,
    )

    print0(f"Training DataLoader: {len(train_loader.files)} files")
    print0(f"Validation DataLoader: {len(val_loader.files)} files")
    print0("=" * 80)

    # --- Model Initialization ---
    print0(f"Model dimension: {hp.model_dim}")
    print0(f"Number of layers: {hp.n_layer}")
    print0(f"Number of heads: {hp.n_head}")

    num_vocab = 50304  # nearest multiple of 128 for efficiency
    gpt_config = GPTConfig(
        sequence_len=hp.sequence_length,
        vocab_size=num_vocab,
        n_layer=hp.n_layer,
        n_head=hp.n_head,
        n_embd=hp.model_dim,
    )
    with torch.device("meta"):
        model = GPT(gpt_config)

    # Shard the model if using a device mesh
    # If replicate_mesh_grad_sync is True, FSDP will not handle data-parallel gradient sync
    # If replicate_mesh_grad_sync is False, we use Pytorch HSDP to do data-parallel gradient sync
    if device_mesh is not None:
        parallelize_gpt_model(
            model,
            device_mesh=device_mesh,
            dp_name=(None if hp.replicate_mesh_grad_sync else "dp"),
            fs_name="fs",
            tp_name="tp",
            fsdp_reshard_after_forward=(not cli_args.fast_fsdp),
        )
        raw_model = model

    # Move model to GPU
    model.to_empty(device="cuda")
    model.init_weights()
    if not cli_args.no_compile:
        model.compile()

    # If no device mesh, we are using DDP
    if device_mesh is None:
        model = DDP(model, device_ids=[data_parallel_rank])
        raw_model = model.module  # the underlying model

    # Ensure parameters are contiguous
    for i, p in enumerate(model.parameters()):
        if not p.is_contiguous():
            raise ValueError(f"Parameter {i} is not contiguous")

    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Total parameters: {num_params}")
    print0(f"Using torch.compile: {not cli_args.no_compile}")

    # Print model architecture
    print0(model)
    print0("=" * 80)

    # --- Optimizer Setup ---
    print0(f"Optimizer: {hp.optimizer}")
    print0(f"Scalar optimizer: {hp.scalar_opt}")
    print0(f"Base learning rate: {hp.lr}")

    optimizer = init_optimizer(
        model=raw_model,
        device_mesh=device_mesh,
        ddp_model=model if isinstance(model, DDP) else None,
        hp=hp,
        cli_args=cli_args,
    )

    # Learning rate scheduler
    def get_lr(it):
        warmup_iters = round(hp.warmup_ratio * hp.num_iterations)
        warmdown_iters = round(hp.warmdown_ratio * hp.num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= hp.num_iterations - warmdown_iters:
            return 1.0
        else:
            return (hp.num_iterations - it) / warmdown_iters

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    print0("=" * 80)

    # --- Logging initialization ---
    # Load hyperparameters and update with CLI arguments
    # Create a name to identify this run
    run_name = f"({hp.optimizer}+{hp.scalar_opt})"
    if "dion" in hp.optimizer:
        run_name += f"_sp={hp.rank_fraction}"
    if cli_args.dp_size is not None:
        run_name += f"_dp={cli_args.dp_size}_fs={cli_args.fs_size}_tp={cli_args.tp_size}_gradsync={cli_args.replicate_mesh_grad_sync}"
    if cli_args.wandb_job_name:
        run_name += f"_{cli_args.wandb_job_name}"

    # --- Set up checkpointing ---
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=hp.checkpoint_dir,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_id=None,
    )

    print0(f"Run name: {run_name}")
    print0(f"Debug mode: {cli_args.debug}")
    print0(f"Checkpoint directory: {hp.checkpoint_dir}")
    print0(
        f"Checkpoint frequency: {hp.checkpoint_freq if hp.checkpoint_freq > 0 else 'disabled'}"
    )

    # Load the latest checkpoint if it exists
    if hp.checkpoint_dir:
        checkpoint_manager.load(allow_missing=True)
        if checkpoint_manager.step is not None:
            print0(f"Resuming from step {checkpoint_manager.step}")
        else:
            print0("No previous checkpoint found, training model from scratch")
    else:
        # No checkpoint path provided
        print0("Training model from scratch")

    print0("=" * 80)

    # --- WandB initialization ---
    if not cli_args.no_wandb and not cli_args.debug:
        assert hp.wandb_project_name, "wandb project name is required"
        if MASTER_PROCESS:
            # Check if we already have a wandb ID from the checkpoint
            wandb_id = checkpoint_manager.wandb_id
            resume = "must" if wandb_id else "never"
            wandb.login(
                key=os.environ.get("WANDB_API_KEY"),
                host=os.environ.get("WANDB_HOST"),
                timeout=0,
            )
            wandb.init(
                project=hp.wandb_project_name,
                name=run_name,
                config=hp.__dict__,
                id=wandb_id,
                resume=resume,
            )
            # If we got a new ID, update the checkpoint manager
            checkpoint_manager.wandb_id = wandb.run.id

        # Broadcast wandb_id to all processes
        # Do this to ensure consistency of distributed checkpoint
        obj_list = [checkpoint_manager.wandb_id]
        dist.broadcast_object_list(obj_list, src=0)
        checkpoint_manager.wandb_id = obj_list[0]

    # --- Training Loop ---
    x, y = train_loader.next_batch()
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.time()

    # Use autocast for mixed precision
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    start_step = 0 if checkpoint_manager.step is None else checkpoint_manager.step + 1
    pbar = tqdm(total=hp.num_iterations, desc="Training", disable=not MASTER_PROCESS)
    pbar.update(start_step)
    for step in range(start_step, hp.num_iterations + 1):
        # Skip the first few steps for timing to avoid torch.compile overhead
        if step == 10:
            training_time_ms = 0
            torch.cuda.synchronize()
            t0 = time.time()
        timed_steps = (step - 10) if step > 10 else float("nan")

        # --- Validation ---
        last_step = step == hp.num_iterations
        if last_step or (hp.val_loss_every > 0 and step % hp.val_loss_every == 0):
            # Measure elapsed time for training
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)

            # Run validation
            model.eval()
            val_loader.reset()
            val_loss = torch.tensor(0.0, device=x.device)
            for _ in range(val_steps):
                with torch.no_grad():
                    x_val, y_val = val_loader.next_batch()
                    with autocast_ctx:
                        loss = model(x_val, y_val)
                    val_loss += loss

            # Average validation loss across devices
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss = val_loss.item() / val_steps
            log_message = (
                f"step:{step}/{hp.num_iterations} val_loss:{val_loss:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps):.2f}ms"
            )
            print0(log_message)
            if MASTER_PROCESS and not cli_args.no_wandb and not cli_args.debug:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "step": step,
                        "time/training_time_ms": training_time_ms,  # Log total elapsed training time in ms
                    }
                )
            pbar.set_postfix(val_loss=f"{val_loss:.4f}")

            # Restart training time for the next iteration
            torch.cuda.synchronize()
            t0 = time.time()

        if last_step:
            break

        model.train()
        for i in range(1, grad_accum_steps + 1):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()  # for logging
            loss = loss / grad_accum_steps
            x, y = train_loader.next_batch()

            # Turn off DDP grad sync if replicate_mesh_grad_sync is True
            ddp_no_sync = i < grad_accum_steps or hp.replicate_mesh_grad_sync
            if isinstance(model, DDP) and ddp_no_sync:
                with model.no_sync():
                    loss.backward()
            else:
                if isinstance(model, FSDPModule):
                    # Gradient accumulation for DP on top of FSDP
                    model.set_is_last_backward(i == grad_accum_steps)
                    if cli_args.fast_fsdp:
                        # Only reshard and reduce-scatter gradients upon the last backward pass
                        # Keep the entire unsharded model in memory during gradient accumulation
                        model.set_reshard_after_backward(i == grad_accum_steps)
                        model.set_requires_gradient_sync(i == grad_accum_steps)
                    else:
                        # FSDP always synchronizes sharded gradients via reduce-scatter
                        model.set_requires_gradient_sync(True)
                loss.backward()

        # Gradient norm
        grad_norm = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None]
        )

        # Optimizer step
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)

        # Approximate updated training time just before logging
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        if MASTER_PROCESS and not cli_args.no_wandb and not cli_args.debug:
            wandb.log(
                {
                    "train/loss": train_loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "step": step,
                    "time/training_time_ms": approx_time,  # Log approximate elapsed training time in ms
                }
            )
        if MASTER_PROCESS and cli_args.debug:
            print0(
                f"Step {step}: train_loss={train_loss.item():.4f}, grad_norm={grad_norm.item():.4f}"
            )
        pbar.update(1)
        pbar.set_postfix(train_loss=f"{train_loss.item():.4f}")

        if hp.checkpoint_freq > 0 and step % hp.checkpoint_freq == 0 and step > 0:
            # See if optimizer defines synchronize_for_checkpoint()
            if hasattr(optimizer, "synchronize_for_checkpoint"):
                # Dion with replicate_mesh_grad_sync will have decoupled optimizer states
                # Calling this is necessary to synchronize state across the replicate mesh
                # Otherwise, checkpoint results will not be consistent
                optimizer.synchronize_for_checkpoint()

            # Save a distributed checkpoint
            checkpoint_manager.save(step=step)

        torch.cuda.synchronize()
        t0 = time.time()  # reset timer after optimizer step

    pbar.close()
    print0(
        f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
