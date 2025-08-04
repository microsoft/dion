# Dion: Distributed Orthonormal Updates

This repository provides efficient implementations of Dion and Muon optimizers for distributed ML training.
 
* See our paper for more information on Dion: https://arxiv.org/pdf/2504.05295.
* See the original blog post on Muon: https://kellerjordan.github.io/posts/muon/

## Table of Contents
<details>
  <summary>Show/Hide</summary>

1. [Requirements](#-requirements)
1. [Quick Start](#-quick-start)
1. [Introduction](#introduction)
1. [Optimizers](#optimizers)
1. [Building Parameter Groups](#building-parameter-groups)
   * [Example Code](#example-code)
1. [Distributed Training Configuration](#distributed-training-configuration)
   * [Flattened Meshes](#flattened-meshes)
   * [Device Mesh for Muon](#device-mesh-for-muon)
   * [Usage with DDP ProcessGroup](#usage-with-ddp-processgroup)
1. [Compressed Data-Parallel Gradient Sync](#compressed-data-parallel-gradient-sync)
   * [Usage with HSDP](#usage-with-hsdp)
   * [Example Code](#example-code-1)
   * [Usage with DDP](#usage-with-ddp)
   * [Checkpointing](#checkpointing)
1. [Best Practices](#best-practices)
1. [Experimental Features](#experimental-features)
   * [Mixed Precision Dion](#mixed-precision-dion)
   * [Accelerating Optimization Step for Lower Ranks](#accelerating-optimization-step-for-lower-ranks)
   * [Triton Kernels for Muon Newton-Schulz](#triton-kernels-for-muon-newton-schulz)
1. [Citation](#citation)

</details>


## Requirements

This code is written for modern PyTorch (version 2.7 or newer) using DTensor-based parallelism. This includes FSDP2 with `fully_shard`and tensor parallelism (TP) with `parallelize_module`. Support for other distributed training APIs is not guaranteed.


## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Download pretokenized FineWeb dataset:
```bash
python data/cached_fineweb10B.py 16
```

### Distributed Data Parallel (DDP) Training

To train a GPT-small model using Dion with 8 GPUs (adjust as needed for your setup):
```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_160m.yaml
```
This will launch Distributed Data Parallel (DDP) training.

### Advanced FSDP / TP / Hybrid Sharded Training

To enable more advanced distributed strategies such as Fully Sharded Data Parallel (FSDP) and Tensor Parallelism (TP), you can specify the configuration in the `dion_160m.yaml` file: 

```yaml
# — Distributed training —
dp_size: 2      # data‐parallel size
fs_size: 2      # FSDP size
tp_size: 2      # tensor‐parallel size
```

This example sets up a hybrid configuration with DDP × FSDP × TP = 2 × 2 × 2.

Alternatively, you can override these values directly from the command line. All three values must be explicitly given, but a size may be set to `1` to omit a parallelism dimension.

```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_160m.yaml \
  --dp_size 2 --fs_size 2 --tp_size 2
```

####  Example Weights & Biases (wandb) Plots

With the appropriate configuration, you should be able to reproduce the results shown in the [validation curves for GPT-small](https://microsoft-research.wandb.io/t-gmagakyan/dion-exp/reports/Validation-curves-for-GPT-small--VmlldzoxNjk5OA?accessToken=52e6z4d18yfkewz1bawlkmwc2m91al9ssa7rpwvnx1f1xa66j15lr7x315wj2kys).


## Introduction

Optimization algorithms are essential to training neural networks, converting gradients into model weight updates to minimize loss. For many years, the state-of-the-art method has been Adam/AdamW. However, recent work has shown that **orthonormalized matrix updates** can significantly accelerate model convergence. See [Bernstein and Newhouse, 2025](https://openreview.net/forum?id=hErdffTsLu) for a theoretical justification.

The practical effectiveness of orthonormal updates was first demonstrated by [Muon](https://kellerjordan.github.io/posts/muon/) in the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), and has since been validated at scale by models such as [Kimi K2](https://arxiv.org/abs/2507.20534) and [GLM-4.5](https://z.ai/blog/glm-4.5). Muon implements orthonormalization via *Newton-Schulz iterations*, which relies on repeated matrix-matrix multiplications. However, large-scale training relies on model sharding, where weight matrices and optimizer states are distributed across multiple processes. As discussed by [Essential AI](https://www.essential.ai/blog/infra), orthonormalizing a sharded matrix with Newton-Schulz iterations involves the communication-intensive procedure of reconstructing the full matrices from their individual shards.

**Dion** is our approach for a more **scalable and communication-efficient** optimizer. Like Muon, it computes orthonormal weight updates and has the same benefits of faster model convergence. The difference is that Dion uses an alternative orthonormalization method based on amortized power iteration (in the style of [PowerSGD](https://arxiv.org/pdf/1905.13727)), which can be applied directly on sharded matrices. Furthermore, Dion introduces a *rank fraction* hyperparameter, allowing for compute and communication reduction via low-rank compression. To mitigate information loss, Dion adds an error feedback mechanism that captures the difference between the original matrix and its low-rank approximation.


## Optimizers

Our main implementations of Dion (`dion.py`) and Muon (`muon.py`) support the following parallelization techniques:

| Parallelization    | Dion | Muon |
|--------------------|------|------|
| Single device      | Yes  | Yes  |
| PyTorch DDP        | Yes  | Yes  |
| PyTorch FSDP2      | Yes  | Yes  |
| PyTorch FSDP2 + TP | Yes  | No   |

For faster performance, both of these optimizers will process parameters in batches and interleave multiple batches to overlap compute with communication.

We include Dion, Muon, and several alternative implementations of the optimizers in the `optimizers/` directory of this repo.
 
* `dion.py`: High-performance version of Dion. Depending on how each batch of matrices is sharded, we select the best communication patterns to compute Dion's orthonormal update. All-reduce operations may be split into reduce-scatter and all-gather across the batch dimension to more efficiently distribute work and avoid redundant computation.
* `muon.py`: High-performance version of Muon. For sharded matrices, all-to-all communication is used to simultaneously unshard and distribute a batch of matrices. For replicated matrices, Muon will distribute work across all devices and all-gather final results.
* `dion_reference.py`: An implementation without batching, communication overlapping, or split all-reduce. This version of Dion is intended to closely follow the algorithms as described in our [Dion paper](https://arxiv.org/pdf/2504.05295).
* `dion_simple.py`: A simplified illustration of the Dion update rule in a single Python function, provided for educational value.
* `muon_reference.py`: A version of Muon by [Moonshot AI](https://github.com/MoonshotAI/Moonlight), modified to take similar arguments as `muon.py`.


## Building Parameter Groups

Unlike typical PyTorch optimizers (e.g. `Adam`/`AdamW`), Dion and Muon require separating your model's parameters into different groups (same in spirit as [Modula](https://docs.modula.systems/)). These orthonormal optimization algorithms are only applicable to two-dimensional matrix weights. Non-matrix parameters require a different scalar optimizer algorithm (element-wise updates) and may also use a different learning rate. We currently support Lion and AdamW.

The details of parameter grouping are dependent on model architecture and implementation. Therefore, we leave it up to you to categorize your model's parameters and create the necessary parameter groups.

* In transformer models and many other neural networks, most parameters are `nn.Linear` layers with two-dimensional weight matrices. These parameters should use Dion or Muon. A shape-dependent learning rate scale factor will be automatically applied for each matrix.
* Biases in `nn.Linear` layers (if used) are one-dimensional vectors, which must be placed into a separate parameter group from the weight matrices. Use Lion or AdamW.
* Normalization layers (e.g. `nn.LayerNorm`, `nn.RMSNorm`) may contain vectors of learnable weights. Use Lion or AdamW.
* Embedding layers (e.g. `nn.Embedding`) are stored as 2D tensors, but should be treated as a collection of 1D vectors using Lion or AdamW. (Warning: using Dion here will run without error, but will give poor performance.)
* Unembedding layers (e.g. LM head) are typically implemented as a `nn.Linear` layer, but shoud also be treated as a collection of 1D vectors. Furthermore, they should use a **smaller scaled learning rate**. It is very important to manually identify this layer and place it into its own parameter group, as it is otherwise indistinguishable from weight matrices!
(Warning: using Dion here will run without error, but will give poor performance.)
* Convolution layers typically use parameter tensors with 3+ dimensions. These are currently not supported for Dion. Support for convolution layers in Muon is experimental, and can be enabled with the option `flatten=True` to automatically flatten them to 2D matrices when computing the optimizer update.

We summarize the above in this table. Let `d_in` be the input dimension of the unembedding layer. In transformer language models, this is the base dimension of the model.

| Type          | Example parameters                          | Optimizer `algorithm` | Learning rate `lr`     |
|---------------|---------------------------------------------|-----------------------|------------------------|
| Weight matrix | `nn.Linear.weight`                          | `"dion"` / `"muon"`   | `lr`                   |
| Bias vector   | `nn.Linear.bias`                            | `"lion"` / `"adamw"`  | `lr`                   |
| Normalization | `nn.LayerNorm.weight`, `nn.LayerNorm.bias`  | `"lion"` / `"adamw"`  | `lr`                   |
| Embedding     | `nn.Embedding.weight`                       | `"lion"` / `"adamw"`  | `lr`                   |
| Unembedding   | `nn.Linear.weight` (must identify manually) | `"lion"` / `"adamw"`  | `lr / math.sqrt(d_in)` |

We emphasize again that **particular care** needs to be taken with **embedding and unembedding layers**. They must be isolated from ordinary matrix parameters, and the unembedding layer furthermore should use a scaled learning rate. Merely checking the dimensions of a parameter (such as `if p.ndim == 2`) or the type of the module (such as `if isinstance(module, nn.Linear)`) **is not sufficient** to identify these special parameters. This is why we require manual parameter group creation.

The optimizer cannot tell if a given parameter is a weight matrix, embedding, or unembedding, because they are all two-dimensional tensors. You will not receive any errors if these parameters are incorrectly grouped with matrix weights!

It is permissible to place biases, embeddings, and normalization parameters into a single parameter group if they share the same hyperparameters. A good rule of thumb is that when training a transformer model, the optimizer should have at least 3 parameter groups---one for the weight matrices, one for the LM head, and one for everything else.

### Example Code

```python
class TransformerModel(nn.Module):
    embedding = nn.Embedding(vocab_dim, model_dim)
    blocks = nn.ModuleList([TransformerBlock(...) for _ in range(10)])
    lm_head = nn.Linear(model_dim, vocab_dim)

model = TransformerModel()

# Note that the following will vary depending on your model architecture
matrix_params = list(p for p in model.blocks.parameters() if p.ndim == 2)
vector_params = list(p for p in model.blocks.parameters() if p.ndim != 2)
embed_params  = list(model.embedding.parameters())
lm_head_params= list(model.lm_head.parameters())

param_groups = [
    dict(params=matrix_params),  # will default to "dion" algorithm
    dict(params=vector_params, algorithm="lion"),
    dict(params=embed_params, algorithm="lion"),
    dict(params=lm_head_params, algorithm="lion", lr=lr / math.sqrt(model_dim))
]

optimizer = Dion(
    param_groups,
    lr=lr,  # used for all param groups except for lm_head_params
    weight_decay=0.1,  # default setting for all param groups
    ...
)
```

Additional hyperparameters may be specified on a per-parameter-group basis to override the defaults. For example, we may set the weight decay to 0 for only the embedding and unembedding parameters by modifying the above example:
```python
param_groups = [
    dict(params=matrix_params),
    dict(params=vector_params, algorithm="lion"),
    dict(params=embed_params, algorithm="lion", weight_decay=0),
    dict(params=lm_head_params, algorithm="lion", lr=lr / math.sqrt(model_dim), weight_decay=0)
]
```


## Distributed Training Configuration

In order for our efficient distributed optimizers to work, they must know about the parallelization scheme for training your model. This is done by passing in `DeviceMesh` objects when constructing the optimizer.

### Device Mesh for Dion

Dion supports up to two sharded mesh dimensions and any number of data-parallel replicated mesh dimensions. The sharded meshes are referred to as `outer_shard_mesh` and `inner_shard_mesh`. Dion's internal optimizer states can be sharded over both meshes. During the update computation, Dion will orthonormalize a low-rank matrix that is replicated across `outer_shard_mesh`, but always remains sharded across `inner_shard_mesh`. Thus, the `inner_shard_mesh` is more communication-intensive and works best with intra-node tensor parallelism. Both sharding meshes must be one-dimensional.

Unused meshes may be omitted or given as `None`. If only one sharding dimension is used (e.g. only FSDP without TP), we recommend providing it as the `outer_shard_mesh`. Dion will execute a faster single-device orthonormalization routine in this case, since the input matrix to be orthonormalized will not be sharded.

```python
# Example with a 3D mesh
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(dp_size, fs_size, tp_size),
    mesh_dim_names=("dp", "fs", "tp")
)

optimizer = Dion(
    param_groups,
    replicate_mesh = mesh["dp"],    # Replicated data parallel
    outer_shard_mesh = mesh["fs"],  # Sharded data parallel
    inner_shard_mesh = mesh["tp"],  # Tensor parallel
    ...
)
```

### Flattened Meshes

When more advanced parallelism strategies are used (such as context parallel or expert parallel), it is common for multiple mesh dimensions to be "flattened" into a 1D sub-mesh for sharding. In this scenario, the flattened mesh needs to be given to Dion.

```python
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(dp_size, cp_size, tp_size),
    mesh_dim_names=("dp", "cp", "tp")
)

# FSDP sharding applied across combined DP and CP meshes
fs_mesh = mesh["dp", "cp"]._flatten()
fully_shard(model, mesh=fs_mesh)

optimizer = Dion(
    param_groups,
    replicate_mesh = None,          # No replicated data parallel used
    outer_shard_mesh = fs_mesh,     # Sharded data parallel across flattened mesh
    inner_shard_mesh = mesh["tp"],  # Tensor parallel
    ...
)
```

### Device Mesh for Muon

Muon uses different device mesh arguments from Dion.

Our implementation of Muon takes a single 1D device mesh as a generic `distributed_mesh` argument. If this mesh is used for sharding parameters, Muon will efficiently perform unsharding using all-to-all. If this mesh is not used for sharding, Muon will distribute work across this mesh and all-gather the final results.

2D sharding is not supported by Muon---use Dion instead. For hybrid-sharded data parallel, with a replicated mesh dimension and a sharded dimension, pass only the sharded sub-mesh to Muon.

```python
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(replicate_size, shard_size),
    mesh_dim_names=("replicate", "shard"),
)

# Hybrid sharded data parallel with 2D device mesh
fully_shard(model, mesh=mesh)

optimizer = Muon(
    param_groups,
    distributed_mesh = mesh["shard"],  # 1D sub-mesh
    ...
)
```

### Usage with DDP ProcessGroup

Training with DistributedDataParallel (DDP) is also supported. DDP uses PyTorch `ProcessGroup` instead of `DeviceMesh`, which is stored in the DDP-wrapped model's `process_group` field. Providing this to the optimizer will allow it to efficiently distribute work across all GPUs. If no `process_group` is provided, the optimizer will run in single-GPU mode, and every device in the DDP world will redundantly perform the same work.

```python
ddp_model = DistributedDataParallel(model, ...)

optimizer = Dion(
    param_groups,
    replicated_mesh=ddp_model.process_group,
    ...
)
# - or -
optimizer = Muon(
    param_groups,
    distributed_mesh=ddp_model.process_group,
    ...
)
```


## Compressed Data-Parallel Gradient Sync

Dion is capable of *skipping the usual full-gradient all-reduce* by only synchronizing low-rank matrices instead. Depending on the rank fraction used, we can greatly compress the amount of communication needed while producing the exact same end result (up to numerical precision). This technique originates from PowerSGD---see [Vogels et al., 2019](https://arxiv.org/abs/1905.13727) for more details.

This feature is applicable across any replicated data-parallel axis for DDP and hybrid-sharded HSDP. It can be enabled or disabled using the `replicate_mesh_grad_sync` option.

* If `replicate_mesh_grad_sync` is True (default) and a `replicate_mesh` is provided, Dion will all-reduce the low-rank compressed states during the optimizer step.
* If `replicate_mesh_grad_sync` is False, Dion will expect that all data-parallel gradients have already been synchronized prior to the optimizer step.

Note that `replicate_mesh_grad_sync=True` results in *decoupled momentum*. The optimizer's internal momentum states will diverge across data-parallel processes. (Model weight updates always remain identical.) Before saving a checkpoint, you must explicitly tell Dion to synchronize internal states. See the [Checkpointing](#checkpointing) section for more details.

### Usage with HSDP

Typically, hybrid sharding with `fully_shard()` uses a 2D device mesh. To use with Dion's compressed gradient synchronization, pass only the sharded sub-mesh to `fully_shard()`.

In other words, we don't let `fully_shard()` see the replicated mesh dimension, so it will not all-reduce gradients across it. Instead, Dion receives the replicated dimension as its `replicate_mesh` argument, and it will synchronize low-rank matrices during the optimizer step.

Note that if we choose to disable Dion's compressed gradient synchronization, we must make sure to provide the 2D mesh to `fully_shard()`.

| Option                       | `fully_shard()` device mesh | `replicate_mesh_grad_sync` | Optimizer states | Model weights       |
|------------------------------|-----------------------------|----------------------------|------------------|---------------------|
| Dion syncs compressed states | 1D shard sub-mesh           | `True`                     | Decoupled        | Always synchronous |
| FSDP syncs full gradients    | 2D hybrid-shard mesh        | `False`                    | Synchronous      | Always synchronous |

### Example Code

```python
# ------------------------------------------------------------
#  Mode 1: Dion handles DP sync (low-rank compressed matrices)
# ------------------------------------------------------------
mesh = init_device_mesh("cuda", (dp, fs), ("dp", "fs"))

fully_shard(model, mesh=mesh["fs"])  # DP mesh not provided here

opt = Dion(
    param_groups,
    replicate_mesh           = mesh["dp"],  # Dion still gets DP mesh
    outer_shard_mesh         = mesh["fs"], 
    replicate_mesh_grad_sync = True         # Dion will synchronize low-rank matrices
)

# ------------------------------------------------------------
#  Mode 2: FSDP handles DP sync (classic full gradients)
# ------------------------------------------------------------
mesh = init_device_mesh("cuda", (dp, fs), ("dp", "fs"))

fully_shard(model, mesh=mesh["dp", "fs"])  # FSDP hybrid sharding

opt = Dion(
    param_groups,
    replicate_mesh           = mesh["dp"],    
    outer_shard_mesh         = mesh["fs"], 
    replicate_mesh_grad_sync = False        # Dion expects gradients already synced
)
```

### Usage with DDP

To use compressed gradient synchronization with DDP, always run the model with the `no_sync()` context.

```python
ddp_model = DistributedDataParallel(model, ...)

optimizer = Dion(
    param_groups,
    replicate_mesh=ddp_model.process_group,
    replicate_mesh_grad_sync=True,
    ...
)

for data in dataloader:
    # Always run with no_sync(), not just for gradient accumulation
    with ddp_model.no_sync():
        loss = ddp_model(data)
        loss.backward()

    optimizer.step()
    model.zero_grad()
```

### Checkpointing

Dion requires synchronizing optimizer state before saving a checkpoint. Because of Dion's decoupled momentum, internal optimizer states will be different across the replicate mesh. Call the `synchronize_for_checkpoint()` function to explicitly perform an all-reduce of optimizer states. This ensures the consistency of distributed checkpoints, since typically each state will only be saved by one process along the replicated data-parallel mesh. This function will be a no-op if `replicate_mesh_grad_sync=False` or no replicate mesh is used.

If model parameters are `DTensor` type, the optimizer states will also be `DTensor`s. Checkpoints should be saved using [torch.distributed.checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html).

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

optimizer = Dion(
    param_groups,
    replicate_mesh = mesh["dp"],
    replicate_mesh_grad_sync=True,
    ...
)

# Train the model
loss = model(data)
loss.backward()
optimizer.step()
model.zero_grad()

# Call this before checkpointing
optimizer.synchronize_for_checkpoint()

# Save a distributed checkpoint
model_state_dict, opt_state_dict = get_state_dict(model, optimizer)
checkpoint = { "model": model_state_dict, "optimizer": opt_state_dict }
dcp.save(checkpoint, ...)
```


## Best Practices

* **Dion rank fraction:** The most important Dion-specific hyperparameter is the *rank fraction*, which controls the amount of low-rank compression. Setting `rank_fraction=1.0` resulting in full-rank updates without any compression, similar to Muon. Empirically, it appears that larger models are more tolerant of low-rank compression. At 3B parameters, `rank_fraction=0.25` (1/4 rank) achieves nearly equivalent performance as full-rank, and we expect that 1/8, 1/16, and perhaps lower rank fractions will work well at 10B+ scale.
* **Lion vs. AdamW:** We have found that Lion performs better than AdamW for optimizing scalar parameters when used with Dion/Muon for orthonormal matrix updates.
* **2D sharding:** If weights are sharded with both FSDP and TP, it is required that the sharding methods are applied to different matrix dimensions. The TP sharding dimension is controlled via `RowwiseParallel` and `ColwiseParallel`, but the FSDP sharding dimension needs to be manually specified when applied on top of TP. See `models/gpt_model.py` for an example of explicitly providing `fully_shard()` with per-parameter shard dimensions. Double-sharded matrices along the same dimension will raise an error in Dion.
* **Learning rate scaling:** Dion will automatically scale the provided learning rate by `sqrt(d_out / d_in)` for matrix parameters. Muon will apply the same scaling by default, but also supports the `0.2 * sqrt(max(d_in, d_out))` scale factor recommended by Moonshot AI. Our default scale factor is intended to induce a consistent change to activation vector values, which enables learning rate transfer across model size. See [Deriving Muon](https://jeremybernste.in/writing/deriving-muon) for more information.
* **Nesterov momentum:** In Muon, we set Nesterov momentum to `False` by default, as we observed better performance without it. Dion does not implement Nesterov momentum.


## Experimental Features

### Mixed Precision Dion

By default, Dion will initialize its optimizer states to use the same data type as the model's parameters. The `DionMixedPrecisionConfig` class may be used to specify custom data types. In preliminary experiments, we have found that using `torch.bfloat16` for Dion's optimizer states can reduce memory use and speed up computation with no impact on training stability.

```python
from dion import Dion, DionMixedPrecisionConfig

dion_mixed_precision_config = DionMixedPrecisionConfig(
    momentum_dtype=torch.bfloat16,
    Q_dtype=torch.bfloat16,  # for the low-rank Q matrix
    variance_dtype=torch.float32,  # only used for AdamW
)
optimizer = Dion(
    ...
    mixed_precision_config=dion_mixed_precision_config,
    ...
)
```

### Faster Dion for lower ranks

After a few warmup iterations, the expensive QR decomposition can be replaced with the Cholesky QR (CQR) algorithm, leading to **2X** optimization step speedups. CQR is faster but less numerically stable. We have found that after some initial warmup period, the input matrix for orthogonalization becomes relatively well-conditioned. If Cholesky decomposition fails, we fall back to the standard QR decomposition procedure.

To try out the CQR accelerated configuration:
```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_efficient_160m.yaml
```

After the training you should be able to reproduce the second plot in [validation curves for GPT-small](https://microsoft-research.wandb.io/t-gmagakyan/dion-exp/reports/Validation-curves-for-GPT-small--VmlldzoxNjk5OA?accessToken=52e6z4d18yfkewz1bawlkmwc2m91al9ssa7rpwvnx1f1xa66j15lr7x315wj2kys).

### Triton Kernels for Muon Newton-Schulz

Muon's Newton-Schulz iteration involves multiplying a matrix by its own transpose. The result is symmetric, so we can accelerate this computation by only computing half of the output and mirroring the result across the diagonal. We implemented this technique with Triton kernels in `optimizers/newton_schulz_triton.py`.

Triton kernels can be enabled in Muon with the option `use_triton=True`. Note that compiling and tuning the kernels may take several minutes when it is first run.


# Citation 

If you use Dion in your research, please cite:

```bash
@article{ahn2025dion,
  title={Dion: Distributed Orthonormalized Updates},
  author={Ahn, Kwangjun and Xu, Byron and Abreu, Natalie and Langford, John},
  journal={arXiv preprint: 2504.05295},
  year={2025}
}
``` 
