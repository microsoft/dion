# Welcome to the Microsoft/Dion Codebase

This repository provides efficient implementations of orthonormal optimizers for distributed ML training.
You can find the following optimizers:
* [Muon](https://kellerjordan.github.io/posts/muon/)
* [MuonH](https://tinyurl.com/muonh)
* [Dion2](https://arxiv.org/abs/2512.16928) and [Dion](https://arxiv.org/pdf/2504.05295) (Dion is a legacy optimizer; we recommend using Dion2)
* [NorMuon](https://arxiv.org/abs/2510.05491) 


## Table of Contents
<details>
  <summary>Show/Hide</summary>

1. [Requirements](#-requirements)
1. [Quick Start](#-quick-start)
1. [Introduction](#introduction)
1. [Optimizers](#optimizers)
   * [MuonH](#muonh)
1. [Building Parameter Groups](#building-parameter-groups)
   * [Example Code](#example-code)
   * [Per-Head Newton-Schulz for Attention Projections](#per-head-newton-schulz-for-attention-projections)
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

This code is written for modern PyTorch (version 2.7 or newer) using DTensor-based parallelism. This includes FSDP2 with `fully_shard` and tensor parallelism (TP) with `parallelize_module`. Support for other distributed training APIs is not implemented.


## Quick Start

Our implementations are available as a `pip` package! Install to use in your project:

```bash
pip install git+https://github.com/microsoft/dion.git
```

Then in your code, you can use:

```python
from dion import Dion2, Muon, MuonH, NorMuon, Dion
```

Please carefully go through this readme for detailed instructions on using our optimizers. There are major differences compared to PyTorch built-in optimizers, such as `Adam`/`AdamW`.

### Running Our Sample Training Script

First clone this repo, then install dependencies for both Dion and training code:
```bash
git clone https://github.com/microsoft/dion.git
cd dion
pip install -e .[train]
```

Download pretokenized FineWeb dataset:
```bash
python data/cached_fineweb10B.py 30
```

### Distributed Data Parallel (DDP) Training

To train a GPT-small model using Dion2 with 4 GPUs (adjust as needed for your setup):
```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/dion2_160m.yaml
```
This will launch Distributed Data Parallel (DDP) training.

### Distributed Training: FSDP / TP / Hybrid Sharding

#### Fully Sharded Data Parallel (FSDP)

To enable FSDP, specify the FSDP group size using `--fs_size`:
```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/dion2_160m.yaml \
  --fs_size 4
```

This configuration trains a GPT-small model using Dion2 with FSDP sharding across all 4 GPUs (a single FSDP group of size 4).

#### Hybrid Sharded Data Parallel (HSDP)

To use Hybrid Sharded Data Parallel, where multiple FSDP groups are replicated using Data Parallel (DP), set `--fs_size` smaller than the total number of GPUs and specify the data parallel dimension via `--dp_size`:
```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/dion2_160m.yaml \
  --fs_size 2 \
  --dp_size 2
```

This configuration creates:
- **2 FSDP groups**, each spanning 2 GPUs
- **2-way data parallelism** across the FSDP groups
- **Total**: 4 GPUs with 2-way FSDP × 2-way DP


The product `dp_size × fs_size` must equal `world_size`. Any unspecified dimension defaults to 1.

#### Tensor Parallelism (TP)

**Note**: Currently, only Dion (our legacy implementation) supports Tensor Parallelism.

You can combine all three parallelism strategies (DP × FSDP × TP). For example, a 2 × 2 × 2 configuration across 8 GPUs:
```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/dion_160m.yaml \
  --dp_size 2 \
  --fs_size 2 \
  --tp_size 2
```

This configuration creates:
- **2-way data parallelism** (outer replication)
- **2-way FSDP**  
- **2-way tensor parallelism**  
- **Total**: 8 GPUs with 2-way DP × 2-way FSDP × 2-way TP

The product `dp_size × fs_size × tp_size` must equal `world_size`. Any unspecified dimension defaults to 1.


## Introduction

Optimization algorithms are essential to training neural networks, converting gradients into model weight updates to minimize loss. For many years, the method of choice has been [Adam](https://arxiv.org/abs/1412.6980)/[AdamW](https://arxiv.org/abs/1711.05101). However, recent work has shown that **orthonormal optimizers** can significantly accelerate model convergence. Check out blog posts by [Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon) and [Laker Newhouse](https://www.lakernewhouse.com/writing/muon-1) for more details.

The practical effectiveness of orthonormal optimizers was first demonstrated by [Muon](https://kellerjordan.github.io/posts/muon/) in the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), and has since been validated at scale by models such as [Kimi K2](https://arxiv.org/abs/2507.20534) and [GLM-4.5](https://z.ai/blog/glm-4.5). Muon implements orthonormalization via *Newton-Schulz iterations*, which relies on repeated matrix-matrix multiplications. However, large-scale training relies on model sharding, where weight matrices and optimizer states are distributed across multiple processes. As discussed by [Essential AI](https://www.essential.ai/blog/infra), orthonormalizing a sharded matrix with Newton-Schulz iterations involves the communication-intensive procedure of reconstructing the full matrices from their individual shards.

**Dion/Dion2** are our methods for building a **scalable, communication-efficient** optimizer. Like Muon, they compute matrix weight updates based on matrix orthonormalization and share similar practical benefits. The key difference is that Dion and Dion2 **shirnk the matrix before orthonormalization**, reducing both computational and communication costs. Dion uses power iteration to compute a low-rank approximation, while Dion2 applies a simple submatrix-selection procedure. To reduce information loss, both methods include an error-feedback mechanism that tracks the discrepancy between the original matrix and its compressed approximation.


## Optimizers

Our current implementations support the following parallelization techniques:

| Parallelization    | Dion | Dion2 | Muon | MuonH | NorMuon |
|--------------------|------|-------|------|-------|---------| 
| Single device      | Yes  |  Yes  | Yes  |  Yes  |   Yes   |
| PyTorch DDP        | Yes  |  Yes  | Yes  |  Yes  |   Yes   |
| PyTorch FSDP2      | Yes  |  Yes  | Yes  |  Yes  |   Yes   |
| PyTorch FSDP2 + TP | Yes  |  No   | No   |   No  |   No    |

For faster performance, these optimizers will process parameters in batches and interleave multiple batches to overlap compute with communication.

We include optimizer implementations in the `dion/` directory of this repo.
 
* `dion.py`: High-performance version of Dion. Depending on how each batch of matrices is sharded, we select the best communication patterns to compute Dion's orthonormal update. All-reduce operations may be split into reduce-scatter and all-gather across the batch dimension to more efficiently distribute work and avoid redundant computation.
* `muon.py`: High-performance version of Muon. For sharded matrices, all-to-all communication is used to simultaneously unshard and distribute a batch of matrices. For replicated matrices, Muon will distribute work across all devices and all-gather final results.
* `muonh.py`: Muon with a hyperball update. After Newton-Schulz orthonormalization (and optional update normalization), each matrix update is scaled relative to that matrix's Frobenius norm and projected back to its initial Frobenius norm.
* **`dion2.py`**: High-performance implementation of Dion2, using a similar all-to-all communication pattern for distributed orthonormalization. Only an α-fraction of the momentum matrix is communicated and orthonormalized, significantly reducing both communication overhead and computation cost.
* `normuon.py`: A variant of the Muon optimizer that introduces neuron-wise normalization to improve stability and convergence efficiency, modified to take similar arguments as `muon.py`. See [the paper](https://arxiv.org/abs/2510.05491) for more details.

### MuonH

MuonH combines Muon's momentum + Newton-Schulz direction with a hyperball projection step. Conceptually, the optimizer keeps each matrix parameter on a Frobenius sphere: after forming the Muon-style update direction, MuonH moves along that direction and projects back to the matrix's initial Frobenius norm. This implementation follows [Fantastic Pretraining Optimizers and Where to Find Them 2.1: Hyperball Optimization](https://tinyurl.com/muonh).

Compared with Muon, MuonH has a few important usage constraints and semantics:

* **Non-zero initialization is required.** MuonH initializes its hyperball radius from the parameter norm on the first optimizer step, so a zero-initialized matrix will raise an error.
* **Hyperball state is per matrix.** For ordinary 2D weights this is one radius per parameter. For packed matrix tensors such as MoE expert weights of shape `(n_experts, d_out, d_in)`, MuonH treats each trailing 2D matrix independently when `flatten=False`, so each expert gets its own momentum / normalization / hyperball radius.
* **`flatten=False` is the right mode for packed experts.** A tensor of shape `(..., d_out, d_in)` is treated as a batch of matrices. This is the intended mode for MoE expert MLP weights and other native 3D matrix batches.
* **`flatten=True` changes the semantics.** With `flatten=True`, MuonH flattens the parameter to one 2D matrix before orthonormalization and hyperball projection. Use this for convolution kernels only when you explicitly want the whole tensor treated as one matrix.
* **`num_heads` is not supported.** MuonH deliberately rejects `num_heads > 1`. If you want per-head MuonH behavior, represent heads as explicit matrices, e.g. a native 3D tensor of shape `(num_heads, head_dim, in_features)` or separate `Parameter`s.
* **Optional normalization matches NorMuon-style update normalization.** `normalization=None` disables it, `normalization="neuron"` normalizes along the last dimension, and `normalization="short_axis"` normalizes along the shorter matrix axis.
* **Matrix params do not use decoupled weight decay.** As with Muon, embeddings, LM heads, biases, and normalization parameters should remain on AdamW or Lion parameter groups.

Example:

```python
optimizer = MuonH(
    param_groups,
    lr=0.01,
    normalization="short_axis",
)
```

We also provide some reference implementations:

* `dion_reference.py`: An implementation without batching, communication overlapping, or split all-reduce. This version of Dion is intended to closely follow the algorithms as described in our [Dion paper](https://arxiv.org/pdf/2504.05295).
* `dion_simple.py`: A simplified illustration of the Dion update rule in a single Python function, provided for educational value.
* `muon_reference.py`: A version of Muon by [Moonshot AI](https://github.com/MoonshotAI/Moonlight), modified to take similar arguments as `muon.py`.



## Building Parameter Groups

Unlike typical PyTorch optimizers (e.g. `Adam`/`AdamW`), Dion and the Muon-family optimizers require separating your model's parameters into different groups (same in spirit as [Modula](https://docs.modula.systems/)). These orthonormal optimization algorithms are intended for matrix weights. For Muon, MuonH, and NorMuon, a parameter with shape `(..., d_out, d_in)` may also be treated as a batch of matrices when that matches the model semantics. Non-matrix parameters require a different scalar optimizer algorithm (element-wise updates) and may also use a different learning rate. We currently support Lion and AdamW.

The details of parameter grouping are dependent on model architecture and implementation. Therefore, we leave it up to you to categorize your model's parameters and create the necessary parameter groups.

* In transformer models and many other neural networks, most parameters are `nn.Linear` layers with two-dimensional weight matrices. These parameters should use Dion, Muon, MuonH, or NorMuon. A shape-dependent learning rate scale factor will be automatically applied for each matrix.
* Biases in `nn.Linear` layers (if used) are one-dimensional vectors, which must be placed into a separate parameter group from the weight matrices. Use Lion or AdamW.
* Normalization layers (e.g. `nn.LayerNorm`, `nn.RMSNorm`) may contain vectors of learnable weights. Use Lion or AdamW.
* Embedding layers (e.g. `nn.Embedding`) are stored as 2D tensors, but should be treated as a collection of 1D vectors using Lion or AdamW. (Warning: using Dion here will run without error, but will give poor performance.)
* Unembedding layers (e.g. LM head) are typically implemented as a `nn.Linear` layer, but shoud also be treated as a collection of 1D vectors. Furthermore, they should use a **smaller scaled learning rate**. It is very important to manually identify this layer and place it into its own parameter group, as it is otherwise indistinguishable from weight matrices!
(Warning: using Dion here will run without error, but will give poor performance.)
* Packed expert weights such as MoE MLP parameters with shape `(n_experts, d_out, d_in)` are a natural fit for Muon, MuonH, and NorMuon with `flatten=False`, since each expert is treated as an independent matrix.
* Convolution layers typically use parameter tensors with 3+ dimensions. These are currently not supported for Dion. Support for convolution layers in Muon-family optimizers is experimental and generally requires `flatten=True` so the whole kernel tensor is treated as one 2D matrix.
* MuonH requires non-zero matrix initialization. Do not assign zero-initialized output projections or other zero-initialized matrices to MuonH.

We summarize the above in this table. Let `d_in` be the input dimension of the unembedding layer. In transformer language models, this is the base dimension of the model.

| Type          | Example parameters                          | Optimizer `algorithm` | Learning rate `lr`     |
|---------------|---------------------------------------------|-----------------------|------------------------|
| Weight matrix | `nn.Linear.weight`                          | `"dion2"` / `"muon"`  | `lr`                   |
| Bias vector   | `nn.Linear.bias`                            | `"adamw"` / `"lion"`  | `lr`                   |
| Normalization | `nn.LayerNorm.weight`, `nn.LayerNorm.bias`  | `"adamw"` / `"lion"`  | `lr`                   |
| Embedding     | `nn.Embedding.weight`                       | `"adamw"` / `"lion"`  | `lr`                   |
| Unembedding   | `nn.Linear.weight` (must identify manually) | `"adamw"` / `"lion"`  | `lr / math.sqrt(d_in)` |

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
    dict(params=vector_params, algorithm="adamw"),
    dict(params=embed_params, algorithm="adamw"),
    dict(params=lm_head_params, algorithm="adamw", lr=lr / math.sqrt(model_dim))
]

optimizer = Dion2(
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
    dict(params=vector_params, algorithm="adamw"),
    dict(params=embed_params, algorithm="adamw", weight_decay=0),
    dict(params=lm_head_params, algorithm="adamw", lr=lr / math.sqrt(model_dim), weight_decay=0)
]
```

### Per-Head Newton-Schulz for Attention Projections

Attention Q / K / V / gate projections are typically stored as a single 2D `nn.Linear` weight of shape `(num_heads * head_dim, in_features)`, but semantically each head is an independent `(head_dim, in_features)` matrix. Orthogonalizing the fused matrix blends information across heads — often not what you want.

You can ask Dion2, Muon, or NorMuon to run Newton-Schulz independently per head without changing the model layout by setting `num_heads` on the parameter group:

```python
param_groups = [
    dict(params=attn_proj_params, num_heads=config.num_attention_heads),
    dict(params=other_matrix_params),
    dict(params=vector_params, algorithm="adamw"),
    ...
]
```

When `num_heads > 1`, the optimizer views each 2D weight as a batch of `num_heads` matrices of shape `(head_dim, in_features)` internally. The learning-rate adjustment (`spectral_norm` / `rms_norm`) is computed per-head, and Newton-Schulz runs on each head independently. With FSDP, sharding dim 0 along head boundaries (i.e. `num_heads % world_size == 0`) avoids the all-to-all that would otherwise be needed to assemble the fused matrix before NS.

Requirements: the parameter must be 2D, `num_heads` must divide dim 0, and when using FSDP it must also divide the world size. Place Q / K / V / gate projections in one group (axis-0 heads); O-projection heads are on axis 1 and are not covered by this option. MuonH does not implement this option; use an explicit 3D parameter or separate Parameters if you want per-head MuonH behavior.

## Distributed Training Configuration

For our efficient distributed optimizers to work correctly, they need information about the model's parallelization scheme. This is provided by passing `DeviceMesh` objects during optimizer construction.

### 1D Sharding Configuration (Dion2, Muon, MuonH, NorMuon)

Most optimizers in this codebase (Dion2, Muon, MuonH, NorMuon) currently support only 1D sharding. They accept a single 1D device mesh via the `distributed_mesh` argument and adapt their behavior based on how this mesh is used:

- **If the mesh is used for parameter sharding**: The optimizer efficiently unshards parameters using all-to-all communication
- **If the mesh is not used for sharding**: The optimizer distributes work across devices and all-gathers the final results
 
For a hybrid sharded data parallel (HSDP) configuration with both replicated and sharded dimensions, pass only the sharded sub-mesh to the optimizer:
```python
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(replicate_size, shard_size),
    mesh_dim_names=("replicate", "shard"),
)

# Apply HSDP with 2D device mesh
# Parameters are sharded across the 1st dim and replicated across the 0th dim
# https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
fully_shard(model, mesh=mesh)

# Pass only the sharded dimension to the optimizer
optimizer = Dion2(                     # or Muon or MuonH or NorMuon
    param_groups,
    distributed_mesh=mesh["shard"],    # 1D sub-mesh (sharded dimension only)
    ...
)
```
  

#### Flattened Meshes

When more advanced parallelism strategies are used (such as context parallel or expert parallel), it is common for multiple mesh dimensions to be "flattened" into a 1D sub-mesh for sharding. In this scenario, the flattened mesh needs to be given to the optimizer.

```python
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(dp_size, cp_size),
    mesh_dim_names=("dp", "cp")
)

# FSDP sharding applied across combined DP and CP meshes
fs_mesh = mesh["dp", "cp"]._flatten()
fully_shard(model, mesh=fs_mesh)

optimizer = Dion2(                  # or Muon or MuonH or NorMuon
    param_groups,
    distributed_mesh = fs_mesh,     # Sharded data parallel across flattened mesh 
    ...
)
```

#### Usage with DDP ProcessGroup

Training with DistributedDataParallel (DDP) is also supported. DDP uses PyTorch `ProcessGroup` instead of `DeviceMesh`, which is stored in the DDP-wrapped model's `process_group` field. Providing this to the optimizer will allow it to efficiently distribute work across all GPUs. If no `process_group` is provided, the optimizer will run in single-GPU mode, and every device in the DDP world will redundantly perform the same work.

```python
ddp_model = DistributedDataParallel(model, ...)

optimizer = Dion2(                              # or Muon or MuonH or NorMuon
    param_groups,
    distributed_mesh=ddp_model.process_group,
    ...
)
```



### 2D-Sharding Support for Dion
  
Our legacy optimizer Dion supports up to two sharded mesh dimensions and any number of data-parallel replicated mesh dimensions. The sharded meshes are referred to as `outer_shard_mesh` and `inner_shard_mesh`. Dion's internal optimizer states can be sharded over both meshes. During the update computation, Dion will orthonormalize a low-rank matrix that is replicated across `outer_shard_mesh`, but always remains sharded across `inner_shard_mesh`. Thus, the `inner_shard_mesh` is more communication-intensive and works best with intra-node tensor parallelism. Both sharding meshes must be one-dimensional.

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




## Best Practices

* **Dion/Dion2 rank fraction:** The most important Dion-specific hyperparameter is the *rank fraction*, which controls the amount of low-rank compression. Setting `rank_fraction=1.0` resulting in full-rank updates without any compression, similar to Muon. Empirically, it appears that larger models are more tolerant of low-rank compression. At 3B parameters, `rank_fraction=0.25` (1/4 rank) achieves nearly equivalent performance as full-rank, and we expect that 1/8, 1/16, and perhaps lower rank fractions will work well at 10B+ scale.
* **2D sharding:** If weights are sharded with both FSDP and TP, it is required that the sharding methods are applied to different matrix dimensions. The TP sharding dimension is controlled via `RowwiseParallel` and `ColwiseParallel`, but the FSDP sharding dimension needs to be manually specified when applied on top of TP. See `models/gpt_model.py` for an example of explicitly providing `fully_shard()` with per-parameter shard dimensions. Double-sharded matrices along the same dimension will raise an error in Dion.
* **Learning rate scaling:** Dion will automatically scale the provided learning rate by `sqrt(d_out / d_in)` for matrix parameters. Muon will apply the same scaling by default, but also supports the `0.2 * sqrt(max(d_in, d_out))` scale factor recommended by Moonshot AI. Our default scale factor is intended to induce a consistent change to activation vector values, which enables learning rate transfer across model size. See [Deriving Muon](https://jeremybernste.in/writing/deriving-muon) for more information.
* **Nesterov momentum:** In Muon, we set Nesterov momentum to `False` by default, as we observed better performance without it. Dion does not implement Nesterov momentum.
* **MuonH initialization and batching:** MuonH requires non-zero matrix initialization. For packed expert tensors such as `(n_experts, d_out, d_in)`, keep `flatten=False` so each expert is treated as its own matrix. For convolution kernels, `flatten=True` is usually the only sensible MuonH mode.


## Other Features


### Compressed Data-Parallel Gradient Sync for Dion

Dion is capable of *skipping the usual full-gradient all-reduce* by only synchronizing low-rank matrices instead. Depending on the rank fraction used, we can greatly compress the amount of communication needed while producing the exact same end result (up to numerical precision). This technique originates from PowerSGD---see [Vogels et al., 2019](https://arxiv.org/abs/1905.13727) for more details.

This feature is applicable across any replicated data-parallel axis for DDP and hybrid-sharded HSDP. It can be enabled or disabled using the `replicate_mesh_grad_sync` option.

* If `replicate_mesh_grad_sync` is True (default) and a `replicate_mesh` is provided, Dion will all-reduce the low-rank compressed states during the optimizer step.
* If `replicate_mesh_grad_sync` is False, Dion will expect that all data-parallel gradients have already been synchronized prior to the optimizer step.

Note that `replicate_mesh_grad_sync=True` results in *decoupled momentum*. The optimizer's internal momentum states will diverge across data-parallel processes. (Model weight updates always remain identical.) Before saving a checkpoint, you must explicitly tell Dion to synchronize internal states. See the [Checkpointing](#checkpointing) section for more details.

#### Usage with HSDP

Typically, hybrid sharding with `fully_shard()` uses a 2D device mesh. To use with Dion's compressed gradient synchronization, pass only the sharded sub-mesh to `fully_shard()`.

In other words, we don't let `fully_shard()` see the replicated mesh dimension, so it will not all-reduce gradients across it. Instead, Dion receives the replicated dimension as its `replicate_mesh` argument, and it will synchronize low-rank matrices during the optimizer step.

Note that if we choose to disable Dion's compressed gradient synchronization, we must make sure to provide the 2D mesh to `fully_shard()`.

| Option                       | `fully_shard()` device mesh | `replicate_mesh_grad_sync` | Optimizer states | Model weights       |
|------------------------------|-----------------------------|----------------------------|------------------|---------------------|
| Dion syncs compressed states | 1D shard sub-mesh           | `True`                     | Decoupled        | Always synchronous |
| FSDP syncs full gradients    | 2D hybrid-shard mesh        | `False`                    | Synchronous      | Always synchronous |

#### Example Codes

We provide example codes for compressed-DP sync under HSDP scenarios.

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

Dion when `replicate_mesh_grad_sync = True` requires synchronizing optimizer state before saving a checkpoint. This is because of Dion's decoupled momentum, where internal optimizer states will be different across the replicate mesh. Call the `synchronize_for_checkpoint()` function to explicitly perform an all-reduce of optimizer states. This ensures the consistency of distributed checkpoints, since typically each state will only be saved by one process along the replicated data-parallel mesh. This function will be a no-op if `replicate_mesh_grad_sync=False` or no replicate mesh is used.

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
# This is only for Dion with `replicate_mesh_grad_sync=True` so
# For other optimizers, this is not required
optimizer.synchronize_for_checkpoint()

# Save a distributed checkpoint
model_state_dict, opt_state_dict = get_state_dict(model, optimizer)
checkpoint = { "model": model_state_dict, "optimizer": opt_state_dict }
dcp.save(checkpoint, ...)
```

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
 
### Triton Kernels for Muon Newton-Schulz

Muon's Newton-Schulz iteration involves multiplying a matrix by its own transpose. The result is symmetric, so we can accelerate this computation by only computing half of the output and mirroring the result across the diagonal. We implemented this technique with Triton kernels in `optimizers/newton_schulz_triton.py`.

Triton kernels can be enabled in Muon with the option `use_triton=True`. Note that compiling and tuning the kernels may take several minutes when it is first run.


# Citation 

If you use Dion/Dion2 in your research, please cite:

```bash
@article{ahn2025dion2,
  title={Dion2: A Simple Method to Shrink Matrix in Muon},
  author={Ahn, Kwangjun and Amsel, Noah and Langford, John},
  journal={arXiv preprint 2512.16928},
  year={2025}
}
``` 

```bash
@article{ahn2025dion,
  title={Dion: Distributed Orthonormalized Updates},
  author={Ahn, Kwangjun and Xu, Byron and Abreu, Natalie and Langford, John},
  journal={arXiv preprint: 2504.05295},
  year={2025}
}
``` 
