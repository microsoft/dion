# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- `CudaGraphOptimizer.release()` drops the captured graph (and restarts the warmup,
  so a later capture is not taken with cold buffers). On the sharded path the graph
  holds the captured megabatch all-to-all, and `dist.destroy_process_group()` blocks
  while those NCCL ops are alive, so a framework that keeps optimizers alive past the
  end of training otherwise hangs at process exit. Previously the only ways to drop a
  graph were `add_param_group` / `load_state_dict`, both with side effects. Those two
  now route through `release()` as well, so they also synchronize and destroy the
  graph rather than dropping a reference and leaving the NCCL ops to a refcount.

- A per-group `weight_decay` supplied as a Tensor (`weight_decay=torch.tensor(0.01)`)
  is carried as a persistent device tensor like the learning rate, so filling it in
  place drives a CUDA-graph-captured step. This makes schedule-coupled weight decay
  (Defazio 2506.02285, AdamC / Muon-C), which rescales `weight_decay` every step,
  usable under capture — previously it was silently frozen at its capture-step value.
  Opt-in via the Tensor: on the AdamW scalar path a live weight decay cannot ride
  `torch._fused_adamw_`, whose `weight_decay` is a float in every overload, so it is
  applied as a separate pass and rounds `X` once more. A float `weight_decay` keeps
  the previous fused path unchanged. The opt-in latches, so it also works when the
  Tensor is assigned to the group after construction, and a later float assignment
  refills the tensor instead of dropping the group back to a baked value. Opting in
  after the graph exists is too late to honor, and `CudaGraphOptimizer` raises instead
  of letting the schedule silently do nothing.

### Fixed

- `adamw_update_foreach` dropped the cautious-weight-decay correction entirely when it
  was given a device-tensor `weight_decay` on the legacy (`step=`) path: the kernel is
  handed `weight_decay=0` there, and the correction was scaled by that float, so
  `cautious_wd=True` silently degraded to plain weight decay.

- CUDA-graph capture now refuses to run while any `requires_grad` parameter still has
  `.grad=None`. The step only touches parameters with a gradient, so capture freezes
  the participating set: a parameter starved of gradients through the warmup steps (a
  modality cadence that runs text-only batches, expert routing that skips an expert)
  was left out of the graph and silently stopped being updated for the rest of the
  run. Freeze such parameters explicitly with `requires_grad=False`, or raise
  `warmup_steps` past the schedule that starves them.

### Changed

- The FSDP2 row-sharded `selection_scope` default (both `Dion2` and `NorDion2`)
  is now `"local"` (per-shard top-k) again, reverting the `"global"` default from
  #98 while keeping that PR's `global_select_size` padding-correctness fix intact.
  `"local"` has cheaper per-shard communication (the win grows with matrix size)
  and, in a 1B / 8-way-FSDP A/B, converges indistinguishably from `"global"`.
  Note that `"local"` selection is sharding/world-size dependent, so default runs
  are no longer bit-reproducible across world sizes; pass
  `selection_scope="global"` for exact, layout-invariant selection (preferable at
  larger scale or higher shard counts, where an earlier 1.5B A/B saw `"local"`
  trail). No effect off the row-sharded path, where the two coincide.

- AdamW scalar fallback now uses the base learning rate for LM head parameters,
  while Lion fallback keeps the `1 / sqrt(d_in)` LM-head scaling. This affects
  shipped `configs/*_160m.yaml` runs, which set `scalar_opt: adamw`.

- **Breaking (install):** `gram-newton-schulz` and `quack-kernels` are no longer
  base dependencies. They moved to an optional `dion[gram-newton-schulz]` extra
  (alias `dion[gns]`), and are also excluded from the `dev` and `train` extras.
  This keeps the default install free of the heavy Gram Newton-Schulz GPU stack
  (and its transitive `nvidia-cutlass-dsl` pin).

  **Action required:** if you run with `use_gram_newton_schulz=True`, install the
  extra (`pip install "dion[gns] @ git+https://github.com/microsoft/dion.git"`, or
  `pip install -e ".[gns]"` from a clone). Without it, optimizer construction now
  raises a clear `ImportError` at runtime instead of the kernels being silently
  present.

- Bumped the optional `dion[gns]` extra to `gram-newton-schulz==0.1.6`
  (`quack-kernels==0.5.0`, `nvidia-cutlass-dsl==4.5.2` unchanged). `0.1.6` turns off
  quack's autotuner in the Gram Newton-Schulz kernel backend (gram-newton-schulz
  PR #22), fixing a reserved-GPU-memory leak that laddered to OOM under sharded
  training. It also lands the GNS algorithm-selection/transpose refactor
  (gram-newton-schulz PR #18); the orthogonalization math is unchanged.

- Bumped the optional `dion[gns]` extra to `gram-newton-schulz==0.1.5`
  (`quack-kernels==0.5.0`). This moves its transitive `nvidia-cutlass-dsl` pin from
  `4.4.2` to `4.5.2`, matching current Flash-Attention-4 / Blackwell stacks, so the
  extra no longer conflicts with them.
