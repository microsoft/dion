# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- `NorDion2` is now also available as `Dion3` (`from dion import Dion3`, or
  `from dion.dion3 import Dion3`), and `train.py` accepts `--optimizer dion3`
  alongside `--optimizer nordion2`. NorDion2 is the third-generation Dion update
  — Dion2's submatrix selection and error feedback with NorMuon's per-neuron
  normalization — so it is exposed under the dion3 name too. This is purely
  additive: `Dion3` is an alias, not a subclass (`Dion3 is NorDion2`), so the
  two names are interchangeable including for `isinstance`. Parameter groups
  keep using `algorithm="nordion2"` under either name — that string keys
  optimizer state and megabatch grouping and is written into
  `state_dict()["param_groups"]`, so existing checkpoints and param groups load
  unchanged. A sample config is included at `configs/dion3_160m.yaml`. Its
  hyperparameters are carried over from `configs/dion2_160m.yaml` as a starting
  point, not a translation: `mu` feeds Dion2's error-feedback decay but
  NorDion2's momentum, NorDion2's `muon_beta2` has no config key, and per-neuron
  normalization redistributes the update across rows, so expect to retune.

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

- With `flatten=True`, orthogonalization megabatches now preserve each
  parameter's matrix geometry: stacked convolution weights become
  `[N, out, prod(rest)]`, not `[N, numel]`. This fixes layer mixing for
  Conv1d/2d/3d and stacked Linear weights, and makes Newton-Schulz agree with
  the existing learning-rate adjustment computed from each parameter's shape.
  Singleton dispatch, `flatten=False`, and row splitting are unchanged.
  Correct convolution orthogonalization can cost more than the erroneous
  layer-mixing operation; previous step times are not equivalent-work baselines.

- `NorDion2` / `Dion3` failed to compile on PyTorch 2.13 as soon as a model had more
  than one parameter shape group (reported in #115).
  `nordion2_normalize_selected_stacked` ran the gather of the selected variance rows,
  the per-row reduction, and the scatter back into the full buffer in one compiled
  graph; after automatic dynamic-shape generalization 2.13's inductor emitted the
  scatter epilogue referencing a temp defined only inside the preceding `tl.range`
  reduction body, and `optimizer.step()` raised `NameError: tmp19 is not defined`
  (upstream pytorch/pytorch#194490, a 2.12 -> 2.13 regression; both use Triton 3.7.1).
  Reproducing it takes a shape group holding at least two parameters — dynamo
  specializes a batch dim of 1, so one-parameter groups stayed on the static path —
  plus a second distinct row/column shape and a reduction long enough not to compile
  as a persistent kernel. The scatter is now compiled as its own graph, which makes
  the invalid fusion unreachable while keeping dynamic shapes; `dynamic=False` also
  avoids it but recompiles per shape and hits the recompile limit (#23) past eight
  shape groups. Where the fused form did compile it was also badly slower: its
  generalized kernel measured ~60x the split one (68 ms vs 1.1 ms per call at N=8,
  8192x2048 on an H100), so multi-shape models gain throughput here, while a model
  with a single shape group pays one extra graph entry, ~40 us of host dispatch per
  call. The split reassociates the fp32 reduction, so `U` differs from the fused
  kernel by up to 1.9e-6 (`V` is bit-exact) — the same decomposition
  `test_normalize_selected_stacked_matches_unfused` already compared at
  `atol=rtol=1e-5`. `Dion2`, `NorMuon` and `Muon` were not affected.

- NorMuon gave `split_sizes` row blocks the wrong learning rate on the FSDP2 sharded
  path (reported in #111). The per-block correction `adjust(block) / adjust(full)` was
  multiplied into each block right after Newton-Schulz, but NorMuon's normalization
  then divides by `sqrt(V)` — an EMA of that same scaled update, so the factor cancels
  — and rescales to the Frobenius norm of its input, which reintroduces the factor only
  where that rescale is per block. Under FSDP the rescale is shard-local, so any shard
  spanning a block boundary applied one blended factor to all its rows: with a fused
  QKV weight `(4096, 512, 512) x 4096` and `adjust_lr="spectral_norm"`, the K and V
  blocks ran at up to ~1.8x their intended learning rate, silently and with no warning,
  matching the "converges faster, then diverges" report. The scales are now applied
  after the normalization, per row block of each rank's shard (block boundaries are
  derivable locally because `split_sizes` already requires dim 0 to be divisible by the
  world size), so the learning rate is exact per block on every path and no longer
  depends on the normalization commuting with it. Only the norm-preserving rescale
  remains shard-local under FSDP. Muon was never affected — it has no normalization
  step after the scales. Unsharded behavior is unchanged in form; numerically it shifts
  by one rounding, because the scale multiply moved off the bf16 Newton-Schulz output
  onto the fp32 normalized update (~1e-4 relative on a 48x16 test case with the default
  bf16 `polar_express`, ~1e-8 with an fp32 orthogonalizer) — the new path is the more
  accurate one. The variance buffer `V` now tracks the *unscaled* update, so its scale
  differs from checkpoints written by earlier versions by
  `(adjust(block) / adjust(full))**2` per block. That difference cancels between the
  division by `sqrt(V)` and the norm-preserving rescale only where the rescale runs
  per block — the same condition that made the old scale placement work unsharded and
  fail under FSDP. So an old checkpoint resumes exactly on the unsharded path (to the
  `+1e-8` in `sqrt(V) + 1e-8`, `sqrt(V)` being of order 1e-2), while on the row-sharded
  path the shard-local rescale cannot undo a *per-block* `V` scale and the first steps
  after a resume reproduce the pre-fix blended learning rate — measured 0.98x / 1.17x /
  1.00x of intended on a `(4096, 512, 512) x 4096` QKV at world size 8. `V` is not
  versioned, so that transient is not correctable; it decays with the `V` EMA over
  ~`1 / (1 - muon_beta2)` steps (~20 at the default 0.95) and is bounded by the behavior
  the checkpoint was already being trained under, so no migration is required.

- `CudaGraphOptimizer.load_state_dict` named its parameter `sd`, so every distributed
  checkpoint resume raised `TypeError: got an unexpected keyword argument 'state_dict'`.
  torch's DCP calls it by keyword (`_load_optim_state_dict` does
  `_state_dict_fn(optim, "load_state_dict")(state_dict=...)`), so a wrapped optimizer
  could train but never resume under FSDP2. The parameter now matches
  `torch.optim.Optimizer`, and a test asserts the wrapper's parameter *names and kinds*
  match the base class for every overridden method (the set is discovered, not listed),
  since substituting for an Optimizer means callers may use any name the base class
  documents and may pass it positionally or by keyword.

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
