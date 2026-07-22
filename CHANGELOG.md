# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- New `RMNP` optimizer (Row-Momentum Normalized Preconditioning,
  [arXiv:2603.20527](https://arxiv.org/abs/2603.20527)). RMNP replaces Muon's
  Newton-Schulz orthogonalization with a single row-wise (input-dimension) ℓ₂
  normalization of the momentum update, `row_normalize(V) = (diag(V Vᵀ))^{-1/2} V`,
  dropping the per-iteration cost from `O(mn·min(m,n))` to `O(mn)` for an `m×n`
  weight while, in the paper's experiments, matching Muon-level quality
  (orthogonalization and row-wise ℓ₂ normalization are asymptotically equivalent
  for the Transformer). It reuses Muon's momentum, distributed assembly, and
  weight update, so it inherits the same FSDP2/DDP sharding support. Learning
  rate is not rescaled by default (`adjust_lr=None`), matching the paper.

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
