# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed

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

- Bumped the optional `dion[gns]` extra to `gram-newton-schulz==0.1.5`
  (`quack-kernels==0.5.0`). This moves its transitive `nvidia-cutlass-dsl` pin from
  `4.4.2` to `4.5.2`, matching current Flash-Attention-4 / Blackwell stacks, so the
  extra no longer conflicts with them.
