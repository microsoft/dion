# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed

- **Breaking (install):** `gram-newton-schulz` and `quack-kernels` are no longer
  base dependencies. They moved to an optional `dion[gram-newton-schulz]` extra
  (alias `dion[gns]`), and are also excluded from the `dev` and `train` extras.
  This keeps the default install free of the transitive `nvidia-cutlass-dsl==4.4.2`
  pin, which conflicts with Flash-Attention-4 / Blackwell stacks built on cutlass
  `4.5.2`.

  **Action required:** if you run with `use_gram_newton_schulz=True`, install the
  extra (`pip install "dion[gns] @ git+https://github.com/microsoft/dion.git"`, or
  `pip install -e ".[gns]"` from a clone). Without it, optimizer construction now
  raises a clear `ImportError` at runtime instead of the kernels being silently
  present. Opting in re-introduces the cutlass `4.4.2` pin, so use a separate
  environment from FA4/Blackwell.
