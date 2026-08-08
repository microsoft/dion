"""Unit tests for the env-gated NaN capture wrapper in
``megabatch_orthogonalize_async``.

The wrapper exists so a user hitting a numerical blow-up in their Newton-
Schulz backend (e.g. microsoft/dion#76, where quack-kernels 0.4.1 produces
NaN params on certain hardware/shapes) can dump the offending input and
output to disk for offline reproduction. The check is local per rank, so
the tests below run single-rank without NCCL.
"""

import os
import pytest
import torch

from dion.megabatch_base import megabatch_orthogonalize_async


def _nan_ns(X, epsilon=None):
    return torch.full_like(X, float("nan"))


def _identity_ns(X, epsilon=None):
    return X.clone()


def _drive(gen):
    # ``megabatch_orthogonalize_async`` is a generator; the sharded path yields
    # at alltoall, but the ``N>1, no process_group`` path runs straight through
    # via ``return value``. Iterating to completion exercises whichever path
    # the args select; the StopIteration carries the result.
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_capture_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("DION_NAN_CAPTURE", raising=False)
    monkeypatch.setenv("DION_NAN_CAPTURE_DIR", str(tmp_path))

    U = [torch.randn(3, 4) for _ in range(2)]
    _drive(
        megabatch_orthogonalize_async(
            U,
            comm_dim=None,
            device_rank=0,
            world_size=1,
            process_group=None,
            newton_schulz_func=_nan_ns,
            flatten=False,
            epsilon=torch.tensor(1e-7),
            global_comm_dim_size=None,
        )
    )

    # No env var set -> no dumps even though the NS func returned NaN.
    assert list(tmp_path.glob("dion_nan_capture_*.pt")) == []


def test_capture_dumps_on_non_finite_output(tmp_path, monkeypatch):
    monkeypatch.setenv("DION_NAN_CAPTURE", "1")
    monkeypatch.setenv("DION_NAN_CAPTURE_DIR", str(tmp_path))
    # Don't raise; we want to inspect the dump.
    monkeypatch.setenv("DION_NAN_CAPTURE_RAISE", "0")

    U = [torch.randn(3, 4) for _ in range(2)]
    _drive(
        megabatch_orthogonalize_async(
            U,
            comm_dim=None,
            device_rank=0,
            world_size=1,
            process_group=None,
            newton_schulz_func=_nan_ns,
            flatten=False,
            epsilon=torch.tensor(1e-7),
            global_comm_dim_size=None,
        )
    )

    dumps = list(tmp_path.glob("dion_nan_capture_rank0_*.pt"))
    assert len(dumps) == 1, f"expected 1 dump, got {dumps}"
    payload = torch.load(dumps[0], weights_only=False)
    assert payload["rank"] == 0
    assert payload["shape"] == (2, 3, 4)  # stacked across N=2
    assert torch.isfinite(payload["input"]).all()
    assert torch.isnan(payload["output"]).all()
    assert payload["epsilon"] == pytest.approx(1e-7)


def test_capture_raises_by_default_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("DION_NAN_CAPTURE", "1")
    monkeypatch.setenv("DION_NAN_CAPTURE_DIR", str(tmp_path))
    monkeypatch.delenv("DION_NAN_CAPTURE_RAISE", raising=False)

    U = [torch.randn(3, 4) for _ in range(2)]
    with pytest.raises(RuntimeError, match="non-finite Newton-Schulz output"):
        _drive(
            megabatch_orthogonalize_async(
                U,
                comm_dim=None,
                device_rank=0,
                world_size=1,
                process_group=None,
                newton_schulz_func=_nan_ns,
                flatten=False,
                epsilon=torch.tensor(1e-7),
                global_comm_dim_size=None,
            )
        )

    # Even when raising, the dump should be written first.
    assert len(list(tmp_path.glob("dion_nan_capture_rank0_*.pt"))) == 1


def test_capture_no_dump_when_finite(tmp_path, monkeypatch):
    monkeypatch.setenv("DION_NAN_CAPTURE", "1")
    monkeypatch.setenv("DION_NAN_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("DION_NAN_CAPTURE_RAISE", "0")

    U = [torch.randn(3, 4) for _ in range(2)]
    _drive(
        megabatch_orthogonalize_async(
            U,
            comm_dim=None,
            device_rank=0,
            world_size=1,
            process_group=None,
            newton_schulz_func=_identity_ns,
            flatten=False,
            epsilon=torch.tensor(1e-7),
            global_comm_dim_size=None,
        )
    )

    # NS output was finite -> no dump regardless of env.
    assert list(tmp_path.glob("dion_nan_capture_*.pt")) == []


def test_capture_filename_includes_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("DION_NAN_CAPTURE", "1")
    monkeypatch.setenv("DION_NAN_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("DION_NAN_CAPTURE_RAISE", "0")

    # device_rank=3 simulates a non-zero rank in a multi-rank job; each rank
    # must write under its own filename so concurrent writes from different
    # ranks don't collide on a shared filesystem.
    U = [torch.randn(3, 4) for _ in range(2)]
    _drive(
        megabatch_orthogonalize_async(
            U,
            comm_dim=None,
            device_rank=3,
            world_size=1,
            process_group=None,
            newton_schulz_func=_nan_ns,
            flatten=False,
            epsilon=torch.tensor(1e-7),
            global_comm_dim_size=None,
        )
    )

    dumps = list(tmp_path.glob("dion_nan_capture_rank3_*.pt"))
    assert len(dumps) == 1
    assert "rank3" in os.path.basename(dumps[0])
