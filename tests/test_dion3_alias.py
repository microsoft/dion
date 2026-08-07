"""Dion3 is an additive alias for NorDion2, not a separate optimizer.

These tests pin the three things that make the alias an alias rather than a
fork: the class identity (so ``isinstance`` and checkpoints do not care which
name was used), the internal ``algorithm="nordion2"`` param-group identifier
(which keys optimizer state and is written into ``state_dict()``), and
``train.py``'s ``--optimizer`` string accepting both names. The existing
NorDion2 tests in ``test_optimizers.py`` cover the behavior itself.
"""

import argparse
import pytest
import sys
import torch

from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_train():
    """Import the top-level ``train`` module, or skip if it is unavailable.

    ``train.py`` sits at the repo root and is deliberately not part of the
    installed package (``setup.py`` packages only ``dion*``), so it is importable
    only when the repo root is on ``sys.path``. That happens by accident under
    ``python -m pytest`` from the repo root, but not under a bare ``pytest`` or
    from any other working directory -- so put the repo root on the path
    explicitly rather than depending on how the suite was invoked.

    ``importorskip`` then covers train.py's optional imports as a set (``wandb``,
    ``yaml``, ``tqdm``), which ship with the ``dion[train]`` extra and not with
    ``dion[dev]``. Guarding on any single one of them would let the others turn a
    dev-only install into an error instead of a skip.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return pytest.importorskip(
        "train", reason="train.py and its deps need the dion[train] extra"
    )


def test_dion3_is_nordion2():
    """The alias must be the same class object, not a subclass or a copy."""
    from dion import Dion3, NorDion2

    assert Dion3 is NorDion2


def test_dion3_module_and_package_exports_agree():
    """``dion.dion3`` and ``dion`` must export the same object."""
    import dion
    from dion.dion3 import Dion3
    from dion.nordion2 import NorDion2

    assert Dion3 is NorDion2
    assert dion.Dion3 is Dion3


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_dion3_runs_and_keeps_nordion2_algorithm():
    """Constructing via the alias yields a NorDion2 with unchanged state keys.

    ``algorithm`` is a param-group key compared against ``self._algo_name`` in
    ``DistributedOrthoBase.step`` and saved in ``state_dict()["param_groups"]``,
    so the alias must not introduce a second identifier.
    """
    from dion import Dion3, NorDion2

    torch.manual_seed(42)
    params = [torch.nn.Parameter(torch.randn(64, 128, device=DEVICE))]
    opt = Dion3(params, lr=0.01)
    assert isinstance(opt, NorDion2)
    assert opt.param_groups[0]["algorithm"] == "nordion2"

    params[0].grad = torch.randn_like(params[0])
    opt.step()
    assert opt.state[params[0]].keys() >= {"momentum", "variance_neuron"}
    assert opt.state_dict()["param_groups"][0]["algorithm"] == "nordion2"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.parametrize("optimizer_name", ["nordion2", "dion3"])
def test_train_optimizer_string_builds_nordion2(optimizer_name):
    """``--optimizer dion3`` must select the same optimizer as ``nordion2``.

    Mirrors ``init_optimizer``'s DDP path, which only reads
    ``ddp_model.process_group``, so a stub stands in for a real DDP wrapper.
    """
    train = _import_train()
    from dion import NorDion2

    class _StubModel(torch.nn.Module):
        """Minimal stand-in exposing the attributes init_optimizer reads."""

        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.Linear(64, 64, bias=False)
            self.transformer.wte = torch.nn.Embedding(32, 64)
            self.lm_head = torch.nn.Linear(64, 32, bias=False)

    hp = train.Hyperparameters(optimizer=optimizer_name, scalar_opt="adamw")
    cli_args = argparse.Namespace(
        use_gram_newton_schulz=False, no_triton=True, use_polar_express=True
    )
    ddp_model = argparse.Namespace(process_group=None)

    opt = train.init_optimizer(
        model=_StubModel().to(DEVICE),
        device_mesh=None,
        ddp_model=ddp_model,
        hp=hp,
        cli_args=cli_args,
    )
    assert type(opt) is NorDion2
    assert opt.param_groups[0]["algorithm"] == "nordion2"
