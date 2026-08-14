"""The sample configs in ``configs/`` must survive train.py's YAML merge.

``parse_cli_args`` copies YAML keys onto the argparse namespace with a bare
``setattr``, and ``override_args_from_cli`` then applies only the keys that
``Hyperparameters`` actually declares. A misspelled key is therefore silently
dropped: the run starts, uses the dataclass default, and nothing reports it.
``ortho_fraction`` typo'd in a config would train at 0.25 instead of 0.5 with no
warning at all, so these tests assert the merged hyperparameters, not just that
the YAML parses.

They pin the merge, not the tuning. Nothing here asserts a particular
hyperparameter value or compares one config against another -- the sample
configs are free to diverge, and two of them sharing a value today is not a
property worth freezing.
"""

import sys
import pytest
import yaml

from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))

# Keys that are real train.py CLI flags but not Hyperparameters fields, so they
# are consumed off the namespace rather than merged into the dataclass.
CLI_ONLY_KEYS = {
    "config",
    "data_dir",
    "debug",
    "dp_size",
    "fast_fsdp",
    "fs_size",
    "no_compile",
    "no_triton",
    "no_wandb",
    "tp_size",
    "use_gram_newton_schulz",
    "use_polar_express",
    "wandb_job_name",
}


def _import_train():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return pytest.importorskip(
        "train", reason="train.py and its deps need the dion[train] extra"
    )


def _merged_hyperparameters(config_path):
    train = _import_train()
    with patch.object(sys, "argv", ["train.py", "--config", str(config_path)]):
        cli_args = train.parse_cli_args()
    return train.override_args_from_cli(train.Hyperparameters(), cli_args)


def test_configs_are_discovered():
    """Guard the glob itself, so an empty configs/ cannot vacuously pass."""
    assert CONFIGS, f"no configs found in {CONFIG_DIR}"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_keys_are_recognized(config_path):
    """Every key must reach either Hyperparameters or a CLI flag."""
    train = _import_train()
    with config_path.open("r") as f:
        yaml_cfg = yaml.safe_load(f)

    unknown = {
        k
        for k in yaml_cfg
        if k not in train.Hyperparameters.__dataclass_fields__
        and k not in CLI_ONLY_KEYS
    }
    assert (
        not unknown
    ), f"{config_path.name} has keys train.py ignores: {sorted(unknown)}"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_optimizer_is_dispatchable(config_path):
    """``optimizer`` must be a string ``init_optimizer`` knows how to build."""
    hp = _merged_hyperparameters(config_path)
    assert hp.optimizer in {
        "dion",
        "dion2",
        "dion3",
        "dion_reference",
        "dion_simple",
        "muon",
        "muon_reference",
        "nordion2",
        "normuon",
    }, f"{config_path.name} selects unknown optimizer {hp.optimizer!r}"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_values_survive_the_merge(config_path):
    """What the file declares is what the optimizer gets.

    Deliberately value-agnostic: it compares each key against the file rather
    than against a hardcoded number or against another config, so retuning any
    config cannot fail it. A silently dropped key still fails, since the merged
    value then falls back to the dataclass default instead of the file's.
    """
    train = _import_train()
    with config_path.open("r") as f:
        yaml_cfg = yaml.safe_load(f)

    hp = _merged_hyperparameters(config_path)
    for key, value in yaml_cfg.items():
        # override_args_from_cli skips None, leaving the dataclass default.
        if key in train.Hyperparameters.__dataclass_fields__ and value is not None:
            assert (
                getattr(hp, key) == value
            ), f"{config_path.name}: {key} did not survive the merge"
