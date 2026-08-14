"""The sample configs in ``configs/`` must survive train.py's YAML merge.

``parse_cli_args`` copies YAML keys onto the argparse namespace with a bare
``setattr``, and ``override_args_from_cli`` then applies only the keys that
``Hyperparameters`` actually declares. A misspelled key is therefore silently
dropped: the run starts, uses the dataclass default, and nothing reports it.
``ortho_fraction`` typo'd in a config would train at 0.25 instead of 0.5 with no
warning at all, so these tests assert the merged hyperparameters, not just that
the YAML parses.
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


def test_dion3_config_matches_dion2_except_optimizer():
    """Dion3 is Dion2's selection plus NorMuon normalization, so the sample
    configs must stay in sync -- a hyperparameter tuned in one and not the other
    turns the pair into an unintended A/B."""
    dion2 = _merged_hyperparameters(CONFIG_DIR / "dion2_160m.yaml")
    dion3 = _merged_hyperparameters(CONFIG_DIR / "dion3_160m.yaml")

    assert dion3.optimizer == "dion3"
    assert dion2.optimizer == "dion2"

    differing = {
        field
        for field in type(dion2).__dataclass_fields__
        if field != "optimizer" and getattr(dion2, field) != getattr(dion3, field)
    }
    assert (
        not differing
    ), f"dion3_160m.yaml drifted from dion2_160m.yaml: {sorted(differing)}"


def test_dion3_config_sets_the_hyperparameters_nordion2_reads():
    """The values init_optimizer passes to NorDion2 must come from the file, not
    from Hyperparameters defaults -- that is what a dropped key would look like."""
    hp = _merged_hyperparameters(CONFIG_DIR / "dion3_160m.yaml")

    assert hp.ortho_fraction == 0.5
    assert hp.mu == 0.95
    assert hp.weight_decay == 0.01
    assert hp.adjust_lr == "spectral_norm"
    assert hp.lr == 0.02
    assert hp.scalar_opt == "adamw"
