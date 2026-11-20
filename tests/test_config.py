from pathlib import Path

import pytest

from a2f_ood.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    config = load_config(ROOT / "configs" / "a2f_ood.yaml")
    assert config["data"]["task_type"] == "binary"
    assert config["model"]["hidden_dim"] % config["model"]["attention_heads"] == 0


def test_config_rejects_invalid_task_type():
    config = load_config(ROOT / "configs" / "a2f_ood.yaml")
    config["data"]["task_type"] = "unknown"
    with pytest.raises(ValueError, match="task_type"):
        validate_config(config)


@pytest.mark.parametrize("section,key,value", [
    ("loss", "group_dro_strength", 1.1),
    ("loss", "group_dro_max_weight_ratio", .5),
    ("loss", "group_dro_update_mode", "unknown"),
    ("loss", "supervised_warmup_epochs", -1),
    ("loss", "classification_weighting", "batch"),
    ("loss", "binary_class_weights", [[1, 0]]),
    ("loss", "binary_class_weights", [[1, 1], [1, 1]]),
    ("training", "lr_scheduler_factor", 1),
    ("training", "validation_smiles_views", 0),
])
def test_unstable_schedule_settings_are_rejected(section, key, value):
    config = load_config(ROOT / "configs" / "a2f_ood.yaml")
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)
