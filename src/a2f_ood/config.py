"""Validated, backward-compatible experiment configuration."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml

TASK_TYPES = {"binary", "multiclass", "multilabel", "regression"}
DEFAULTS = {
    "experiment": {"name": "a2f_ood", "seed": 42, "output_dir": "outputs/a2f_ood"},
    "data": {"smiles_column": "smiles", "label_columns": ["label"], "group_column": None,
             "task_type": "binary", "num_classes": None, "split_column": None,
             "split_strategy": "preassigned", "split_fractions": [0.8, 0.1, 0.1],
             "group_strategy": "column", "valid_groups": [], "test_groups": [],
             "overlap_policy": "error",
             "max_smiles_length": 256, "random_smiles_per_molecule": 2,
             "invalid_smiles_policy": "skip"},
    "model": {"hidden_dim": 256, "graph_layers": 4, "smiles_layers": 4,
              "attention_heads": 4, "dropout": 0.1, "max_atom_count": 256,
              "use_cross_modal_decoder": True},
    "fusion": {"mode": "adaptive", "atom_alignment": True,
               "bidirectional_cross_attention": True, "reliability_gate": True,
               "unimodal_auxiliary_heads": True},
    "loss": {"lambda_atom_alignment": 0.02, "lambda_serialization_consistency": 0.1,
             "lambda_cross_modal": 0.1, "lambda_generation": 0.01,
             "lambda_unimodal": 0.05, "use_group_dro": False,
             "group_dro_step_size": 0.1, "group_dro_update_mode": "epoch",
             "group_dro_max_weight_ratio": 3.0, "group_dro_strength": 0.25,
             "group_dro_warmup_epochs": 5, "group_dro_ramp_epochs": 5,
             "supervised_warmup_epochs": 2, "temperature": 0.1,
             "max_alignment_atoms": 512, "regression_loss": "mse",
             "auxiliary_warmup_epochs": 10, "classification_weighting": "none",
             "binary_class_weights": None},
    "training": {"epochs": 100, "batch_size": 64, "learning_rate": 0.0001,
                 "weight_decay": 0.00001, "patience": 15, "num_workers": 0,
                 "device": "auto", "gradient_clip": 1.0, "cpu_threads": 4,
                 "monitor": "task_loss", "monitor_mode": "min", "monitor_min_delta": 0.001,
                 "learning_rate_warmup_epochs": 3, "lr_scheduler": "plateau",
                 "lr_scheduler_patience": 3, "lr_scheduler_factor": 0.5,
                 "min_learning_rate": 0.000001, "validation_smiles_views": 3},
    "evaluation": {"metrics": ["roc_auc", "average_precision"],
                   "report_group_metrics": True, "random_smiles_views": 5,
                   "generate_smiles": False},
}


def _merge(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8-sig") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping")
    config = _merge(DEFAULTS, raw)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Expected --set section.key=value, got {override!r}")
        dotted, value = override.split("=", 1)
        keys = dotted.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"Unknown config section: {dotted}")
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"Unknown config option: {dotted}")
        target[keys[-1]] = yaml.safe_load(value)
    root = next((p for p in config_path.parents if (p / "pyproject.toml").is_file()), config_path.parent)
    config["_project_root"] = str(root)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def resolve_path(config: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config.get("_project_root", ".")) / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    missing = set(DEFAULTS) - config.keys()
    if missing:
        raise ValueError(f"Missing top-level config sections: {sorted(missing)}")
    data, model, training = config["data"], config["model"], config["training"]
    if not data.get("csv_path") or not data.get("smiles_column"):
        raise ValueError("data.csv_path and data.smiles_column are required")
    if not isinstance(data.get("label_columns"), list) or not data["label_columns"]:
        raise ValueError("data.label_columns must be a nonempty list")
    if len(set(data["label_columns"])) != len(data["label_columns"]):
        raise ValueError("data.label_columns must be unique")
    if data.get("task_type") not in TASK_TYPES:
        raise ValueError(f"data.task_type must be one of {sorted(TASK_TYPES)}")
    if data["task_type"] in {"binary", "multiclass"} and len(data["label_columns"]) != 1:
        raise ValueError("binary/multiclass tasks require exactly one label column")
    if data["task_type"] == "multiclass" and (not isinstance(data.get("num_classes"), int) or data["num_classes"] < 2):
        raise ValueError("data.num_classes must be >= 2 for multiclass")
    if data.get("invalid_smiles_policy") not in {"skip", "error"}:
        raise ValueError("data.invalid_smiles_policy must be skip or error")
    if data.get("split_strategy") not in {"preassigned", "random", "scaffold", "size", "environment"}:
        raise ValueError("Unknown data.split_strategy")
    if data.get("group_strategy") not in {"column", "scaffold", "size", "none"}:
        raise ValueError("Unknown data.group_strategy")
    if data.get("overlap_policy") not in {"error", "allow_official"}:
        raise ValueError("data.overlap_policy must be error or allow_official")
    if data.get("overlap_policy") == "allow_official" and data["split_strategy"] != "preassigned":
        raise ValueError("allow_official is supported only for externally preassigned splits")
    fractions = data.get("split_fractions", [])
    if len(fractions) != 3 or any(v <= 0 for v in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("data.split_fractions must be three positive values summing to one")
    if data.get("random_smiles_per_molecule", 0) < 1 or data.get("max_smiles_length", 0) < 4:
        raise ValueError("SMILES views must be >= 1 and max length >= 4")
    for name in ("hidden_dim", "attention_heads", "graph_layers", "smiles_layers", "max_atom_count"):
        if model.get(name, 0) <= 0:
            raise ValueError(f"model.{name} must be positive")
    if model["hidden_dim"] % model["attention_heads"]:
        raise ValueError("model.hidden_dim must be divisible by model.attention_heads")
    if not 0 <= model.get("dropout", -1) < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if config["fusion"].get("mode", "adaptive") not in {"adaptive", "concat", "graph", "smiles"}:
        raise ValueError("fusion.mode must be adaptive/concat/graph/smiles")
    for name in ("batch_size", "epochs", "patience", "learning_rate", "cpu_threads"):
        if training.get(name, 0) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if training.get("num_workers", -1) < 0:
        raise ValueError("training.num_workers must be >= 0")
    if training.get("monitor_mode") not in {"min", "max"}:
        raise ValueError("training.monitor_mode must be min or max")
    for key, value in config["loss"].items():
        if key.startswith("lambda_") and (not math.isfinite(value) or value < 0):
            raise ValueError(f"loss.{key} must be nonnegative")
    loss = config["loss"]
    if loss.get("classification_weighting") not in {"none", "balanced"}:
        raise ValueError("loss.classification_weighting must be none or balanced")
    weights = loss.get("binary_class_weights")
    if weights is not None:
        if (not isinstance(weights, list) or len(weights) != len(data["label_columns"])
                or any(not isinstance(pair, list) or len(pair) != 2 for pair in weights)
                or any(not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0
                       for pair in weights for w in pair)):
            raise ValueError("loss.binary_class_weights must be positive [num_tasks, 2] negative/positive weights")
    if loss.get("group_dro_update_mode") not in {"batch", "epoch"}:
        raise ValueError("loss.group_dro_update_mode must be batch or epoch")
    if not 0 <= loss.get("group_dro_strength", -1) <= 1:
        raise ValueError("loss.group_dro_strength must be in [0, 1]")
    if not math.isfinite(loss["group_dro_max_weight_ratio"]) or loss["group_dro_max_weight_ratio"] < 1:
        raise ValueError("loss.group_dro_max_weight_ratio must be finite and >= 1")
    if not math.isfinite(loss["group_dro_step_size"]) or loss["group_dro_step_size"] < 0:
        raise ValueError("loss.group_dro_step_size must be finite and nonnegative")
    for section, options in ((loss, ("supervised_warmup_epochs", "auxiliary_warmup_epochs",
                                    "group_dro_warmup_epochs", "group_dro_ramp_epochs")),
                             (training, ("learning_rate_warmup_epochs", "lr_scheduler_patience"))):
        for key in options:
            if not isinstance(section[key], int) or section[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
    if training["lr_scheduler"] not in {"none", "plateau"}:
        raise ValueError("training.lr_scheduler must be none or plateau")
    if not 0 < training["lr_scheduler_factor"] < 1:
        raise ValueError("training.lr_scheduler_factor must be in (0, 1)")
    if not 0 < training["min_learning_rate"] <= training["learning_rate"]:
        raise ValueError("training.min_learning_rate must be positive and <= learning_rate")
    if not math.isfinite(training["monitor_min_delta"]) or training["monitor_min_delta"] < 0:
        raise ValueError("training.monitor_min_delta must be finite and nonnegative")
    if not isinstance(training["validation_smiles_views"], int) or training["validation_smiles_views"] < 1:
        raise ValueError("training.validation_smiles_views must be a positive integer")
    if not math.isfinite(training["gradient_clip"]) or training["gradient_clip"] <= 0:
        raise ValueError("training.gradient_clip must be finite and positive")
    if config["loss"].get("temperature", 0) <= 0 or config["loss"].get("max_alignment_atoms", 0) < 1:
        raise ValueError("Alignment temperature and max_alignment_atoms must be positive")
    if config["evaluation"].get("random_smiles_views", 0) < 1:
        raise ValueError("evaluation.random_smiles_views must be >= 1")

