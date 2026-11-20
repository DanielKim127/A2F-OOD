"""Train A2F-OOD with train-only vocabulary, early stopping and held-out reports."""
from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import math
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import yaml

from a2f_ood.config import load_config, resolve_path
from a2f_ood.data import molecular_feature_schema
from a2f_ood.engine import (build_model, evaluate_model, get_device, load_checkpoint,
                           make_loader, prepare_data, seed_everything, train_epoch,
                           write_json, write_predictions)
from a2f_ood.losses import GroupDRO, fit_binary_class_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train A2F-OOD graph/SMILES molecular predictor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE", help="Override a YAML option; repeat as needed")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run full chemistry preprocessing, splits, train vocab and batch checks")
    parser.add_argument("--selection-only", action="store_true",
                        help="Train and export validation results without evaluating the test set")
    return parser


def epoch_objectives(config: dict, epoch: int) -> tuple[dict, float]:
    """Stage auxiliaries and robust reweighting after supervised learning starts."""
    result = copy.deepcopy(config)
    loss = result["loss"]
    supervised = int(loss.get("supervised_warmup_epochs", 0))
    auxiliary_age = max(0, epoch - supervised)
    warmup = int(loss.get("auxiliary_warmup_epochs", 0))
    scale = min(1.0, auxiliary_age / warmup) if warmup else float(auxiliary_age > 0)
    for key in loss:
        if key.startswith("lambda_"):
            loss[key] *= scale
    dro_age = max(0, epoch - int(loss.get("group_dro_warmup_epochs", 0)))
    dro_ramp = int(loss.get("group_dro_ramp_epochs", 0))
    dro_scale = min(1.0, dro_age / dro_ramp) if dro_ramp else float(dro_age > 0)
    loss["group_dro_strength"] = float(loss.get("group_dro_strength", 1.0)) * dro_scale
    return result, scale


def main(argv: Sequence[str] | None = None, *, prepared_data: tuple | None = None) -> int:
    """Train one run; search callers may reuse already audited, fixed-split data.

    Callers supplying prepared_data must use the same data/preprocessing options
    and copy their shared preprocessing audit into the run directory.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    seed_everything(config["experiment"]["seed"], config["training"]["cpu_threads"])
    output_dir = resolve_path(config, config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
                        force=True)
    logger = logging.getLogger("a2f_ood")
    import rdkit
    write_json(output_dir / "runtime.json", {"python": sys.version, "executable": sys.executable,
               "platform": platform.platform(), "torch": torch.__version__, "rdkit": rdkit.__version__,
               "cuda_available": torch.cuda.is_available(), "seed": config["experiment"]["seed"],
               "package_source": str(Path(__file__).resolve().parent),
               "feature_schema": molecular_feature_schema(),
               "source_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in Path(__file__).resolve().parent.glob("*.py")}})
    datasets, vocabulary, group_to_id = prepared_data if prepared_data is not None else prepare_data(config, output_dir)
    logger.info("Prepared train=%d valid=%d test=%d vocab=%d train_groups=%d",
                len(datasets["train"]), len(datasets["valid"]), len(datasets["test"]),
                len(vocabulary), len(group_to_id))
    if (config["data"]["task_type"] in {"binary", "multilabel"}
            and config["loss"]["classification_weighting"] == "balanced"):
        class_weighting = fit_binary_class_weights(
            (m.row.labels for m in datasets["train"].molecules), config["data"]["label_columns"],
        )
        # Always refit on this run's training split, even if a saved YAML has weights.
        config["loss"]["binary_class_weights"] = class_weighting["weights"]
        write_json(output_dir / "class_weighting.json", class_weighting)
        for task in class_weighting["tasks"]:
            logger.info("Balanced BCE label=%s negatives=%d positives=%d w_negative=%.6f w_positive=%.6f",
                        task["label"], task["negative_count"], task["positive_count"],
                        task["negative_weight"], task["positive_weight"])
            if task["neutral_fallback"]:
                logger.warning("Label %s lacks both training classes; using neutral BCE weights", task["label"])
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if args.validate_only:
        for name, dataset in datasets.items():
            batch = next(iter(make_loader(dataset, config)))
            logger.info("%s: atoms=%s tokens=%s", name, tuple(batch["atom_features"].shape), tuple(batch["token_ids"].shape))
        return 0
    device = get_device(config["training"]["device"])
    model = build_model(config, vocabulary).to(device)
    logger.info("Device=%s trainable_parameters=%d", device, sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    tc = config["training"]
    scheduler = None
    if tc["lr_scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=tc["monitor_mode"], factor=tc["lr_scheduler_factor"],
            patience=tc["lr_scheduler_patience"], threshold=tc["monitor_min_delta"],
            threshold_mode="abs", min_lr=tc["min_learning_rate"],
        )
    group_dro = None
    if config["loss"]["use_group_dro"]:
        lc = config["loss"]
        counts = [0] * len(group_to_id)
        for molecule in datasets["train"].molecules:
            if any(label is not None for label in molecule.row.labels):
                counts[group_to_id[molecule.row.group]] += 1
        group_dro = GroupDRO(
            len(group_to_id), lc["group_dro_step_size"], update_mode=lc["group_dro_update_mode"],
            group_counts=counts if lc["group_dro_update_mode"] == "epoch" else None,
            max_weight_ratio=lc["group_dro_max_weight_ratio"],
        ).to(device)
        logger.info("GroupDRO mode=%s strength=%.3f max_importance=%.2f warmup=%d ramp=%d",
                    lc["group_dro_update_mode"], lc["group_dro_strength"], lc["group_dro_max_weight_ratio"],
                    lc["group_dro_warmup_epochs"], lc["group_dro_ramp_epochs"])
        if len(group_to_id) == 1:
            logger.warning("Only one training group: GroupDRO is equivalent to ERM")
    train_loader = make_loader(datasets["train"], config, shuffle=True)
    valid_loader = make_loader(datasets["valid"], config)
    monitor = config["training"]["monitor"]
    mode = config["training"]["monitor_mode"]
    best_score = math.inf if mode == "min" else -math.inf
    patience_score = best_score
    best_epoch, bad_epochs = 0, 0
    history = []
    best_path = output_dir / "best.pt"
    for epoch in range(1, config["training"]["epochs"] + 1):
        epoch_started = time.perf_counter()
        datasets["train"].set_epoch(epoch)
        epoch_config, scale = epoch_objectives(config, epoch)
        lr_warmup = tc["learning_rate_warmup_epochs"]
        if lr_warmup and epoch <= lr_warmup:
            for group in optimizer.param_groups:
                group["lr"] = tc["learning_rate"] * epoch / lr_warmup
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_epoch(model, train_loader, optimizer, epoch_config, device, group_dro)
        valid_metrics, _ = evaluate_model(model, valid_loader, config, device, vocabulary=vocabulary)
        score = valid_metrics.get(monitor)
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError(f"Validation monitor {monitor!r} is absent or undefined. "
                             "Use task_loss or a metric defined on validation labels.")
        improved = score < best_score if mode == "min" else score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        delta = tc["monitor_min_delta"]
        significant = score < patience_score - delta if mode == "min" else score > patience_score + delta
        if significant:
            patience_score, bad_epochs = score, 0
        else:
            bad_epochs += 1
        if scheduler is not None and epoch >= lr_warmup:
            scheduler.step(score)
        checkpoint = {"format_version": 1, "config": config, "epoch": epoch,
                          "feature_schema": molecular_feature_schema(),
                          "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                          "group_dro_state": group_dro.state_dict() if group_dro else None,
                          "scheduler_state": scheduler.state_dict() if scheduler else None,
                          "early_stopping_state": {"bad_epochs": bad_epochs, "patience_score": patience_score},
                          "group_to_id": group_to_id, "vocabulary": vocabulary.to_dict(),
                          "monitor": monitor, "best_score": best_score}
        for path in ([best_path] if improved else []) + [output_dir / "last.pt"]:
            temporary = path.with_suffix(".tmp.pt")
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        history.append({"epoch": epoch, "auxiliary_scale": scale,
                        "group_dro_strength": epoch_config["loss"]["group_dro_strength"],
                        "learning_rate": learning_rate, "next_learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch_seconds": time.perf_counter() - epoch_started,
                        "train": train_metrics, "valid": valid_metrics})
        write_json(output_dir / "history.json", history)
        logger.info("epoch=%d lr=%.3g aux_scale=%.2f dro_strength=%.3f train_total=%.5f "
                    "train_task=%.5f train_supervised=%.5f aux=%.5f grad_norm=%.3f clip_fraction=%.3f "
                    "valid_%s=%.5f best_epoch=%d",
                    epoch, learning_rate, scale, epoch_config["loss"]["group_dro_strength"],
                    train_metrics["total"], train_metrics["task"], train_metrics["robust"],
                    train_metrics["auxiliary_weighted"], train_metrics["gradient_norm_mean"],
                    train_metrics["gradient_clip_fraction"], monitor, score, best_epoch)
        if bad_epochs >= config["training"]["patience"]:
            logger.info("Early stopping at epoch %d", epoch)
            break
    model, vocabulary, _ = load_checkpoint(best_path, device)
    reports = {"best_epoch": best_epoch, "monitor": monitor, "best_validation_score": best_score}
    # Test metrics are evaluated only after model selection is complete.
    for name in (("valid",) if args.selection_only else ("valid", "test")):
        evaluation_dataset = datasets[name]
        evaluation_dataset.num_views = config["evaluation"]["random_smiles_views"]
        metrics, rows = evaluate_model(model, make_loader(evaluation_dataset, config), config,
                                       device, vocabulary=vocabulary)
        reports[name] = metrics
        write_predictions(output_dir / f"{name}_predictions.csv", rows)
    reports["test_evaluated"] = not args.selection_only
    write_json(output_dir / ("selection_metrics.json" if args.selection_only else "metrics.json"), reports)
    logger.info("Saved best checkpoint and reports to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

