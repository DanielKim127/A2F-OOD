"""Shared training/inference utilities; checkpoints contain tensors and metadata only."""
from __future__ import annotations

import csv
import json
import logging
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from a2f_ood.config import resolve_path
from a2f_ood.data import (
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM, MoleculeDataset, SmilesVocabulary,
    collate_molecules, load_molecule_table, molecular_feature_schema, preprocess_rows,
)
from a2f_ood.losses import compute_losses, configured_binary_class_weights
from a2f_ood.metrics import compute_group_metrics, compute_metrics
from a2f_ood.model import A2FModel
from a2f_ood.splits import assign_scaffold_groups, audit_split_overlaps, make_split_indices

LOGGER = logging.getLogger("a2f_ood")


def seed_everything(seed: int, cpu_threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot use CUDA")
    return torch.device(name)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset: MoleculeDataset, config: dict, *, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=shuffle,
                      num_workers=config["training"].get("num_workers", 0),
                      collate_fn=collate_molecules, worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=False)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def validate_labels(molecules: list, config: dict) -> None:
    task = config["data"]["task_type"]
    for item in molecules:
        for label in item.row.labels:
            if label is None:
                continue
            if task in {"binary", "multilabel"} and label not in {0.0, 1.0}:
                raise ValueError(f"Row {item.row.row_id}: binary labels must be 0/1 or missing; got {label}")
            if task == "multiclass" and (label != int(label) or not 0 <= label < config["data"]["num_classes"]):
                raise ValueError(f"Row {item.row.row_id}: class labels must be integer indices in [0, num_classes)")


def prepare_data(config: dict, output_dir: Path | None = None) -> tuple:
    """Fit vocabulary/group IDs strictly on train, then construct evaluation datasets."""
    dc, mc = config["data"], config["model"]
    rows = load_molecule_table(resolve_path(config, dc["csv_path"]),
                               smiles_column=dc["smiles_column"], label_columns=dc["label_columns"],
                               group_column=dc.get("group_column"), split_column=dc.get("split_column"))
    molecules = preprocess_rows(rows, max_atom_count=mc["max_atom_count"],
                                max_smiles_length=dc["max_smiles_length"],
                                invalid_smiles_policy=dc["invalid_smiles_policy"])
    if not molecules:
        raise ValueError("No valid molecules remain after preprocessing")
    validate_labels(molecules, config)
    strategy = dc.get("group_strategy", "column")
    if strategy == "scaffold":
        group_names = assign_scaffold_groups(molecules)
    elif strategy == "size":
        # Fixed chemical-size bins do not learn thresholds from held-out data.
        group_names = [f"atoms_{m.num_atoms // 10 * 10:03d}" for m in molecules]
    elif strategy == "none":
        group_names = ["all"] * len(molecules)
    else:
        group_names = [m.row.group or "all" for m in molecules]
    molecules = [replace(m, row=replace(m.row, group=str(group))) for m, group in zip(molecules, group_names)]
    indices = make_split_indices(molecules, strategy=dc["split_strategy"],
                                  seed=config["experiment"]["seed"],
                                  fractions=tuple(dc["split_fractions"]),
                                  valid_groups=dc.get("valid_groups"), test_groups=dc.get("test_groups"),
                                  overlap_policy=dc.get("overlap_policy", "error"))
    subsets = {name: [molecules[i] for i in idx] for name, idx in indices.items()}
    if any(not subsets.get(name) for name in ("train", "valid", "test")):
        raise ValueError("train, valid and test must all be nonempty; adjust split strategy/fractions")
    if not any(any(y is not None for y in m.row.labels) for m in subsets["train"]):
        raise ValueError("Training set contains no observed property labels")
    vocabulary = SmilesVocabulary.build(subsets["train"],
                                        random_views=dc["random_smiles_per_molecule"],
                                        seed=config["experiment"]["seed"])
    group_to_id = {g: i for i, g in enumerate(sorted({m.row.group for m in subsets["train"]}))}
    datasets = {}
    for name in ("train", "valid", "test"):
        datasets[name] = MoleculeDataset(
            subsets[name], vocabulary,
            num_views=(dc["random_smiles_per_molecule"] if name == "train" else
                       config["training"].get("validation_smiles_views", 1) if name == "valid" else 1),
            randomize=name == "train", seed=config["experiment"]["seed"], group_to_id=group_to_id,
        )
    if output_dir:
        valid_ids = {m.row.row_id for m in molecules}
        write_json(output_dir / "skipped_rows.json", [
            {"row_id": r.row_id, "smiles": r.smiles} for r in rows if r.row_id not in valid_ids
        ])
        write_json(output_dir / "vocabulary.json", vocabulary.to_dict())
        write_json(output_dir / "overlap_audit.json", audit_split_overlaps(molecules, indices))
        write_json(output_dir / "preprocessing_summary.json", {
            "input_rows": len(rows), "retained_rows": len(molecules),
            "skipped_rows": len(rows) - len(molecules),
            "retained_by_split": {name: len(items) for name, items in subsets.items()},
            "input_by_split": {name: sum(r.split == name for r in rows) for name in ("train", "valid", "test")},
            "max_atom_count": mc["max_atom_count"], "max_smiles_length": dc["max_smiles_length"],
        })
        write_json(output_dir / "split_manifest.json", {
            name: [{"row_id": m.row.row_id, "canonical_smiles": m.canonical_smiles,
                    "group": m.row.group} for m in members]
            for name, members in subsets.items()
        })
    return datasets, vocabulary, group_to_id


def build_model(config: dict, vocabulary: SmilesVocabulary) -> A2FModel:
    return A2FModel(config, len(vocabulary), ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                    pad_id=vocabulary.pad_id, bos_id=vocabulary.bos_id, eos_id=vocabulary.eos_id)


def probabilities(logits: torch.Tensor, task: str) -> torch.Tensor:
    if task in {"binary", "multilabel"}:
        return logits.sigmoid()
    if task == "multiclass":
        return logits.softmax(dim=-1)
    return logits


def train_epoch(model: A2FModel, loader: DataLoader, optimizer: torch.optim.Optimizer,
                config: dict, device: torch.device, group_dro: Any = None) -> dict:
    model.train()
    if group_dro is not None:
        group_dro.train()
        group_dro.begin_epoch()
    sums: dict[str, float] = {}
    count = 0
    gradient_norm_sum, gradient_norm_max, clipped_steps, steps = 0.0, 0.0, 0, 0
    clip = float(config["training"].get("gradient_clip", 5.0))
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(
            model(batch), batch, config, group_dro=group_dro,
            update_dro=float(config["loss"].get("group_dro_strength", 1.0)) > 0,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite training loss at rows {raw_batch['row_ids']}")
        losses["total"].backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True))
        gradient_norm_sum += norm
        gradient_norm_max = max(gradient_norm_max, norm)
        clipped_steps += int(norm > clip)
        steps += 1
        optimizer.step()
        n = len(batch["labels"])
        count += n
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                multiplier = 1 if key == "num_labeled" else n
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * multiplier
    result = {k: v if k == "num_labeled" else v / max(count, 1) for k, v in sums.items()}
    result.update(gradient_norm_mean=gradient_norm_sum / max(steps, 1),
                  gradient_norm_max=gradient_norm_max,
                  gradient_clip_fraction=clipped_steps / max(steps, 1))
    if group_dro is not None:
        result.update(group_dro.end_epoch(update=float(config["loss"].get("group_dro_strength", 1.0)) > 0))
    return result


@torch.no_grad()
def evaluate_model(model: A2FModel, loader: DataLoader, config: dict, device: torch.device,
                   *, vocabulary: SmilesVocabulary | None = None) -> tuple[dict, list[dict]]:
    model.eval()
    task = config["data"]["task_type"]
    label_columns = config["data"]["label_columns"]
    all_labels, all_predictions, all_groups = [], [], []
    graph_predictions, smiles_predictions = [], []
    rows = []
    generation_total, generation_valid, generation_exact = 0, 0, 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        # Property inference never runs the teacher-forced decoder.
        outputs = model({key: value for key, value in batch.items() if key != "target_ids"})
        views = probabilities(outputs["logits"], task)
        pred = views.mean(dim=1).cpu().numpy()
        view_variance = views.var(dim=1, unbiased=False).mean(dim=-1).cpu().numpy()
        graph_pred = probabilities(outputs["graph_logits"], task).cpu().numpy()
        smiles_pred = probabilities(outputs["smiles_logits"], task).mean(dim=1).cpu().numpy()
        cross_error = outputs["cross_modal_error"].mean(dim=(1, 2)).cpu().numpy()
        labels = batch["labels"].cpu().numpy().copy()
        labels[~batch["label_mask"].cpu().numpy()] = np.nan
        all_labels.append(labels)
        all_predictions.append(pred)
        graph_predictions.append(graph_pred)
        smiles_predictions.append(smiles_pred)
        groups = raw_batch["groups"]
        all_groups.extend(groups)
        generated = None
        if config["evaluation"].get("generate_smiles") and model.config["model"].get("use_cross_modal_decoder"):
            from rdkit import Chem
            if vocabulary is None:
                raise ValueError("Generation evaluation requires the saved vocabulary")
            generated_ids = model.generate_smiles(batch, max_length=config["data"]["max_smiles_length"])
            generated = []
            for seq, source in zip(generated_ids.cpu().tolist(), raw_batch["smiles"]):
                # Keep UNK visible so it cannot disappear and inflate validity.
                text = vocabulary.decode(seq, skip_special=False)
                for token in ("<bos>", "<cls>", "<pad>"):
                    text = text.replace(token, "")
                generated.append(text)
                mol = Chem.MolFromSmiles(text)
                generation_total += 1
                if text and mol is not None and mol.GetNumAtoms() > 0:
                    generation_valid += 1
                    generation_exact += int(Chem.MolToSmiles(mol, isomericSmiles=True) == source)
        for i, row_id in enumerate(raw_batch["row_ids"]):
            row = {"row_id": int(row_id), "smiles": raw_batch["smiles"][i], "group": groups[i],
                   "cross_modal_error": float(cross_error[i]), "smiles_prediction_variance": float(view_variance[i])}
            for j, label in enumerate(label_columns):
                row[label] = float(labels[i, j]) if np.isfinite(labels[i, j]) else ""
            if task == "multiclass":
                for j in range(pred.shape[1]):
                    row[f"prediction_class_{j}"] = float(pred[i, j])
                    row[f"graph_prediction_class_{j}"] = float(graph_pred[i, j])
                    row[f"smiles_prediction_class_{j}"] = float(smiles_pred[i, j])
            else:
                for j, label in enumerate(label_columns):
                    suffix = "" if len(label_columns) == 1 else f"_{label}"
                    row[f"prediction{suffix}"] = float(pred[i, j])
                    row[f"graph_prediction{suffix}"] = float(graph_pred[i, j])
                    row[f"smiles_prediction{suffix}"] = float(smiles_pred[i, j])
            if generated is not None:
                row["generated_smiles"] = generated[i]
            rows.append(row)
    labels = np.concatenate(all_labels)
    pred = np.concatenate(all_predictions)
    metrics = compute_metrics(labels, pred, task_type=task)
    metrics["task_loss"] = prediction_risk(labels, pred, task)
    class_weights = configured_binary_class_weights(config)
    if class_weights is not None:
        metrics["class_weighted_task_loss"] = prediction_risk(labels, pred, task, binary_class_weights=class_weights)
    metrics["unimodal_graph"] = compute_metrics(labels, np.concatenate(graph_predictions), task_type=task)
    metrics["unimodal_smiles"] = compute_metrics(labels, np.concatenate(smiles_predictions), task_type=task)
    if config["evaluation"].get("report_group_metrics", True):
        metrics["group_metrics"] = compute_group_metrics(labels, pred, all_groups, task_type=task)
    metrics["mean_cross_modal_error"] = float(np.mean([r["cross_modal_error"] for r in rows]))
    metrics["mean_smiles_prediction_variance"] = float(np.mean([r["smiles_prediction_variance"] for r in rows]))
    # These are disagreement diagnostics, not calibrated probabilities of error.
    order = np.argsort([r["cross_modal_error"] for r in rows])
    metrics["error_by_cross_modal_bin"] = {
        str(i): {"count": len(idx), **compute_metrics(labels[idx], pred[idx], task_type=task)}
        for i, idx in enumerate(np.array_split(order, min(4, len(order))))
    }
    if generation_total:
        metrics["generation_validity"] = generation_valid / generation_total
        metrics["generation_exact_match"] = generation_exact / generation_total
    return metrics, rows


def prediction_risk(labels: np.ndarray, pred: np.ndarray, task: str,
                    *, binary_class_weights: list[list[float]] | None = None) -> float:
    """Natural evaluation loss of the averaged view predictions, ignoring missing labels."""
    if task == "multiclass":
        valid = np.isfinite(labels[:, 0])
        return float(-np.log(np.clip(pred[valid, labels[valid, 0].astype(int)], 1e-7, 1)).mean()) if valid.any() else float("nan")
    valid = np.isfinite(labels)
    if not valid.any():
        return float("nan")
    if task == "regression":
        return float(((labels[valid] - pred[valid]) ** 2).mean())
    p = np.clip(pred[valid], 1e-7, 1 - 1e-7)
    y = labels[valid]
    losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if binary_class_weights is not None:
        weights = np.asarray(binary_class_weights, dtype=np.float64)
        if weights.shape != (labels.shape[-1], 2) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("binary_class_weights must be positive [num_tasks, 2] negative/positive weights")
        element_weights = np.where(labels == 1, weights[:, 1], weights[:, 0])
        weighted = np.zeros_like(labels, dtype=np.float64)
        weighted[valid] = losses * element_weights[valid]
        counts = valid.sum(-1)
        return float((weighted.sum(-1) / counts.clip(min=1))[counts > 0].mean())
    return float(losses.mean())


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty prediction table")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[A2FModel, SmilesVocabulary, dict]:
    # Backward compatible with PyTorch 1.12; only load checkpoints you trust.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    schema = checkpoint.get("feature_schema")
    projection = checkpoint["model_state"].get("graph_encoder.input_projection.weight")
    if (schema is not None and schema != molecular_feature_schema()) or (
        projection is not None and projection.shape[1] != ATOM_FEATURE_DIM
    ):
        raise ValueError(
            "Checkpoint feature schema is incompatible with the current atom/bond attributes "
            f"({ATOM_FEATURE_DIM}/{BOND_FEATURE_DIM} dimensions). Retrain with the current schema."
        )
    vocabulary = SmilesVocabulary.from_dict(checkpoint["vocabulary"])
    model = build_model(checkpoint["config"], vocabulary).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, vocabulary, checkpoint
