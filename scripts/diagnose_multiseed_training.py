"""Diagnose saved training runs without updating parameters or selecting models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from a2f_ood.engine import (evaluate_model, get_device, json_safe, load_checkpoint,
                           make_loader, move_batch, prepare_data, seed_everything)
from a2f_ood.data import collate_molecules
from a2f_ood.losses import compute_losses
from a2f_ood.train import epoch_objectives


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def group_pair_diagnosis(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in records:
        groups[row["group"]].append(row)
    labels = np.array([float(row["label"]) for row in records])
    predictions = np.array([float(row["prediction"]) for row in records])
    total_pairs = float(labels.sum() * (1 - labels).sum())
    pairs = concordance = 0.0
    aucs = []
    for group in groups.values():
        y = np.array([float(row["label"]) for row in group])
        p = np.array([float(row["prediction"]) for row in group])
        count = float(y.sum() * (1 - y).sum())
        if count:
            score = float(roc_auc_score(y, p))
            pairs += count
            concordance += count * score
            aucs.append(score)
    auc = float(roc_auc_score(labels, predictions))
    return {"pooled_auc": auc, "groups": len(groups), "groups_with_both_classes": len(aucs),
            "macro_within_group_auc": float(np.mean(aucs)), "pair_weighted_within_group_auc": concordance / pairs,
            "within_group_pair_fraction": pairs / total_pairs,
            "cross_group_pair_auc": (auc * total_pairs - concordance) / (total_pairs - pairs)}


def checkpoint_probe(model, datasets, config, device, epoch, batches):
    dataset = datasets["train"]
    dataset.randomize = True
    dataset.num_views = 2
    dataset.set_epoch(epoch)
    indices = np.random.RandomState(20261120).choice(len(dataset), size=16 * batches, replace=False)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    params = [parameter for _, parameter in named]
    graph_indices = [i for i, (name, _) in enumerate(named) if name.startswith("graph_encoder.")]
    objectives = ["task", "auxiliary_weighted", "weighted_atom_alignment", "weighted_generation",
                  "weighted_unimodal", "weighted_cross_modal", "weighted_serialization_consistency"]
    epoch_config, _ = epoch_objectives(config, epoch)
    results = []
    model.train()
    for batch_index in range(batches):
        seed_everything(20261120 + batch_index, config["training"]["cpu_threads"])
        batch = move_batch(collate_molecules([dataset[int(i)] for i in indices[batch_index * 16:(batch_index + 1) * 16]]), device)
        losses = compute_losses(model(batch), batch, epoch_config)
        summary, graph_vectors = {}, {}
        for key in objectives:
            gradients = torch.autograd.grad(losses[key], params, retain_graph=True, allow_unused=True)
            squared = sum(float(g.detach().float().square().sum()) for g in gradients if g is not None)
            graph_vector = torch.cat([gradients[i].detach().reshape(-1) if gradients[i] is not None else
                                      torch.zeros_like(params[i]).reshape(-1) for i in graph_indices])
            summary[key] = {"loss": float(losses[key].detach()), "global_grad_norm": squared ** .5,
                            "graph_grad_norm": float(graph_vector.norm())}
            graph_vectors[key] = graph_vector
        for key in objectives[1:]:
            first, second = graph_vectors["task"], graph_vectors[key]
            denominator = float(first.norm() * second.norm())
            summary[key]["cosine_with_task_on_graph"] = float(torch.dot(first, second)) / denominator if denominator > 1e-12 else None
        summary["batch_index"] = batch_index
        summary["row_ids"] = batch["row_ids"]
        results.append(summary)
        del graph_vectors, losses, batch
    aggregates = {}
    for key in objectives:
        aggregates[key] = {field + "_mean": float(np.mean([r[key][field] for r in results])) for field in
                           ("loss", "global_grad_norm", "graph_grad_norm")}
        if key != "task":
            cosines = [r[key]["cosine_with_task_on_graph"] for r in results if r[key]["cosine_with_task_on_graph"] is not None]
            aggregates[key].update(cosine_with_task_on_graph_mean=float(np.mean(cosines)) if cosines else None,
                                   negative_cosine_batch_fraction=float(np.mean(np.array(cosines) < 0)) if cosines else None)
    return {"batches": batches, "molecules": len(indices), "batch_size": 16,
            "mode": "training mode with controlled dropout seed; no optimizer updates",
            "aggregates": aggregates, "per_batch": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-batches", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    out = root / "training_diagnosis"
    out.mkdir(exist_ok=True)
    state, plan = read(root / "experiment_status.json"), read(root / "experiment_plan.json")
    assert state["status"] == "completed"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src/a2f_ood").glob("*.py")}
    assert hashes == plan["training_source_hashes"]
    device = get_device("cuda")
    first_model, _, first = load_checkpoint(Path(state["runs"][0]["directory"]) / "best.pt", "cpu")
    config = first["config"]
    config["experiment"]["seed"] = plan["data_and_smiles_seed"]
    seed_everything(42, config["training"]["cpu_threads"])
    datasets, vocabulary, _ = prepare_data(config, None)
    del first_model, first
    results = {"purpose": "Read-only diagnosis of the saved best and last checkpoints",
               "test_predictions_reused": True, "additional_test_forward_evaluations": 0,
               "no_optimizer_updates": True, "source_hashes_match": True, "runs": []}
    for run in state["runs"]:
        folder = Path(run["directory"])
        history, metrics = read(folder / "history.json"), read(folder / "metrics.json")
        best_row = history[metrics["best_epoch"] - 1]
        item = {"seed": run["seed"], "best_epoch": metrics["best_epoch"],
                "best_valid_auc": best_row["valid"]["roc_auc"], "last_valid_auc": history[-1]["valid"]["roc_auc"],
                "test_auc": metrics["test"]["roc_auc"], "best_valid_weighted_bce": best_row["valid"]["class_weighted_task_loss"],
                "last_valid_weighted_bce": history[-1]["valid"]["class_weighted_task_loss"],
                "gradient_phases": {}, "saved_prediction_groups": {}, "checkpoints": {}}
        for lo, hi in [(1, 2), (3, 12), (13, 20), (21, 50)]:
            segment = history[lo - 1:hi]
            item["gradient_phases"][f"{lo}-{hi}"] = {"clip_fraction_mean": float(np.mean([r["train"]["gradient_clip_fraction"] for r in segment])),
                                                      "preclip_norm_mean": float(np.mean([r["train"]["gradient_norm_mean"] for r in segment])),
                                                      "preclip_norm_max": max(r["train"]["gradient_norm_max"] for r in segment)}
        for split in ("valid", "test"):
            item["saved_prediction_groups"][split] = group_pair_diagnosis(folder / f"{split}_predictions.csv")
        for kind in ("best", "last"):
            model, checkpoint_vocab, checkpoint = load_checkpoint(folder / f"{kind}.pt", device)
            cp_config = checkpoint["config"]
            assert checkpoint_vocab.to_dict() == vocabulary.to_dict()
            entry = {"epoch": checkpoint["epoch"], "evaluation_mode": {}}
            for split in ("train", "valid"):
                dataset = datasets[split]
                dataset.num_views, dataset.randomize = 1, False
                dataset.set_epoch(0)
                scores, rows = evaluate_model(model, make_loader(dataset, cp_config), cp_config, device, vocabulary=vocabulary)
                labels = np.array([float(row["label"]) for row in rows])
                probabilities = np.array([float(row["prediction"]) for row in rows])
                entry["evaluation_mode"][split] = {key: scores[key] for key in ("roc_auc", "task_loss", "class_weighted_task_loss", "ece", "brier")}
                entry["evaluation_mode"][split].update(positive_fraction=float(labels.mean()), mean_probability=float(probabilities.mean()),
                                                       positive_class_bce=float(-np.log(np.clip(probabilities[labels == 1], 1e-7, 1)).mean()),
                                                       negative_class_bce=float(-np.log(np.clip(1 - probabilities[labels == 0], 1e-7, 1)).mean()))
            entry["gradient_probe"] = checkpoint_probe(model, datasets, cp_config, device, checkpoint["epoch"], args.probe_batches)
            item["checkpoints"][kind] = entry
            print(json.dumps({"seed": run["seed"], "checkpoint": kind, "epoch": checkpoint["epoch"],
                              "evaluation": entry["evaluation_mode"], "gradient_probe": entry["gradient_probe"]["aggregates"]}), flush=True)
            del model, checkpoint
            torch.cuda.empty_cache()
        results["runs"].append(item)
        (out / "diagnosis.json").write_text(json.dumps(json_safe(results), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src/a2f_ood").glob("*.py")}
    print("Completed read-only diagnosis; no training weights or test selections changed", flush=True)


if __name__ == "__main__":
    main()
