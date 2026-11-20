"""Verify completed fixed-split trials and export comparison artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, log_loss, roc_auc_score, roc_curve)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    state, plan = read(root / "experiment_status.json"), read(root / "experiment_plan.json")
    assert state["status"] == "completed" and len(state["runs"]) == 3
    reference = Path(plan["reference_run"])
    manifest = read(reference / "split_manifest.json")
    vocab = read(reference / "vocabulary.json")
    weighting = read(reference / "class_weighting.json")
    source_root = Path(__file__).resolve().parents[1] / "src/a2f_ood"
    assert {p.name: digest(p) for p in source_root.glob("*.py")} == plan["training_source_hashes"]
    rows, initialization_hashes = [], []
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for run in state["runs"]:
        assert run["status"] == "completed"
        folder = Path(run["directory"])
        history, metrics = read(folder / "history.json"), read(folder / "metrics.json")
        frozen, initial = read(folder / "frozen_selection.json"), read(folder / "initialization_audit.json")
        assert [h["epoch"] for h in history] == list(range(1, plan["epochs"] + 1))
        assert read(folder / "split_manifest.json") == manifest
        assert read(folder / "vocabulary.json") == vocab
        assert read(folder / "class_weighting.json") == weighting
        assert read(folder / "runtime.json")["source_files_sha256"] == plan["training_source_hashes"]
        assert metrics["test_evaluation_count"] == 1 and metrics["early_stopping_enabled"] is False
        assert 'Early stopping at epoch' not in (folder / "train.log").read_text(encoding="utf-8")
        assert digest(folder / "best.pt") == frozen["checkpoint_sha256"]
        assert frozen["test_evaluated_before_selection"] is False
        initialization_hashes.append(initial["initial_model_state_sha256"])
        best = max(history, key=lambda h: h["valid"]["roc_auc"])
        assert best["epoch"] == metrics["best_epoch"] == frozen["best_epoch"]
        assert best["valid"]["roc_auc"] == metrics["best_validation_score"]
        checkpoint = torch.load(folder / "best.pt", map_location="cpu")
        last = torch.load(folder / "last.pt", map_location="cpu")
        assert checkpoint["epoch"] == best["epoch"] and last["epoch"] == plan["epochs"]
        config = checkpoint["config"]
        assert config["training"]["epochs"] == plan["epochs"]
        assert config["training"]["patience"] > plan["epochs"]
        assert config["loss"]["classification_weighting"] == "balanced"
        assert checkpoint["feature_schema"]["atom_dim"] == 157 and checkpoint["feature_schema"]["bond_dim"] == 14
        with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        assert [int(r["row_id"]) for r in records] == [r["row_id"] for r in manifest["test"]]
        y, p = [np.array([float(r[key]) for r in records]) for key in ("label", "prediction")]
        auc, ap, loss = float(roc_auc_score(y, p)), float(average_precision_score(y, p)), float(log_loss(y, p))
        assert abs(auc - metrics["test"]["roc_auc"]) < 1e-12
        assert abs(ap - metrics["test"]["average_precision"]) < 1e-12
        assert abs(loss - metrics["test"]["log_loss"]) < 1e-7
        matrix = confusion_matrix(y, p >= .5, labels=[0, 1])
        row = dict(seed=run["seed"], epochs=len(history), best_epoch=best["epoch"],
                   selection_valid_auc=metrics["best_validation_score"], valid_auc=metrics["valid"]["roc_auc"],
                   test_auc=auc, test_ap=ap, test_log_loss=loss,
                   test_balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
                   negative_recall=float(matrix[0, 0] / matrix[0].sum()), positive_recall=float(matrix[1, 1] / matrix[1].sum()),
                   total_seconds=run["seconds"], best_checkpoint_sha256=frozen["checkpoint_sha256"])
        rows.append(row)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"Seed {run['seed']} (epoch {best['epoch']}): {auc:.4f}")
        epochs = [h["epoch"] for h in history]
        line, = axes[0].plot(epochs, [h["valid"]["roc_auc"] for h in history], label=f"Seed {run['seed']}")
        axes[0].scatter([best["epoch"]], [best["valid"]["roc_auc"]], color=line.get_color(), marker="*", s=120)
        del checkpoint, last
    assert len(set(initialization_hashes)) == 3
    aucs = np.array([r["test_auc"] for r in rows])
    summary = read(root / "summary.json")
    assert np.isclose(summary["test_auc_mean"], aucs.mean())
    assert np.isclose(summary["test_auc_sample_std"], aucs.std(ddof=1))
    selected = max(rows, key=lambda row: row["selection_valid_auc"])
    assert selected["seed"] == summary["selected_by_validation_seed"]
    folder = root / f"seed_{selected['seed']}"
    shutil.copy2(folder / "best.pt", root / "selected_by_validation.pt")
    shutil.copy2(folder / "config.yaml", root / "selected_by_validation_config.yaml")
    selection = {"criterion": "maximum validation ROC-AUC across the three per-seed best checkpoints",
                 "seed": selected["seed"], "best_epoch": selected["best_epoch"],
                 "validation_auc": selected["selection_valid_auc"], "test_auc": selected["test_auc"],
                 "test_used_for_selection": False, "checkpoint_sha256": digest(root / "selected_by_validation.pt")}
    (root / "selected_by_validation.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    with (root / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ax.plot([0, 1], [0, 1], "--", color="#aaa")
    ax.set(title="Fixed-split 50-epoch repeats: validation-selected checkpoints", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=9)
    fig.savefig(root / "test_roc_curves.png", dpi=170)
    plt.close(fig)
    axes[0].set(title="Validation AUC and per-seed selected checkpoints", xlabel="Epoch", ylabel="Validation ROC-AUC")
    axes[0].legend(fontsize=9)
    axes[1].bar([str(r["seed"]) for r in rows], aucs, color=["#17628a", "#489d73", "#b08d3c"])
    axes[1].axhline(aucs.mean(), ls="--", color="#444", label=f"Mean {aucs.mean():.4f}; sample SD {aucs.std(ddof=1):.4f}")
    axes[1].set(title="Test AUROC of each validation-selected checkpoint", xlabel="Random seed", ylabel="Test AUROC", ylim=(0, 1))
    for index, value in enumerate(aucs):
        axes[1].text(index, value + .015, f"{value:.4f}", ha="center")
    axes[1].legend(fontsize=9, loc="lower right")
    fig2.savefig(root / "results_overview.png", dpi=170)
    plt.close(fig2)
    audit = {"passed": True, "runs": rows, "all_full_epochs": plan["epochs"], "identical_reference_split_and_vocabulary": True,
             "identical_train_fitted_class_weights": True, "distinct_initial_weight_hashes": initialization_hashes,
             "same_training_source_hashes": True, "checkpoint_selection_matches_validation_maximum": True,
             "saved_test_metrics_recomputed": True, "test_evaluations_per_seed": 1,
             "mean_test_auc": float(aucs.mean()), "sample_std_test_auc": float(aucs.std(ddof=1)),
             "selected_by_validation": selection}
    (root / "result_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with (root / "MULTISEED_REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\n已独立核对每次50轮历史、best/last检查点轮数、初始化差异、原始划分与词表、训练类别权重，并从预测CSV重算AUROC/AP/log loss。全部通过。\n\n")
        stream.write("![复验结果](results_overview.png)\n\n![各次测试ROC](test_roc_curves.png)\n\n")
        stream.write("[结果CSV](results.csv) · [核验记录](result_audit.json) · [按验证集选出的模型](selected_by_validation.pt) · [选模记录](selected_by_validation.json)\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
