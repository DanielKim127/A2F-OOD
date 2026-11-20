"""Audit and report a completed single-seed learning-rate/warmup experiment.

Uses saved histories, checkpoints and prediction CSVs; performs no model forwards.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_multiseed_training import group_pair_diagnosis


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    root, baseline = Path(args.output).resolve(), Path(args.baseline).resolve()
    state, plan = read(root / "experiment_status.json"), read(root / "experiment_plan.json")
    assert state["status"] == "completed" and len(state["runs"]) == 1
    run = state["runs"][0]
    assert run["status"] == "completed" and run["seed"] == 42
    folder, reference = Path(run["directory"]), Path(plan["reference_run"])
    history, metrics = read(folder / "history.json"), read(folder / "metrics.json")
    frozen, initial = read(folder / "frozen_selection.json"), read(folder / "initialization_audit.json")
    best = max(history, key=lambda row: row["valid"]["roc_auc"])
    assert len(history) == 50 and [h["epoch"] for h in history] == list(range(1, 51))
    assert metrics["best_epoch"] == frozen["best_epoch"] == best["epoch"]
    assert metrics["best_validation_score"] == best["valid"]["roc_auc"]
    assert metrics["test_evaluation_count"] == 1 and not metrics["early_stopping_enabled"]
    assert not frozen["test_evaluated_before_selection"]
    assert initial["seed"] == 42 and not initial["loaded_pretrained_weights"]
    assert digest(folder / "best.pt") == frozen["checkpoint_sha256"]
    assert {p.name: digest(p) for p in (ROOT / "src/a2f_ood").glob("*.py")} == plan["training_source_hashes"]
    assert read(folder / "runtime.json")["source_files_sha256"] == plan["training_source_hashes"]
    for filename in ("split_manifest.json", "vocabulary.json", "preprocessing_summary.json", "class_weighting.json"):
        assert read(folder / filename) == read(reference / filename), filename
    data_audit = read(folder / "fixed_data_audit.json")
    assert data_audit["data_and_smiles_seed"] == 42
    assert data_audit["retained_by_split"] == {"train": 4898, "valid": 2638, "test": 2719}
    checkpoint = torch.load(folder / "best.pt", map_location="cpu")
    last = torch.load(folder / "last.pt", map_location="cpu")
    assert checkpoint["epoch"] == best["epoch"] and last["epoch"] == 50
    config = checkpoint["config"]
    assert config["experiment"]["seed"] == 42
    assert config["training"]["learning_rate"] == 7.5e-5
    assert config["training"]["learning_rate_warmup_epochs"] == 5
    assert config["training"]["epochs"] == 50 and config["training"]["patience"] == 51
    assert config["loss"]["classification_weighting"] == "balanced"
    assert checkpoint["feature_schema"]["atom_dim"] == 157
    assert checkpoint["feature_schema"]["bond_dim"] == 14
    for h in history[:5]:
        assert np.isclose(h["learning_rate"], 7.5e-5 * h["epoch"] / 5, rtol=1e-10, atol=0)
    for h in history:
        assert all(np.isfinite(v) for v in h["train"].values())
        assert h["train"]["num_labeled"] == 4898
        assert np.isfinite(h["valid"]["roc_auc"])
    log = (folder / "train.log").read_text(encoding="utf-8")
    assert "Early stopping at epoch" not in log and "FloatingPointError" not in log
    manifest = read(folder / "split_manifest.json")
    with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert [int(r["row_id"]) for r in predictions] == [r["row_id"] for r in manifest["test"]]
    labels, probabilities = [np.array([float(r[key]) for r in predictions]) for key in ("label", "prediction")]
    auc, ap = float(roc_auc_score(labels, probabilities)), float(average_precision_score(labels, probabilities))
    assert abs(auc - metrics["test"]["roc_auc"]) < 1e-12
    assert abs(ap - metrics["test"]["average_precision"]) < 1e-12
    groups = {split: group_pair_diagnosis(folder / f"{split}_predictions.csv") for split in ("valid", "test")}
    del checkpoint, last

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    items = [("Validation ROC-AUC", "ROC-AUC", [h["valid"]["roc_auc"] for h in history]),
             ("Online training balanced BCE", "Balanced BCE", [h["train"]["task"] for h in history]),
             ("Validation balanced BCE", "Balanced BCE", [h["valid"]["class_weighted_task_loss"] for h in history]),
             ("Learning rate: 5-epoch linear warmup + plateau", "Learning rate", [h["learning_rate"] for h in history]),
             ("Optimizer steps triggering gradient clipping", "Clipped steps (%)", [100 * h["train"]["gradient_clip_fraction"] for h in history]),
             ("Mean gradient norm BEFORE clipping", "Pre-clip gradient norm", [h["train"]["gradient_norm_mean"] for h in history])]
    for ax, (title, ylabel, values) in zip(axes.flat, items):
        ax.plot(epochs, values, color="#17628a", label="Seed 42: LR 7.5e-5, warmup 5")
        ax.axvline(best["epoch"], color="#b08d3c", ls="--", alpha=.8, label=f"Selected epoch {best['epoch']}")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0, 0].scatter([best["epoch"]], [best["valid"]["roc_auc"]], color="#b08d3c", marker="*", s=150, zorder=4)
    axes[1, 1].set_yscale("log")
    axes[2, 0].set_ylim(0, 102)
    axes[2, 1].axhline(1, color="#555", ls=":")
    fig.savefig(root / "training_curves.png", dpi=165)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.plot(fpr, tpr, color="#17628a", label=f"Seed 42, epoch {best['epoch']}: AUROC {auc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", color="#999")
    ax.set(title="Test ROC of the validation-selected checkpoint", xlabel="False positive rate", ylabel="True positive rate")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(root / "test_roc_curve.png", dpi=165)
    plt.close(fig)

    previous = read(baseline / "summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for old in previous["runs"]:
        old_history = read(Path(old["directory"]) / "history.json")
        for ax, key in zip(axes, ("roc_auc", "class_weighted_task_loss")):
            ax.plot([h["epoch"] for h in old_history], [h["valid"][key] for h in old_history], alpha=.5, lw=1,
                    label=f"Previous seed {old['seed']}: LR 1.5e-4, warmup 2")
    for ax, key, ylabel in zip(axes, ("roc_auc", "class_weighted_task_loss"), ("Validation AUROC", "Validation balanced BCE")):
        ax.plot(epochs, [h["valid"][key] for h in history], color="#161616", lw=2.2, label="New seed 42: LR 7.5e-5, warmup 5")
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Different seeds: descriptive comparison, not an isolated LR/warmup effect")
    fig.savefig(root / "baseline_comparison.png", dpi=165)
    plt.close(fig)

    phases = {}
    for lo, hi in ((1, 5), (6, 12), (13, 20), (21, 50)):
        segment = history[lo - 1:hi]
        phases[f"{lo}-{hi}"] = {"mean_clip_fraction": float(np.mean([h["train"]["gradient_clip_fraction"] for h in segment])),
                               "mean_preclip_norm": float(np.mean([h["train"]["gradient_norm_mean"] for h in segment]))}
    row = {"seed": 42, "epochs": 50, "best_epoch": best["epoch"], "best_validation_auc": metrics["best_validation_score"],
           "test_auc": auc, "test_ap": ap, "last_validation_auc": history[-1]["valid"]["roc_auc"],
           "validation_auc_drop_to_last": metrics["best_validation_score"] - history[-1]["valid"]["roc_auc"],
           "test_within_assay_pair_weighted_auc": groups["test"]["pair_weighted_within_group_auc"],
           "test_within_assay_macro_auc": groups["test"]["macro_within_group_auc"],
           "total_seconds": run["seconds"]}
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    audit = {"passed": True, "complete_epochs": 50, "warmup_schedule_verified": True,
             "reference_split_vocabulary_preprocessing_weights_identical": True, "training_source_hashes_match": True,
             "feature_dimensions": {"atom": 157, "bond": 14}, "seed": 42,
             "checkpoint_selection_matches_validation_maximum": True, "saved_test_metrics_recomputed": True,
             "test_evaluation_count": 1, "no_additional_test_forwards_for_reporting": True,
             "best_checkpoint_sha256": frozen["checkpoint_sha256"]}
    save(root / "result_audit.json", audit)
    save(root / "run_summary.json", {**row, "group_metrics": groups, "gradient_phases": phases,
         "baseline": {"seeds": [r["seed"] for r in previous["runs"]], "test_auc_mean": previous["test_auc_mean"],
                      "test_auc_sample_std": previous["test_auc_sample_std"], "matched_seed_comparison": False},
         "test_used_for_checkpoint_selection": False,
         "configuration_origin": "LR/warmup candidate derived from the previously explored 034_refine architecture"})
    summary = read(root / "summary.json")
    summary.update(configuration_origin="LR 7.5e-5 and warmup 5, derived from previously explored 034_refine", **row)
    save(root / "summary.json", summary)

    lines = ["# DrugOOD EC50 assay：学习率与预热候选，seed=42", "",
             "已完成 50/50 个 epoch，从头初始化，无早停。按逐轮验证 AUROC 最高的检查点进行一次最终测试。", "",
             "| 指标 | 结果 |", "|---|---:|", f"| 验证最优轮 | {best['epoch']} |",
             f"| 最优验证 AUROC | {metrics['best_validation_score']:.6f} |", f"| 对应测试 AUROC | {auc:.6f} |",
             f"| 对应测试 AP | {ap:.6f} |", f"| 第50轮验证 AUROC | {row['last_validation_auc']:.6f} |",
             f"| 测试 assay 内按样本对加权 AUROC | {row['test_within_assay_pair_weighted_auc']:.6f} |",
             f"| 测试 assay 内宏平均 AUROC | {row['test_within_assay_macro_auc']:.6f} |",
             f"| 总耗时 | {run['seconds'] / 60:.2f} 分钟 |", "",
             "相对上一轮配置，只将峰值学习率从 1.5e-4 改为 7.5e-5，学习率线性预热从 2 轮改为 5 轮，并使用训练种子 42。",
             "AdamW、batch=16、weight decay=0.01、gradient clip=1.0、Plateau patience=3/factor=0.5/min_lr=1e-6 均沿用。",
             "hidden=192，图4层、SMILES3层，dropout=0.2，graph 主分类；原子157维、化学键14维。分类使用训练比例加权 BCE，辅助损失与其启用时序沿用。",
             "数据/SMILES 种子固定42；官方划分、词表、预处理和训练类别权重与上一轮逐项核对一致：train=4898、valid=2638、test=2719。", "",
             f"验证最优轮到第50轮 AUROC 下降 {row['validation_auc_drop_to_last']:.6f}；第21–50轮平均裁剪比例为 {phases['21-50']['mean_clip_fraction']:.2%}，平均裁剪前梯度范数为 {phases['21-50']['mean_preclip_norm']:.3f}。",
             f"此前种子43/44/45的测试 AUROC 为 {previous['test_auc_mean']:.6f} ± {previous['test_auc_sample_std']:.6f}（样本标准差）。本次与其种子不同，不能把差异全部归因于学习率和预热。",
             "整体 AUROC 与 assay 内 AUROC 衡量的排序范围不同；单类 assay 不计入组内 AUROC。原始配置曾参考测试集进行探索，本次也不构成全新独立保留集评估。", "",
             "![训练曲线](training_curves.png)", "", "![与此前曲线的描述性比较](baseline_comparison.png)", "",
             "![测试ROC](test_roc_curve.png)", "",
             "[逐轮训练表](seed_42/TRAINING_LOG.md) · [原始日志](seed_42/train.log) · [完整历史](seed_42/history.json) · [实际配置](seed_42/config.yaml)", "",
             "[验证最优权重](seed_42/best.pt) · [第50轮权重](seed_42/last.pt) · [完整指标](seed_42/metrics.json) · [结果CSV](results.csv) · [核验记录](result_audit.json)"]
    report = "\n".join(lines) + "\n"
    (root / "TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    (root / "MULTISEED_REPORT.md").write_text(report, encoding="utf-8")
    (root / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DrugOOD EC50 assay seed42</title>'
        '<style>body{font:16px/1.7 Arial;margin:35px}img{max-width:100%}</style>'
        f'<h1>DrugOOD EC50 assay：seed42 已完成50轮</h1><p>验证最优轮：{best["epoch"]}；'
        f'验证 AUROC：{metrics["best_validation_score"]:.6f}；测试 AUROC：{auc:.6f}。</p>'
        '<p><a href="TRAINING_REPORT.md">完整训练报告</a> · <a href="seed_42/TRAINING_LOG.md">逐轮训练日志</a></p>'
        '<img src="training_curves.png"><img src="baseline_comparison.png"><img src="test_roc_curve.png"></html>', encoding="utf-8")
    print(json.dumps({"result": row, "audit": audit, "gradient_phases": phases, "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
