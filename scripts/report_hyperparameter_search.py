"""Audit a frozen search and plot saved predictions without reevaluating models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, log_loss, matthews_corrcoef,
                             roc_auc_score, roc_curve)


ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
    "Previous ordinary BCE": ROOT / "outputs/drugood_ec50_assay_stable_20261119_01/drugood_ec50_assay",
    "Previous balanced BCE": ROOT / "outputs/drugood_ec50_assay_balanced_bce_20261119_01/drugood_ec50_assay",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    return records, np.array([float(r["label"]) for r in records]), np.array([float(r["prediction"]) for r in records])


def summarize(y, p):
    cm = confusion_matrix(y, p >= .5, labels=[0, 1])
    return dict(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                accuracy=float(np.mean(y == (p >= .5))), log_loss=float(log_loss(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
                mcc=float(matthews_corrcoef(y, p >= .5)), confusion_matrix=cm.tolist(),
                negative_recall=float(cm[0, 0] / cm[0].sum()), positive_recall=float(cm[1, 1] / cm[1].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    state, metrics, frozen = [read(root / name) for name in ("search_status.json", "final_metrics.json", "frozen_selection.json")]
    assert state["status"] == "completed" and state["test_evaluation_count"] == 1
    assert state["test_used_for_search"] is False
    ranked = sorted([t for t in state["trials"] if t["status"] == "completed"], key=lambda t: t["valid_auc"], reverse=True)
    winner = ranked[0]
    assert winner["id"] == metrics["trial"] == frozen["trial"]
    assert hashlib.sha256((root / "best.pt").read_bytes()).hexdigest() == frozen["checkpoint_sha256"]
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src/a2f_ood").glob("*.py")}
    assert source == state["source_hashes"]
    folder = Path(winner["directory"])
    runtime = read(folder / "runtime.json")
    assert runtime["source_files_sha256"] == source
    assert runtime["feature_schema"]["atom_dim"] == 157 and runtime["feature_schema"]["bond_dim"] == 14
    manifest = read(root / "shared_preprocessing/split_manifest.json")
    weights = read(folder / "class_weighting.json")
    assert weights["fitted_on"] == "retained_training_labels_only"
    assert weights["mode"] == "balanced"
    # Check every completed trial used the same examples, objective and source.
    for trial in ranked:
        trial_root = Path(trial["directory"])
        assert read(trial_root / "split_manifest.json") == manifest
        assert read(trial_root / "class_weighting.json") == weights
        assert read(trial_root / "runtime.json")["source_files_sha256"] == source
        assert not (trial_root / "test_predictions.csv").exists()
        assert read(trial_root / "selection_metrics.json")["test_evaluated"] is False

    records, y, p = predictions(root / "final_test_predictions.csv")
    assert [int(r["row_id"]) for r in records] == [r["row_id"] for r in manifest["test"]]
    checks = summarize(y, p)
    for key in ("roc_auc", "average_precision", "accuracy", "log_loss", "balanced_accuracy", "mcc"):
        assert np.isclose(checks[key], metrics["test"][key], atol=1e-7)
    comparisons = {}
    probability_vectors = {}
    for name, directory in BASELINES.items():
        baseline_records, baseline_y, baseline_p = predictions(directory / "test_predictions.csv")
        assert [r["row_id"] for r in baseline_records] == [r["row_id"] for r in records]
        assert np.array_equal(baseline_y, y)
        saved = read(directory / "metrics.json")
        comparisons[name] = {"valid_auc": saved["valid"]["roc_auc"], **summarize(y, baseline_p)}
        probability_vectors[name] = baseline_p
    comparisons["Selected search model"] = {"valid_auc": metrics["valid"]["roc_auc"], **checks}

    rng = np.random.RandomState(20261120)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    bootstrap_auc, bootstrap_delta = [], []
    previous = probability_vectors["Previous balanced BCE"]
    for _ in range(args.bootstrap):
        indices = np.concatenate((rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)))
        auc = roc_auc_score(y[indices], p[indices])
        bootstrap_auc.append(auc)
        bootstrap_delta.append(auc - roc_auc_score(y[indices], previous[indices]))
    confidence = {"method": "stratified paired bootstrap of saved test rows; conditional on selected model",
                  "seed": 20261120, "resamples": args.bootstrap,
                  "test_auc_95_percentile_interval": np.percentile(bootstrap_auc, [2.5, 97.5]).tolist(),
                  "auc_delta_vs_previous_balanced_95_interval": np.percentile(bootstrap_delta, [2.5, 97.5]).tolist()}
    audit = {"passed": True, "completed_trials_checked": len(ranked), "same_source_hashes": True,
             "same_split_manifests": True, "same_train_fitted_balanced_weights": True,
             "no_test_predictions_in_trials": True, "frozen_checkpoint_hash_matches": True,
             "recomputed_saved_test_metrics": checks, "test_rows": len(y),
             "test_negatives": len(negative), "test_positives": len(positive),
             "comparisons": comparisons, "confidence_intervals": confidence}
    (root / "result_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    config = yaml.safe_load((root / "best_config.yaml").read_text(encoding="utf-8"))
    config["experiment"].update(name="drugood_ec50_assay_search_reproduce", output_dir=str(
        ROOT / "outputs" / ("drugood_ec50_assay_search_reproduce_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
    (root / "rerun_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    history = read(folder / "history.json")
    colors = {"adaptive": "#17628a", "concat": "#489d73", "graph": "#b08d3c", "smiles": "#bb6b52"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for mode, color in colors.items():
        trials = [t for t in ranked if t["changes"]["fusion.mode"] == mode]
        axes[0, 0].scatter([int(t["id"].split("_")[0]) for t in trials], [t["valid_auc"] for t in trials], label=mode, color=color)
    axes[0, 0].scatter([int(winner["id"].split("_")[0])], [winner["valid_auc"]], marker="*", s=180, color="#cc3344", label="selected")
    axes[0, 0].set(title="Validation search (5 fixed views)", xlabel="Trial", ylabel="Validation ROC-AUC")
    axes[0, 0].legend(fontsize=8)
    epochs = [h["epoch"] for h in history]
    axes[0, 1].plot(epochs, [h["valid"]["roc_auc"] for h in history], color="#17628a", label="Validation AUC (3 views)")
    axes[0, 1].axvline(metrics["best_epoch"], color="#cc3344", ls="--", label=f"Saved epoch {metrics['best_epoch']}")
    axes[0, 1].set(title=f"Selected model: {winner['id']}", xlabel="Epoch", ylabel="Validation ROC-AUC")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(epochs, [h["train"]["task"] for h in history], label="Training balanced BCE")
    axes[1, 0].plot(epochs, [h["valid"]["class_weighted_task_loss"] for h in history], label="Validation balanced BCE")
    axes[1, 0].set(title="Classification loss with identical class weights", xlabel="Epoch", ylabel="Balanced BCE")
    axes[1, 0].legend(fontsize=8)
    for name, vector in probability_vectors.items():
        fpr, tpr, _ = roc_curve(y, vector)
        axes[1, 1].plot(fpr, tpr, lw=1.3, label=f"{name}: {roc_auc_score(y, vector):.4f}")
    fpr, tpr, _ = roc_curve(y, p)
    axes[1, 1].plot(fpr, tpr, lw=2.2, color="#cc3344", label=f"Selected model: {checks['roc_auc']:.4f}")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="#aaa")
    axes[1, 1].set(title="Final OOD test: saved predictions", xlabel="False positive rate", ylabel="True positive rate")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(root / "training_effects.png", dpi=180)
    plt.close(fig)

    changes = winner["changes"]
    lo, hi = confidence["test_auc_95_percentile_interval"]
    delta = checks["roc_auc"] - comparisons["Previous balanced BCE"]["roc_auc"]
    counts = dict(Counter(t["changes"]["fusion.mode"] for t in ranked))
    lines = ["# DrugOOD EC50 assay 广泛超参数搜索", "",
             f"最终选择 `{winner['id']}`，验证 ROC-AUC **{metrics['valid']['roc_auc']:.6f}**，测试 ROC-AUC **{checks['roc_auc']:.6f}**。",
             f"测试 AUC 约 0.75 的目标：**{'达到' if metrics['target_reached'] else '未达到'}**（预先约定近似判断 >=0.745）；比上一轮加权 BCE 的测试 AUC 变化 **{delta:+.6f}**。",
             "", f"完成 {len(ranked)} 个训练任务，累计 {sum(t['epochs'] for t in ranked)} 轮；另有 {sum(t['status'] == 'interrupted' for t in state['trials'])} 个中断任务，检查点保留。结构分布：`{json.dumps(counts)}`。",
             "仅按验证 AUC 搜索，所有候选保留训练集拟合的类别加权 BCE。冻结检查点的 SHA-256 后，仅对所选模型评估测试集一次。", "",
             "| 实验 | 验证 ROC-AUC | 测试 ROC-AUC | 测试 AP | 准确率 (0.5) | 平衡准确率 (0.5) | Log loss |", "|---|---:|---:|---:|---:|---:|---:|"]
    names = {"Previous ordinary BCE": "原普通 BCE", "Previous balanced BCE": "上一轮加权 BCE", "Selected search model": "本次搜索最优模型"}
    for name, values in comparisons.items():
        lines.append(f"| {names[name]} | {values['valid_auc']:.6f} | {values['roc_auc']:.6f} | {values['average_precision']:.6f} | {values['accuracy']:.6f} | {values['balanced_accuracy']:.6f} | {values['log_loss']:.6f} |")
    lines += ["", f"固定阈值 0.5：负类召回率 {checks['negative_recall']:.6f}，正类召回率 {checks['positive_recall']:.6f}，MCC {checks['mcc']:.6f}。混淆矩阵（行真实、列预测，顺序0/1）：`{checks['confusion_matrix']}`。",
              f"测试集负例 {len(negative)}、正例 {len(positive)}，正例占 {y.mean():.2%}。AP 和准确率需结合这一比例解读。",
              f"对固定所选模型做 {args.bootstrap} 次分层 bootstrap，测试 AUC 的 95% 百分位区间为 [{lo:.4f}, {hi:.4f}]；此区间不包含搜索或不同种子产生的变异。", "",
              "## 所选模型与训练参数", "",
              f"融合模式 `{changes['fusion.mode']}`，hidden={changes['model.hidden_dim']}，图编码层数={changes['model.graph_layers']}，SMILES层数={changes['model.smiles_layers']}，attention heads={changes['model.attention_heads']}，dropout={changes['model.dropout']:g}。",
              f"学习率={changes['training.learning_rate']:g}，weight decay={changes['training.weight_decay']:g}，batch size={changes['training.batch_size']}，最佳第 {metrics['best_epoch']} 轮。",
              "原子属性157维、化学键属性14维，先投影到所选hidden维度。", ""]
    if changes["fusion.mode"] == "graph":
        auxiliary = any(value > 0 for key, value in changes.items() if key.startswith("loss.lambda_"))
        lines += ["主分类预测使用图分支。" +
                  ("SMILES编码与非零辅助目标仍参与训练；这是在原框架中选择的图分支预测配置。" if auxiliary else
                   "此配置的辅助目标权重均为0，训练监督来自图分支分类目标。"), ""]
    lines += ["```json", json.dumps(changes, indent=2, ensure_ascii=False), "```", "",
              "## 数据、搜索范围与核验", "",
              "采用官方划分，过滤后 train=4898、valid=2638、test=2719；209条超出既定原子数/SMILES长度限制的样本被过滤，与前两轮对照一致。",
              "训练集负例1177、正例3721，类别权重 w_negative=2.080713679、w_positive=0.658156410；权重仅由训练标签拟合。",
              "训练每分子2个SMILES视图，每轮选模使用验证集3个固定视图，候选排序及最终验证/测试使用5个固定视图。seed=42。", "",
              "| 参数 | 实际完成搜索的取值 |", "|---|---|"]
    for key in ("model.hidden_dim", "model.graph_layers", "model.smiles_layers", "model.dropout", "training.learning_rate", "training.weight_decay", "training.batch_size", "loss.group_dro_strength"):
        values = sorted({t["changes"][key] for t in ranked})
        lines.append(f"| {key} | {', '.join(f'{v:g}' for v in values)} |")
    lines += ["", "采用24组初筛、12组局部细化，再对前三名延长训练的方案；完整参数和得分见排行榜。",
              "已逐一核对完成任务的训练源码哈希、划分清单、类别权重；各试验均未导出测试预测。最终检查点哈希一致，测试指标已从保存的预测文件独立重算，且与报告一致。",
              "同配置同种子重跑时，曾出现首轮接近、后续训练曲线分叉的现象，表明训练对细微数值差异仍敏感；本次使用实际验证分数最高的已保存检查点。CUDA图聚合存在非确定性操作，但尚未把曲线差异归因到单一原因。",
              "长时间训练过程中观察到GPU温度降频，以及eFootballOnline.exe占用约87%–95%图形引擎的资源竞争；最后三组改用独立训练进程，保留中断记录和检查点。",
              "当前结果为单种子探索结果，广泛使用同一验证集也可能产生选择偏差；既有历史测试结果已知。", "",
              f"所选模型的加权训练 BCE 从 {history[0]['train']['task']:.6f} 变为 {history[-1]['train']['task']:.6f}；验证 AUC 从 {history[0]['valid']['roc_auc']:.6f} 变为 {history[-1]['valid']['roc_auc']:.6f}（均按历史记录末轮，最佳检查点另标出）。AUC 选模优化排序，验证 log loss 未必同步改善。", "",
              "| 预测模式 | 该模式最高验证 AUC | 试验 |", "|---|---:|---|"]
    for mode in colors:
        candidates = [t for t in ranked if t["changes"]["fusion.mode"] == mode]
        if candidates:
            best = candidates[0]
            lines.append(f"| {mode} | {best['valid_auc']:.6f} | {best['id']} |")
    lines += ["", "其余模式未追加测试评估；表中数值均为验证结果。", "",
              "![训练与效果对比](training_effects.png)", "", "![测试ROC与PR](final_test_curves.png)", "",
              "[完整排行榜](leaderboard.csv) · [完整指标](final_metrics.json) · [核验结果与区间](result_audit.json) · [原始最佳配置](best_config.yaml) · [重新训练配置](rerun_config.yaml) · [冻结选择记录](frozen_selection.json)", "",
              "重新训练命令（配置已指向新的输出目录）：", "", "```powershell",
              f'.\\.venv\\Scripts\\python.exe -m a2f_ood.train --config "{root / "rerun_config.yaml"}" --selection-only',
              "```", "同一种子重跑仍可能有CUDA数值差异；直接复核本次结果请使用保存的 `best.pt` 与预测文件。", ""]
    (root / "SEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"trial": winner["id"], "metrics": checks, "confidence": confidence}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
