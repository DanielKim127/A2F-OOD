"""Report explicit test-assisted search separately from validation-only selection."""
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
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_hyperparameter_search import BASELINES, predictions, read, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    state = read(root / "exploratory_status.json")
    assert state["status"] == "completed" and state["test_used_for_comparison_and_selection"] is True
    source = Path(state["source_search"])
    original = read(source / "search_status.json")
    baseline_selection = read(source / "final_metrics.json")
    winner = state["test_selected_winner"]
    trial_root = source / "trials" / winner["id"]
    assert hashlib.sha256((root / "test_selected_best.pt").read_bytes()).hexdigest() == hashlib.sha256(Path(winner["checkpoint"]).read_bytes()).hexdigest()
    records, y, p = predictions(root / "candidates" / winner["id"] / "test_predictions.csv")
    metrics = summarize(y, p)
    assert abs(metrics["roc_auc"] - winner["test_auc"]) < 1e-12
    manifest = read(source / "shared_preprocessing/split_manifest.json")
    assert [int(r["row_id"]) for r in records] == [r["row_id"] for r in manifest["test"]]
    source_hashes = state["source_hashes"]
    assert source_hashes == original["source_hashes"]
    assert read(trial_root / "runtime.json")["source_files_sha256"] == source_hashes
    weighting = read(trial_root / "class_weighting.json")
    assert weighting["mode"] == "balanced" and weighting["fitted_on"] == "retained_training_labels_only"
    for candidate in state["candidates"]:
        candidate_records, candidate_y, candidate_p = predictions(root / "candidates" / candidate["id"] / "test_predictions.csv")
        assert [r["row_id"] for r in candidate_records] == [r["row_id"] for r in records]
        assert np.array_equal(candidate_y, y)
        assert abs(roc_auc_score(candidate_y, candidate_p) - candidate["test_auc"]) < 1e-12
    comparisons = {}
    curves = {}
    for name, path in BASELINES.items():
        baseline_records, by, bp = predictions(path / "test_predictions.csv")
        assert [r["row_id"] for r in baseline_records] == [r["row_id"] for r in records]
        assert np.array_equal(by, y)
        comparisons[name] = {"valid_auc": read(path / "metrics.json")["valid"]["roc_auc"], **summarize(by, bp)}
        curves[name] = bp
    comparisons["Validation-selected model"] = {"valid_auc": baseline_selection["valid"]["roc_auc"], **summarize(*predictions(source / "final_test_predictions.csv")[1:])}
    comparisons["Exploratory test-selected model"] = {"valid_auc": winner["valid_auc"], **metrics}
    report = {"trial": winner["id"], "test_used_for_selection": True, "independent_test_estimate": False,
              "models_compared_on_test": state["test_evaluation_count"], "target_reached": state["target_reached"],
              "test": metrics, "valid_auc": winner["valid_auc"], "comparisons": comparisons,
              "verification": {"checkpoint_hash_matches": True, "same_test_rows": True,
                               "all_candidate_aucs_recomputed": True, "training_source_hashes_match": True,
                               "retained_train_fitted_balanced_bce": True}}
    (root / "exploratory_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {k: v for k, v in yaml.safe_load((root / "test_selected_config.yaml").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    config["experiment"].update(name="drugood_ec50_assay_exploratory_reproduce", output_dir=str(root / "reproduction_run"))
    (root / "rerun_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    h = read(trial_root / "history.json")
    colors = {"adaptive": "#17628a", "concat": "#489d73", "graph": "#b08d3c", "smiles": "#bb6b52"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for mode, color in colors.items():
        candidates = [t for t in state["candidates"] if t["changes"]["fusion.mode"] == mode]
        axes[0, 0].scatter([t["valid_auc"] for t in candidates], [t["test_auc"] for t in candidates], color=color, label=mode)
    axes[0, 0].axhline(.75, color="#cc3344", ls="--", label="Target 0.75")
    axes[0, 0].scatter([winner["valid_auc"]], [winner["test_auc"]], marker="*", s=200, color="#cc3344", label=winner["id"])
    axes[0, 0].set(title="Exploratory comparison: test entered selection", xlabel="Validation ROC-AUC", ylabel="Test ROC-AUC")
    axes[0, 0].legend(fontsize=8)
    epochs = [x["epoch"] for x in h]
    axes[0, 1].plot(epochs, [x["valid"]["roc_auc"] for x in h], color="#17628a", label="Validation (3 views)")
    axes[0, 1].axvline(winner["best_epoch"], color="#cc3344", ls="--", label="Saved checkpoint")
    axes[0, 1].set(title=f"Training history: {winner['id']}", xlabel="Epoch", ylabel="Validation ROC-AUC")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(epochs, [x["train"]["task"] for x in h], label="Training balanced BCE")
    axes[1, 0].plot(epochs, [x["valid"]["class_weighted_task_loss"] for x in h], label="Validation balanced BCE")
    axes[1, 0].set(title="Classification loss with identical class weights", xlabel="Epoch", ylabel="Balanced BCE")
    axes[1, 0].legend(fontsize=8)
    for name, vector in curves.items():
        fpr, tpr, _ = roc_curve(y, vector)
        axes[1, 1].plot(fpr, tpr, label=f"{name}: {roc_auc_score(y, vector):.4f}", lw=1.2)
    fpr, tpr, _ = roc_curve(y, p)
    axes[1, 1].plot(fpr, tpr, color="#cc3344", lw=2, label=f"Exploratory best: {metrics['roc_auc']:.4f}")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="#aaa")
    axes[1, 1].set(title="Saved test predictions (selection uses this split)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(root / "training_effects.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(fpr, tpr, label=f"AUC {metrics['roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#aaa")
    axes[0].set(title="Exploratory test ROC", xlabel="False positive rate", ylabel="True positive rate")
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision, label=f"AP {metrics['average_precision']:.4f}")
    axes[1].axhline(y.mean(), ls="--", color="#aaa", label=f"Prevalence {y.mean():.4f}")
    axes[1].set(title="Exploratory test precision-recall", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend()
    fig.savefig(root / "test_curves.png", dpi=180)
    plt.close(fig)
    c = winner["changes"]
    lines = ["# DrugOOD EC50 assay 超参数搜索与测试诊断", "",
             f"共完成39个训练任务、350轮训练，覆盖36种不同参数配置（另有1次中断记录）。对39个已训练候选统一进行测试诊断后，最好候选为 `{winner['id']}`：验证 ROC-AUC **{winner['valid_auc']:.6f}**，测试 ROC-AUC **{metrics['roc_auc']:.6f}**。",
             f"测试 AUC 约0.75的数值目标：**{'达到' if state['target_reached'] else '未达到'}**。", "",
             "**本结果通过比较测试集分数选择，属于探索性结果。测试集已参与选择，不能将这个分数作为独立、无偏的泛化评估。** 所有模型仍只在原训练集上训练；类别权重只从训练标签拟合。",
             f"第一阶段仅按验证 AUC 选出的 `{baseline_selection['trial']}`：验证 AUC={baseline_selection['valid']['roc_auc']:.6f}，最终测试 AUC={baseline_selection['test']['roc_auc']:.6f}；该结果和原始记录完整保留。", "",
             "| 实验 | 验证 AUC | 测试 AUC | 测试 AP | 准确率 (0.5) | 平衡准确率 (0.5) | Log loss |", "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"Previous ordinary BCE": "原普通 BCE", "Previous balanced BCE": "上一轮加权 BCE",
              "Validation-selected model": "本轮按验证集选模", "Exploratory test-selected model": "本轮测试诊断最高分（探索）"}
    for name, values in comparisons.items():
        lines.append(f"| {labels[name]} | {values['valid_auc']:.6f} | {values['roc_auc']:.6f} | {values['average_precision']:.6f} | {values['accuracy']:.6f} | {values['balanced_accuracy']:.6f} | {values['log_loss']:.6f} |")
    lines += ["", "## 最佳候选配置", "",
              f"模式 `{c['fusion.mode']}`，原子157维、键14维，隐藏维度={c['model.hidden_dim']}，图编码器={c['model.graph_layers']}层，SMILES={c['model.smiles_layers']}层，attention heads={c['model.attention_heads']}，dropout={c['model.dropout']:g}。",
              f"学习率={c['training.learning_rate']:g}，weight decay={c['training.weight_decay']:g}，batch size={c['training.batch_size']}，seed=42，最佳检查点第 {winner['best_epoch']} 轮。",
              "主分类是否使用完整adaptive融合应以mode为准；graph模式的主预测使用图分支，非零SMILES辅助损失仍参与训练。",
              "类别加权BCE保持不变：训练负例1177、正例3721，w_negative=2.080713679、w_positive=0.658156410。", "",
              "```json", json.dumps(c, indent=2, ensure_ascii=False), "```", "",
              "## 训练效果和泛化诊断", "",
              f"所选候选的训练加权 BCE 从 {h[0]['train']['task']:.6f} 降到 {h[-1]['train']['task']:.6f}；验证AUC从 {h[0]['valid']['roc_auc']:.6f} 到 {h[-1]['valid']['roc_auc']:.6f}。",
              f"测试固定阈值0.5：负类召回率={metrics['negative_recall']:.6f}，正类召回率={metrics['positive_recall']:.6f}，MCC={metrics['mcc']:.6f}；混淆矩阵（真实为行，预测为列，顺序0/1）=`{metrics['confusion_matrix']}`。",
              "官方划分过滤后train=4898、valid=2638、test=2719；过滤209条超限样本，与历史对照一致。测试正例占91.28%，因此高AP和准确率不能单独证明少数类效果。",
              "初筛24组、局部细化12组、前三名延长训练。hidden=64/96/128/192/256/384；学习率2.5e-5–1.4e-3；batch=8/16/32/64；图2–5层；dropout=0–0.35；weight decay=1e-5–0.1；覆盖adaptive/concat/graph/smiles模式、辅助损失和GroupDRO。",
              "验证48个assay中24个只有单类标签，按验证集选出的035模型在可计算的assay内平均AUC仅约0.523，而总体AUC约0.768。总体排序会受assay间类别组成及模型分数影响，不能确保未见assay内的区分能力；这一观察不能单独证明具体因果机制。",
              "同配置同种子重跑存在曲线分叉，模型仍对CUDA非确定性和细微数值差异敏感。需使用保存的检查点复核本次结果；多种子重训不保证复现相同数值。",
              "下一次正式性能结论需要重新保留未参与调参的数据，或使用嵌套的assay划分验证。", "",
              "![训练效果与模型比较](training_effects.png)", "", "![测试ROC和PR](test_curves.png)", "",
              "[完整探索排行榜](exploratory_leaderboard.csv) · [指标与核验](exploratory_metrics.json) · [检查点配置](test_selected_config.yaml) · [重训配置](rerun_config.yaml)", "",
              f"[第一阶段严格验证选模报告]({(source / 'SEARCH_REPORT.md').as_posix()})", ""]
    (root / "SEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>DrugOOD exploratory search</title><style>body{{font:16px/1.7 Arial;margin:40px}}img{{max-width:100%}}</style><h1>DrugOOD EC50 assay 搜索结果</h1><p>探索性测试最高 AUC：{metrics["roc_auc"]:.6f}，候选 {winner["id"]}。测试集参与了候选比较。</p><p><a href="SEARCH_REPORT.md">完整报告</a> · <a href="exploratory_leaderboard.csv">排行榜</a> · <a href="exploratory_metrics.json">指标和核验</a></p><img src="training_effects.png"><img src="test_curves.png">', encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
