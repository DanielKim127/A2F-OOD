# -*- coding: utf-8 -*-
"""Summarize real benchmark artifacts without fitting or selecting on test data."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import yaml


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def short(name):
    return name.replace("drugood_", "DrugOOD ").replace("admeood_", "ADMEOOD ").replace("_", " ").replace("ec50", "EC50").replace("ic50", "IC50").replace("ki", "KI")


def rows_from_suite(suite, manifest):
    rows = []
    for run in manifest["runs"]:
        folder = suite / run["name"]
        row = dict(run)
        config_path = folder / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
        row["selection_smiles_views"] = config.get("training", {}).get("validation_smiles_views", 1)
        row["evaluation_smiles_views"] = config.get("evaluation", {}).get("random_smiles_views", 5)
        row["hidden_dim"] = config.get("model", {}).get("hidden_dim")
        row["classification_weighting"] = config.get("loss", {}).get("classification_weighting", "none")
        prep = read_json(folder / "preprocessing_summary.json", {})
        retained = prep.get("retained_by_split", {})
        row.update(train_n=retained.get("train"), valid_n=retained.get("valid"),
                   test_n=retained.get("test"), skipped=prep.get("skipped_rows"))
        history = read_json(folder / "history.json", [])
        if history:
            row["initial_train_loss"] = history[0]["train"]["task"]
            row["final_train_loss"] = history[-1]["train"]["task"]
        metrics = read_json(folder / "metrics.json")
        if metrics:
            row["best_epoch"] = metrics["best_epoch"]
            row["selection_valid_auc"] = metrics["best_validation_score"]
            for split in ("valid", "test"):
                values = metrics[split]
                for key in ("roc_auc", "average_precision", "accuracy", "log_loss", "brier", "ece"):
                    row[f"{split}_{key}"] = values.get(key)
            groups = metrics["test"].get("group_metrics", {})
            row["test_worst_group_auc"] = groups.get("worst", {}).get("roc_auc")
            row["test_mean_group_auc"] = groups.get("mean", {}).get("roc_auc")
            reports = groups.get("groups", {})
            row["test_groups"] = len(reports)
            row["test_groups_with_auc"] = sum(r.get("roc_auc") is not None for r in reports.values())
            for branch in ("graph", "smiles"):
                row[f"test_{branch}_auc"] = metrics["test"]["unimodal_" + branch]["roc_auc"]
            with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            labels = np.array([float(r["label"]) for r in predictions])
            row["test_positive_rate"] = float(labels.mean())
            row["majority_accuracy"] = max(row["test_positive_rate"], 1 - row["test_positive_rate"])
            # A random ranking has expected AP equal to positive prevalence.
            row["ap_over_prevalence"] = row["test_average_precision"] - row["test_positive_rate"]
            if groups:
                group_rows = [{"group": group, **report} for group, report in reports.items()]
                if group_rows:
                    write_csv(folder / "test_group_metrics.csv", group_rows)
        rows.append(row)
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row if key != "command"))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(suite, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160})
    complete = [r for r in rows if r.get("test_roc_auc") is not None]
    if complete:
        fig, axes = plt.subplots(1, 2, figsize=(15, max(4, len(complete) * 0.48)), constrained_layout=True)
        y = np.arange(len(complete))
        labels = [short(r["name"]) for r in complete]
        axes[0].barh(y - .18, [r["valid_roc_auc"] for r in complete], .34, label="OOD validation (5 views)", color="#7aa6c2")
        axes[0].barh(y + .18, [r["test_roc_auc"] for r in complete], .34, label="OOD test (5 views)", color="#17628a")
        axes[0].set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel="ROC-AUC")
        axes[0].axvline(.5, color="#777777", ls="--", lw=1, label="Random ranking")
        axes[0].invert_yaxis(); axes[0].legend(loc="lower right", fontsize=8)
        for i, row in enumerate(complete):
            axes[0].text(min(row["test_roc_auc"] + .01, .91), i + .18, f'{row["test_roc_auc"]:.3f}', va="center", fontsize=9)
        axes[1].barh(y - .18, [r["test_average_precision"] for r in complete], .34, label="Test AP", color="#17628a")
        axes[1].barh(y + .18, [r["test_positive_rate"] for r in complete], .34, label="Test positive prevalence", color="#c7cdd2")
        axes[1].set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel="Average precision / positive prevalence")
        axes[1].invert_yaxis(); axes[1].legend(loc="lower right", fontsize=8)
        fig.suptitle("A2F-OOD official benchmark results | one seed (42)")
        fig.savefig(suite / "benchmark_overview.png"); plt.close(fig)
    active = [r for r in rows if (suite / r["name"] / "history.json").exists()]
    if active:
        for kind in ("loss", "validation_auc"):
            nrows = (len(active) + 1) // 2
            fig, axes = plt.subplots(nrows, 2, figsize=(15, 3.2 * nrows), squeeze=False, constrained_layout=True)
            for ax, row in zip(axes.flat, active):
                history = read_json(suite / row["name"] / "history.json", [])
                epochs = [h["epoch"] for h in history]
                if kind == "loss":
                    weighted = row["classification_weighting"] == "balanced"
                    ax.plot(epochs, [h["train"]["task"] for h in history],
                            label="Train balanced BCE" if weighted else "Train BCE (random views)", color="#17628a")
                    ax.plot(epochs, [h["valid"]["task_loss"] for h in history], label="OOD validation BCE", color="#ce7435")
                    if weighted:
                        ax.plot(epochs, [h["train"]["task_unweighted"] for h in history],
                                label="Train ordinary BCE", color="#489d73", ls="--")
                        ax.plot(epochs, [h["valid"]["class_weighted_task_loss"] for h in history],
                                label="OOD validation balanced BCE", color="#b08d3c", ls="--")
                    ax.set_ylabel("Supervised BCE")
                else:
                    ax.plot(epochs, [h["valid"]["roc_auc"] for h in history], color="#17628a",
                            label=f'Validation ROC-AUC ({row["selection_smiles_views"]} views)')
                    ax.axhline(.5, color="#999999", ls=":", lw=1)
                    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1)
                best = max(history, key=lambda h: h["valid"]["roc_auc"])
                ax.axvline(best["epoch"], color="#777777", ls="--", lw=1, label=f'Best checkpoint: epoch {best["epoch"]}')
                ax.set_title(short(row["name"])); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
                ax.grid(axis="y", alpha=.15)
            for ax in list(axes.flat)[len(active):]: ax.set_visible(False)
            fig.savefig(suite / (kind + "_curves.png")); plt.close(fig)
    for row in complete:
        folder = suite / row["name"]
        if (folder / "test_curves.png").exists():
            continue
        with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        labels = np.array([float(r["label"]) for r in predictions])
        p = np.array([float(r["prediction"]) for r in predictions])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        fpr, tpr, _ = roc_curve(labels, p)
        axes[0].plot(fpr, tpr, color="#17628a", label=f'ROC-AUC = {row["test_roc_auc"]:.4f}')
        axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="OOD test ROC"); axes[0].legend()
        precision, recall, _ = precision_recall_curve(labels, p)
        axes[1].plot(recall, precision, color="#17628a", label=f'AP = {row["test_average_precision"]:.4f}')
        axes[1].axhline(labels.mean(), color="#999999", ls="--", label=f'Prevalence = {labels.mean():.3f}')
        axes[1].set(xlabel="Recall", ylabel="Precision", title="OOD test precision-recall"); axes[1].legend()
        bins = np.minimum((p * 10).astype(int), 9)
        x, y = [], []
        for index in range(10):
            mask = bins == index
            if mask.any(): x.append(p[mask].mean()); y.append(labels[mask].mean())
        axes[2].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
        axes[2].plot(x, y, "o-", color="#17628a", label=f'ECE = {row["test_ece"]:.4f}')
        axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="OOD test calibration"); axes[2].legend()
        fig.suptitle(short(row["name"]) + " | best validation checkpoint | 5 SMILES views")
        fig.savefig(folder / "test_curves.png"); plt.close(fig)


def make_reports(suite, manifest, rows):
    status_names = {"queued": "等待", "running": "训练中", "paused": "已暂停", "completed": "已完成", "failed": "失败"}
    columns = [("数据版本", lambda r: short(r["name"])),
               ("状态", lambda r: status_names.get(r["status"], r["status"])),
               ("隐藏维度", lambda r: str(r.get("hidden_dim") or "—")),
               ("BCE 类别权重", lambda r: r["classification_weighting"]),
               ("训练/验证/测试", lambda r: "/".join(str(r.get(k) or "—") for k in ("train_n", "valid_n", "test_n"))),
               ("完成轮次", lambda r: str(r.get("epochs_completed", 0))),
               ("最佳轮次", lambda r: str(r.get("best_epoch", "—"))),
               ("最佳验证 AUC（选模）", lambda r: fmt(r.get("best_valid_auc"))),
               ("选模视图数", lambda r: str(r["selection_smiles_views"])),
               ("当前训练 BCE", lambda r: fmt(r.get("latest_train_loss"))),
               ("验证 AUC（5视图）", lambda r: fmt(r.get("valid_roc_auc"))),
               ("测试 AUC", lambda r: fmt(r.get("test_roc_auc"))),
               ("测试 AP", lambda r: fmt(r.get("test_average_precision"))),
               ("测试 Accuracy", lambda r: fmt(r.get("test_accuracy"))),
               ("正例比例", lambda r: fmt(r.get("test_positive_rate"))),
               ("测试 ECE", lambda r: fmt(r.get("test_ece")))]
    n = sum(r["status"] == "completed" for r in rows)
    notes = ["完整数据，官方 train/ood_val/ood_test 划分；单随机种子 42。",
             "完整 A2F-OOD：GINE、Transformer、双向跨注意力、门控、图到 SMILES 解码辅助损失与 GroupDRO；隐藏维度和特征定义以各任务运行时配置为准。",
             "最多 100 轮，patience=15；按固定视图的 OOD 验证 ROC-AUC 选 checkpoint；选模视图数见表，最终验证/测试使用 5 个 SMILES 视图平均。",
             "各任务的实际损失权重、GroupDRO 策略、学习率和 warmup 设置保存在各自 config.yaml；历史结果按运行时配置展示。",
             "原子上限 256、序列上限 512（含特殊 Token）；超限样本跳过并记录，无抽样缩减训练集。",
             "训练曲线的 train BCE 是随机增强视图上的在线训练损失；valid BCE 是固定视图的评估损失。总损失另含五项辅助目标及权重 warmup，不宜与 BCE 直接比较。",
             "类别比例变化较大：应结合 ROC-AUC、AP 与正例比例判断，Accuracy 不能单独代表模型效果。AP 指 average precision。",
             "单模态分支结果来自完整模型中的辅助头，并非独立训练的基线。单次运行不提供跨随机种子的均值或方差。",
             "最差组 AUC 仅统计含正负两类的组；小样本组的极值可能不稳定。"]
    md = ["# A2F-OOD 全部官方版本训练报告", "", f"已完成：{n}/{len(rows)}；更新：{manifest['updated_at']}", "", *[f"- {note}" for note in notes], "", "| " + " | ".join(k for k, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    md += ["| " + " | ".join(getter(row) for _, getter in columns) + " |" for row in rows]
    md += ["", "## 环境与分支诊断", "", "| 数据 | 图辅助头 AUC | SMILES 辅助头 AUC | 平均组 AUC | 最差组 AUC | 可计算 AUC 的组/全部组 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row.get("test_roc_auc") is not None:
            md.append("| " + " | ".join([short(row["name"]), fmt(row.get("test_graph_auc")), fmt(row.get("test_smiles_auc")), fmt(row.get("test_mean_group_auc")), fmt(row.get("test_worst_group_auc")), f'{row.get("test_groups_with_auc")}/{row.get("test_groups")}']) + " |")
    for image in ("benchmark_overview.png", "loss_curves.png", "validation_auc_curves.png"):
        if (suite / image).exists(): md += ["", f"![{image}]({image})"]
    md += ["", "每个数据目录保存 best.pt、history.json、metrics.json、预测 CSV、分组指标、预处理审计和 ROC/PR/校准曲线。", ""]
    (suite / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    table_head = "".join(f"<th>{html.escape(k)}</th>" for k, _ in columns)
    table_rows = "".join("<tr>" + "".join(f"<td>{html.escape(getter(row))}</td>" for _, getter in columns) + "</tr>" for row in rows)
    figures = "".join(f'<figure><img src="{name}" alt="{name}"></figure>' for name in ("benchmark_overview.png", "loss_curves.png", "validation_auc_curves.png") if (suite / name).exists())
    links = "".join(f'<li>{html.escape(short(r["name"]))}：<a href="{r["name"]}/train.log">训练日志</a> · <a href="{r["name"]}/history.json">训练历史</a>' + (f' · <a href="{r["name"]}/metrics.json">最终指标</a> · <a href="{r["name"]}/test_curves.png">ROC/PR/校准曲线</a>' if r.get("test_roc_auc") is not None else "") + "</li>" for r in rows)
    live = "<meta http-equiv='refresh' content='30'>" if manifest["status"] == "running" else ""
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">{live}<title>A2F-OOD 训练结果</title>
<style>body{{font:15px/1.65 "Microsoft YaHei",Arial,sans-serif;color:#182c3c;background:#f5f7fa;margin:0}}main{{max-width:1500px;margin:auto;padding:32px}}h1{{margin:0}}.status{{color:#17628a;font-size:20px}}.scroll{{overflow:auto;background:white;padding:12px;border-radius:8px}}table{{border-collapse:collapse;white-space:nowrap;width:100%;font-size:13px}}th,td{{text-align:right;padding:10px;border-bottom:1px solid #e5e9ed}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;height:auto}}figure{{margin:24px 0;background:white;padding:12px}}a{{color:#17628a}}li{{margin:5px 0}}.notes{{max-width:1100px}}</style>
<main><h1>A2F-OOD 官方数据训练结果</h1>
<p class="status">已完成 {n} / {len(rows)} · {html.escape(manifest['status'])}</p>
<p>更新：{html.escape(manifest['updated_at'])}</p>
<div class="scroll"><table><thead><tr>{table_head}</tr></thead><tbody>{table_rows}</tbody></table></div>
{figures}<h2>实验口径</h2><ul class="notes">{''.join('<li>'+html.escape(note)+'</li>' for note in notes)}</ul>
<h2>详细结果</h2><ul>{links}</ul>
<p><a href="summary.csv">下载汇总 CSV</a> · <a href="REPORT.md">文字报告</a> · <a href="suite_status.json">运行状态</a></p>
</main></html>'''
    (suite / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    suite = Path(args.suite).resolve()
    manifest = read_json(suite / "suite_status.json")
    rows = rows_from_suite(suite, manifest)
    write_csv(suite / "summary.csv", rows)
    plotted = read_json(suite / "plot_status.json", {})
    epochs = {row["name"]: row.get("epochs_completed", 0) for row in rows}
    # Keep the live report useful without redrawing every polling interval.
    if args.figures or any(epoch - plotted.get(name, 0) >= 5 for name, epoch in epochs.items()):
        make_figures(suite, rows)
        (suite / "plot_status.json").write_text(json.dumps(epochs), encoding="utf-8")
    make_reports(suite, manifest, rows)


if __name__ == "__main__":
    main()
