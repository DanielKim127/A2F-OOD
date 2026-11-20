"""Retrain a fixed configuration with new seeds and validation-only checkpoints.

The reference split, vocabulary and SMILES seed remain fixed. Each independent
worker trains the full epoch budget before one evaluation of its best checkpoint.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from a2f_ood.config import load_config, resolve_path
from a2f_ood.engine import (build_model, evaluate_model, get_device, json_safe,
                           load_checkpoint, make_loader, prepare_data,
                           seed_everything, write_predictions)
from a2f_ood.train import main as train_main


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {p.name: digest(p) for p in (ROOT / "src/a2f_ood").glob("*.py")}


def worker(args):
    config = load_config(args.worker_config)
    folder = resolve_path(config, config["experiment"]["output_dir"])
    reference = Path(args.reference_run).resolve()
    expected = read(folder.parent / "experiment_plan.json")
    assert sources() == expected["training_source_hashes"]
    assert digest(resolve_path(config, config["data"]["csv_path"])) == expected["input_csv_sha256"]
    # Data randomness is held fixed; train_main resets the training RNG afterwards.
    preprocessing = copy.deepcopy(config)
    preprocessing["experiment"]["seed"] = args.data_seed
    seed_everything(args.data_seed, config["training"]["cpu_threads"])
    prepared = prepare_data(preprocessing, folder)
    for filename in ("split_manifest.json", "vocabulary.json", "preprocessing_summary.json"):
        if read(folder / filename) != read(reference / filename):
            raise RuntimeError(f"Fixed reference mismatch: {filename}")
    datasets, vocabulary, _ = prepared
    save(folder / "fixed_data_audit.json", {
        "reference_run": str(reference), "data_and_smiles_seed": args.data_seed,
        "model_dropout_shuffle_seed": config["experiment"]["seed"],
        "split_manifest_matches": True, "vocabulary_matches": True,
        "preprocessing_matches": True, "input_csv_sha256": expected["input_csv_sha256"],
        "dataset_seeds": {name: dataset.seed for name, dataset in datasets.items()},
        "retained_by_split": {name: len(dataset) for name, dataset in datasets.items()}})
    seed_everything(config["experiment"]["seed"], config["training"]["cpu_threads"])
    initial = build_model(config, vocabulary)
    state_hash = hashlib.sha256()
    for name, tensor in initial.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    save(folder / "initialization_audit.json", {
        "seed": config["experiment"]["seed"], "initial_model_state_sha256": state_hash.hexdigest(),
        "trainable_parameters": sum(p.numel() for p in initial.parameters() if p.requires_grad),
        "loaded_pretrained_weights": False})
    del initial
    assert config["training"]["patience"] > config["training"]["epochs"]
    if train_main(["--config", args.worker_config, "--selection-only"], prepared_data=prepared):
        raise RuntimeError("Training returned a nonzero status")
    history = read(folder / "history.json")
    assert len(history) == config["training"]["epochs"]
    assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))
    selection = read(folder / "selection_metrics.json")
    assert selection["test_evaluated"] is False and "test" not in selection
    assert not (folder / "test_predictions.csv").exists()
    best_row = max(history, key=lambda row: row["valid"]["roc_auc"])
    assert best_row["epoch"] == selection["best_epoch"]
    assert read(folder / "class_weighting.json") == read(reference / "class_weighting.json")
    checkpoint_path = folder / "best.pt"
    save(folder / "frozen_selection.json", {
        "frozen_at": now(), "seed": config["experiment"]["seed"],
        "selection": "maximum per-epoch validation ROC-AUC, with 3 fixed views",
        "best_epoch": selection["best_epoch"], "best_validation_score": selection["best_validation_score"],
        "checkpoint_sha256": digest(checkpoint_path), "test_evaluated_before_selection": False})
    device = get_device(config["training"]["device"])
    model, checkpoint_vocab, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]  # Includes the class weights fitted by train_main.
    assert checkpoint["epoch"] == best_row["epoch"]
    assert checkpoint_vocab.to_dict() == vocabulary.to_dict()
    datasets["test"].num_views = config["evaluation"]["random_smiles_views"]
    test_metrics, rows = evaluate_model(model, make_loader(datasets["test"], config), config, device, vocabulary=vocabulary)
    write_predictions(folder / "test_predictions.csv", rows)
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    labels = np.array([float(row["label"]) for row in rows])
    probabilities = np.array([float(row["prediction"]) for row in rows])
    recomputed = float(roc_auc_score(labels, probabilities))
    assert abs(recomputed - test_metrics["roc_auc"]) < 1e-10
    test_metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, probabilities >= .5))
    assert [row["row_id"] for row in rows] == [row["row_id"] for row in read(reference / "split_manifest.json")["test"]]
    assert sources() == expected["training_source_hashes"]
    reports = {**selection, "test": test_metrics, "test_evaluated": True,
               "test_evaluation_count": 1, "epochs_completed": len(history),
               "early_stopping_enabled": False, "data_seed": args.data_seed,
               "seed": config["experiment"]["seed"], "test_auc_recomputed": recomputed,
               "finished_at": now()}
    save(folder / "metrics.json", reports)
    print(f"COMPLETE seed={config['experiment']['seed']} epochs={len(history)} best_epoch={selection['best_epoch']} "
          f"valid_auc={selection['valid']['roc_auc']:.6f} test_auc={test_metrics['roc_auc']:.6f}", flush=True)


def progress(root, state):
    state["updated_at"] = now()
    save(root / "experiment_status.json", state)
    rows = "<tr><th>Seed</th><th>Status</th><th>Best epoch</th><th>Valid AUC</th><th>Test AUC</th></tr>"
    for run in state["runs"]:
        values = [run["seed"], run["status"], run.get("best_epoch", ""), run.get("valid_auc", ""), run.get("test_auc", "")]
        rows += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>"
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>DrugOOD fixed-split multi-seed training</title><style>body{{font:15px/1.7 Arial;margin:35px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}img{{max-width:100%}}</style>
<h1>DrugOOD EC50 assay 多种子复验</h1><p>状态：{state['status']}；每次完整训练 {state['epochs']} 轮，按验证 AUC 选择检查点。</p><table>{rows}</table>
<p><a href="MULTISEED_REPORT.md">完整报告</a> · <a href="experiment_status.json">运行状态</a></p>'''
    if (root / "training_curves.png").exists():
        page += '<img src="training_curves.png">'
    (root / "index.html").write_text(page, encoding="utf-8")


def report(root, state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    completed = [run for run in state["runs"] if run["status"] == "completed"]
    if not completed:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for run in completed:
        folder = Path(run["directory"])
        history = read(folder / "history.json")
        epochs = [h["epoch"] for h in history]
        label = f"Seed {run['seed']} (best epoch {run['best_epoch']})"
        line, = axes[0, 0].plot(epochs, [h["valid"]["roc_auc"] for h in history], label=label)
        axes[0, 0].scatter([run["best_epoch"]], [run["best_validation_score"]], color=line.get_color(), marker="*", s=120)
        axes[0, 1].plot(epochs, [h["train"]["task"] for h in history], label=f"Seed {run['seed']}")
        axes[1, 0].plot(epochs, [h["valid"]["class_weighted_task_loss"] for h in history], label=f"Seed {run['seed']}")
        axes[1, 1].step(epochs, [h["learning_rate"] for h in history], where="mid", label=f"Seed {run['seed']}")
        lines = [f"# Seed {run['seed']}：逐轮训练记录", "", "[原始日志](train.log) · [完整历史](history.json) · [最终指标](metrics.json)", "",
                 "| Epoch | LR | 训练加权BCE | 训练总损失 | 验证加权BCE | 验证AUC | 梯度裁剪比例 |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for h in history:
            lines.append(f"| {h['epoch']} | {h['learning_rate']:.3g} | {h['train']['task']:.6f} | {h['train']['total']:.6f} | {h['valid']['class_weighted_task_loss']:.6f} | {h['valid']['roc_auc']:.6f} | {h['train']['gradient_clip_fraction']:.2%} |")
        (folder / "TRAINING_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = [("Validation ROC-AUC (fixed 3 views)", "Validation ROC-AUC"), ("Training balanced BCE", "Balanced BCE"),
                ("Validation balanced BCE", "Balanced BCE"), ("Learning rate", "Learning rate")]
    for ax, (title, ylabel) in zip(axes.flat, settings):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    axes[1, 1].set_yscale("log")
    fig.savefig(root / "training_curves.png", dpi=170)
    plt.close(fig)
    aucs = np.array([run["test_auc"] for run in completed])
    selected = max(completed, key=lambda run: run["best_validation_score"])
    summary = {"runs": completed, "completed_runs": len(completed),
               "test_auc_mean": float(aucs.mean()), "test_auc_sample_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
               "selected_by_validation_seed": selected["seed"], "selected_by_validation_test_auc": selected["test_auc"],
               "test_used_for_selection": False, "configuration_origin": "Previously selected by exploratory test comparison",
               "separate_test_evaluations": len(completed)}
    save(root / "summary.json", summary)
    init_hashes = [read(Path(run["directory"]) / "initialization_audit.json")["initial_model_state_sha256"] for run in completed]
    assert len(set(init_hashes)) == len(init_hashes)
    lines = ["# DrugOOD EC50 assay：固定划分、不同种子的50轮复验", "",
             f"已完成 {len(completed)}/{len(state['runs'])} 次训练；每次完整训练 {state['epochs']} 轮，无早停。每次按逐轮验证 ROC-AUC 最高的检查点评估测试集一次。", "",
             "| Seed | 完成轮数 | 最佳轮数 | 选模验证AUC（3视图） | 最终验证AUC（5视图） | 测试AUROC | 测试AP |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for run in completed:
        lines.append(f"| {run['seed']} | {run['epochs_completed']} | {run['best_epoch']} | {run['best_validation_score']:.6f} | {run['valid_auc']:.6f} | {run['test_auc']:.6f} | {run['test_ap']:.6f} |")
    sd = summary["test_auc_sample_std"]
    if sd is not None:
        lines += ["", f"测试 AUROC 均值 ± 样本标准差：**{aucs.mean():.6f} ± {sd:.6f}**。"]
    lines += ["", f"如需在三次中选择一个模型，按验证 AUC 选择 seed={selected['seed']}，其测试 AUROC={selected['test_auc']:.6f}。",
              "", "hidden=192，图4层、SMILES3层，graph主预测，dropout=0.2，AdamW lr=1.5e-4，weight decay=0.01，batch=16，训练集比例加权BCE，辅助权重沿用034_refine。学习率预热和plateau调度保留。",
              "数据预处理和SMILES种子固定为42；种子43/44/45分别控制权重初始化、dropout和训练样本打乱。官方划分、词表、保留样本、类别权重均与034_refine逐一核对一致。各初始参数哈希不同，均从头初始化。",
              f"早停通过将patience设为{state['epochs'] + 1}（大于总轮数{state['epochs']}）使触发条件不可达；日志核对每次均包含全部轮次。",
              "此次各随机种子的检查点和跨种子模型均按验证集选择，不按测试分数挑选。配置来自此前测试集参与比较的探索结果，因此本次是在已知测试集上的稳定性复验。", "",
              "![三次训练曲线](training_curves.png)", "", "日志与模型："]
    for run in completed:
        name = Path(run["directory"]).name
        lines.append(f"- Seed {run['seed']}：[原始日志]({name}/train.log) · [逐轮表格]({name}/TRAINING_LOG.md) · [完整历史]({name}/history.json) · [配置]({name}/config.yaml) · [最优权重]({name}/best.pt) · [指标]({name}/metrics.json)")
    (root / "MULTISEED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def coordinator(args):
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True)
    reference = Path(args.reference_run).resolve()
    base = load_config(args.config)
    assert base["data"]["split_strategy"] == "preassigned"
    assert len(args.seeds) == len(set(args.seeds))
    plan = {"created_at": now(), "config": str(Path(args.config).resolve()), "reference_run": str(reference),
            "seeds": args.seeds, "data_and_smiles_seed": args.data_seed, "epochs": args.epochs,
            "early_stopping_enabled": False, "patience": args.epochs + 1,
            "training_source_hashes": sources(), "input_csv_sha256": digest(resolve_path(base, base["data"]["csv_path"])),
            "test_policy": "Per-seed maximum validation AUC; one post-training test evaluation per seed; report all seeds",
            "configuration_origin": "034_refine, previously chosen by exploratory test comparison"}
    save(root / "experiment_plan.json", plan)
    snapshot = root / "source_snapshot"
    shutil.copytree(ROOT / "src/a2f_ood", snapshot / "a2f_ood", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    shutil.copy2(args.config, snapshot / "base_config.yaml")
    state = {"status": "running", "started_at": now(), "epochs": args.epochs, "runs": []}
    for seed in args.seeds:
        folder = root / f"seed_{seed}"
        folder.mkdir()
        config = copy.deepcopy(base)
        config["experiment"].update(name=f"drugood_ec50_assay_seed_{seed}_50epochs", seed=seed, output_dir=str(folder))
        config["training"].update(epochs=args.epochs, patience=args.epochs + 1, device=args.device)
        path = folder / "requested_config.yaml"
        path.write_text(yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")}, sort_keys=False), encoding="utf-8")
        state["runs"].append({"seed": seed, "status": "pending", "directory": str(folder), "config": str(path)})
    progress(root, state)
    for run in state["runs"]:
        assert sources() == plan["training_source_hashes"]
        folder = Path(run["directory"])
        run.update(status="running", started_at=now())
        started = time.monotonic()
        progress(root, state)
        print(f"{now()} START seed={run['seed']}", flush=True)
        try:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-config", run["config"],
                       "--reference-run", str(reference), "--data-seed", str(args.data_seed)]
            with (folder / "training_console.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                run["launcher_pid"] = process.pid
                progress(root, state)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            metrics = read(folder / "metrics.json")
            assert metrics["epochs_completed"] == args.epochs
            run.update(status="completed", epochs_completed=metrics["epochs_completed"], best_epoch=metrics["best_epoch"],
                       best_validation_score=metrics["best_validation_score"], valid_auc=metrics["valid"]["roc_auc"],
                       test_auc=metrics["test"]["roc_auc"], test_ap=metrics["test"]["average_precision"])
        except Exception as exc:
            run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (folder / "runner_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
        run.update(finished_at=now(), seconds=round(time.monotonic() - started, 2))
        report(root, state)
        progress(root, state)
        print(f"{now()} END seed={run['seed']} status={run['status']} test_auc={run.get('test_auc')}", flush=True)
    state.update(status="completed" if all(r["status"] == "completed" for r in state["runs"]) else "failed", finished_at=now())
    report(root, state)
    progress(root, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/experiments/drugood_ec50_assay_exploratory_best.yaml"))
    parser.add_argument("--reference-run", default=str(ROOT / "outputs/drugood_ec50_assay_hparam_20261120_01/trials/034_refine"))
    parser.add_argument("--output")
    parser.add_argument("--worker-config")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.worker_config:
        worker(args)
    elif args.output:
        coordinator(args)
    else:
        parser.error("--output is required for the coordinator")


if __name__ == "__main__":
    main()
