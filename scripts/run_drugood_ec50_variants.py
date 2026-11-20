"""Run fixed 50-epoch EC50 scaffold/size experiments and audit saved results."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
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
from a2f_ood.losses import fit_binary_class_weights
from a2f_ood.train import main as train_main


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    plan = read(folder.parent / "experiment_plan.json")
    variant = folder.name
    assert sources() == plan["training_source_hashes"]
    csv_path = resolve_path(config, config["data"]["csv_path"])
    assert digest(csv_path) == plan["input_csv_sha256"][variant]
    seed_everything(42, config["training"]["cpu_threads"])
    prepared = prepare_data(config, folder)
    datasets, vocabulary, _ = prepared
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        original = list(csv.DictReader(stream))
    mapping = {"train": "train", "valid": "ood_val", "test": "ood_test"}
    for split, dataset in datasets.items():
        for molecule in dataset.molecules:
            row = original[molecule.row.row_id]
            assert row["split"] == split and row["official_split"] == mapping[split]
            assert float(row["label"]) == molecule.row.labels[0]
            assert row["group"] == molecule.row.group
    expected_weights = fit_binary_class_weights((m.row.labels for m in datasets["train"].molecules), ["label"])
    save(folder / "data_audit.json", {
        "passed": True, "official_preassigned_membership_and_labels_verified": True,
        "no_resplitting": True, "data_and_training_seed": 42,
        "input_csv_sha256": digest(csv_path), "retained_by_split": {s: len(d) for s, d in datasets.items()},
        "positive_counts": {s: sum(m.row.labels[0] == 1 for m in d.molecules) for s, d in datasets.items()},
        "class_weights_fitted_on": "this variant's retained training labels only"})
    seed_everything(42, config["training"]["cpu_threads"])
    initial = build_model(config, vocabulary)
    model_hash = hashlib.sha256()
    for name, tensor in initial.state_dict().items():
        model_hash.update(name.encode())
        model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    save(folder / "initialization_audit.json", {"seed": 42, "loaded_pretrained_weights": False,
         "initial_model_state_sha256": model_hash.hexdigest(),
         "trainable_parameters": sum(p.numel() for p in initial.parameters() if p.requires_grad)})
    del initial
    assert config["training"]["patience"] > config["training"]["epochs"] == 50
    assert train_main(["--config", args.worker_config, "--selection-only"], prepared_data=prepared) == 0
    history = read(folder / "history.json")
    assert [h["epoch"] for h in history] == list(range(1, 51))
    assert read(folder / "class_weighting.json") == expected_weights
    selection = read(folder / "selection_metrics.json")
    assert not selection["test_evaluated"] and not (folder / "test_predictions.csv").exists()
    best = max(history, key=lambda h: h["valid"]["roc_auc"])
    assert best["epoch"] == selection["best_epoch"]
    save(folder / "frozen_selection.json", {"frozen_at": now(), "criterion": "maximum validation ROC-AUC",
         "best_epoch": best["epoch"], "best_validation_score": best["valid"]["roc_auc"],
         "test_evaluated_before_selection": False, "checkpoint_sha256": digest(folder / "best.pt")})
    device = get_device(config["training"]["device"])
    model, checkpoint_vocabulary, checkpoint = load_checkpoint(folder / "best.pt", device)
    assert checkpoint["epoch"] == best["epoch"]
    assert checkpoint_vocabulary.to_dict() == vocabulary.to_dict()
    config = checkpoint["config"]
    datasets["test"].num_views = config["evaluation"]["random_smiles_views"]
    test_metrics, rows = evaluate_model(model, make_loader(datasets["test"], config), config, device, vocabulary=vocabulary)
    write_predictions(folder / "test_predictions.csv", rows)
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    labels, predictions = [np.array([float(r[key]) for r in rows]) for key in ("label", "prediction")]
    assert abs(roc_auc_score(labels, predictions) - test_metrics["roc_auc"]) < 1e-12
    assert abs(average_precision_score(labels, predictions) - test_metrics["average_precision"]) < 1e-12
    assert [r["row_id"] for r in rows] == [r["row_id"] for r in read(folder / "split_manifest.json")["test"]]
    test_metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions >= .5))
    assert sources() == plan["training_source_hashes"]
    save(folder / "metrics.json", {**selection, "test": test_metrics, "test_evaluated": True,
         "test_evaluation_count": 1, "epochs_completed": 50, "early_stopping_enabled": False,
         "seed": 42, "variant": variant, "finished_at": now()})
    print(f"COMPLETE {variant}: best_epoch={best['epoch']} valid_auc={selection['best_validation_score']:.6f} test_auc={test_metrics['roc_auc']:.6f}", flush=True)


def report(root, state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    completed = [r for r in state["runs"] if r["status"] == "completed"]
    if not completed:
        return
    plan = read(root / "experiment_plan.json")
    assert sources() == plan["training_source_hashes"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    roc_fig, roc_ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    results = []
    for run in completed:
        folder = Path(run["directory"])
        history, metrics = read(folder / "history.json"), read(folder / "metrics.json")
        best = max(history, key=lambda h: h["valid"]["roc_auc"])
        frozen = read(folder / "frozen_selection.json")
        assert [h["epoch"] for h in history] == list(range(1, 51))
        assert best["epoch"] == metrics["best_epoch"] == frozen["best_epoch"]
        assert metrics["test_evaluation_count"] == 1 and not metrics["early_stopping_enabled"]
        assert digest(folder / "best.pt") == frozen["checkpoint_sha256"]
        assert read(folder / "runtime.json")["source_files_sha256"] == plan["training_source_hashes"]
        for kind, epoch in (("best", best["epoch"]), ("last", 50)):
            checkpoint = torch.load(folder / f"{kind}.pt", map_location="cpu")
            assert checkpoint["epoch"] == epoch
            assert checkpoint["feature_schema"]["atom_dim"] == 157 and checkpoint["feature_schema"]["bond_dim"] == 14
            del checkpoint
        with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        labels, probabilities = [np.array([float(r[key]) for r in predictions]) for key in ("label", "prediction")]
        assert abs(roc_auc_score(labels, probabilities) - metrics["test"]["roc_auc"]) < 1e-12
        assert abs(average_precision_score(labels, probabilities) - metrics["test"]["average_precision"]) < 1e-12
        for h in history:
            assert all(np.isfinite(v) for v in h["train"].values())
            if h["epoch"] <= 5:
                assert np.isclose(h["learning_rate"], 7.5e-5 * h["epoch"] / 5, rtol=1e-10, atol=0)
        audit = read(folder / "data_audit.json")
        summary = {"variant": run["variant"], "seed": 42, "epochs": 50, "best_epoch": best["epoch"],
                   "valid_auc": metrics["best_validation_score"], "test_auc": metrics["test"]["roc_auc"],
                   "test_ap": metrics["test"]["average_precision"], "test_ece": metrics["test"]["ece"],
                   "last_valid_auc": history[-1]["valid"]["roc_auc"],
                   "last_valid_weighted_bce": history[-1]["valid"]["class_weighted_task_loss"],
                   "late_clip_fraction": float(np.mean([h["train"]["gradient_clip_fraction"] for h in history[20:]])),
                   "late_preclip_norm": float(np.mean([h["train"]["gradient_norm_mean"] for h in history[20:]])),
                   "train_count": audit["retained_by_split"]["train"], "valid_count": audit["retained_by_split"]["valid"],
                   "test_count": audit["retained_by_split"]["test"], "seconds": run["seconds"]}
        results.append(summary)
        epochs = [h["epoch"] for h in history]
        series = [("Validation ROC-AUC", "ROC-AUC", [h["valid"]["roc_auc"] for h in history]),
                  ("Online training balanced BCE", "Balanced BCE", [h["train"]["task"] for h in history]),
                  ("Validation balanced BCE", "Balanced BCE", [h["valid"]["class_weighted_task_loss"] for h in history]),
                  ("Learning rate", "Learning rate", [h["learning_rate"] for h in history]),
                  ("Steps triggering gradient clipping", "Clipped steps (%)", [100*h["train"]["gradient_clip_fraction"] for h in history]),
                  ("Mean gradient norm BEFORE clipping", "Pre-clip gradient norm", [h["train"]["gradient_norm_mean"] for h in history])]
        single, single_axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
        for ax, solo, (title, ylabel, values) in zip(axes.flat, single_axes.flat, series):
            line, = ax.plot(epochs, values, label=run["variant"])
            solo.plot(epochs, values, color=line.get_color(), label=run["variant"])
            solo.axvline(best["epoch"], ls="--", color="#b08d3c", label=f"Selected epoch {best['epoch']}")
            for panel in (ax, solo):
                panel.set(title=title, xlabel="Epoch", ylabel=ylabel)
                panel.grid(alpha=.2)
                panel.legend(fontsize=8)
        for panel in (axes[1, 1], single_axes[1, 1]):
            panel.set_yscale("log")
        single_axes[2, 0].set_ylim(0, 102)
        single.suptitle(f"DrugOOD EC50 {run['variant']}: seed42, LR 7.5e-5, warmup5")
        single.savefig(folder / "training_curves.png", dpi=155)
        plt.close(single)
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_ax.plot(fpr, tpr, label=f"{run['variant']}, epoch {best['epoch']}: AUC {summary['test_auc']:.4f}")
        lines = [f"# DrugOOD EC50 {run['variant']}：逐轮训练日志", "", "| Epoch | LR | 训练加权BCE | 验证加权BCE | 验证AUROC | 裁剪比例 |", "|---:|---:|---:|---:|---:|---:|"]
        for h in history:
            lines.append(f"| {h['epoch']} | {h['learning_rate']:.4g} | {h['train']['task']:.6f} | {h['valid']['class_weighted_task_loss']:.6f} | {h['valid']['roc_auc']:.6f} | {h['train']['gradient_clip_fraction']:.2%} |")
        (folder / "TRAINING_LOG.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    axes[2, 0].set_ylim(0, 102)
    fig.suptitle("DrugOOD EC50: seed42, LR 7.5e-5, warmup5, 50 epochs")
    fig.savefig(root / "training_curves.png", dpi=155)
    plt.close(fig)
    roc_ax.plot([0, 1], [0, 1], ls="--", color="#999")
    roc_ax.set(title="Test ROC: validation-selected checkpoints", xlabel="False positive rate", ylabel="True positive rate")
    roc_ax.legend()
    roc_ax.grid(alpha=.2)
    roc_fig.savefig(root / "test_roc_curves.png", dpi=165)
    plt.close(roc_fig)
    save(root / "summary.json", {"runs": results, "test_used_for_selection": False, "status": state["status"]})
    save(root / "result_audit.json", {"passed": True, "completed_runs": len(results), "full_epochs_each": 50,
         "training_source_hashes_match": True, "official_split_membership_verified": True,
         "class_weights_fitted_per_training_split": True, "validation_selection_verified": True,
         "test_metrics_recomputed_from_saved_predictions": True, "test_evaluations_per_run": 1})
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ["# DrugOOD EC50 scaffold / size：固定配置训练", "",
             f"当前状态：{state['status']}；完成 {len(results)}/{len(state['runs'])} 个版本。每个版本从头初始化，seed=42，训练满50轮，无早停；按验证AUROC最优检查点评估测试集一次。", "",
             "| 版本 | 训练/验证/测试样本数 | 最优轮 | 验证AUROC | 测试AUROC | 测试AP | 末轮验证AUROC |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['variant']} | {r['train_count']}/{r['valid_count']}/{r['test_count']} | {r['best_epoch']} | {r['valid_auc']:.6f} | {r['test_auc']:.6f} | {r['test_ap']:.6f} | {r['last_valid_auc']:.6f} |")
    lines += ["", "沿用assay候选配置：hidden=192，图4层、SMILES3层、heads=4、dropout=0.2，graph主分类；原子157维、键14维。AdamW lr=7.5e-5、预热5轮、weight decay=0.01、batch=16、clip=1.0、Plateau patience=3/factor=0.5/min_lr=1e-6；辅助目标与启用时序沿用。",
              "使用各版本官方 train / ood_val / ood_test 划分，不重新随机划分；词表仅由各自训练集建立，balanced BCE权重仅由各自保留训练标签拟合。保留既有原子数与SMILES长度上限，跳过情况保存在各目录的preprocessing_summary.json和skipped_rows.json。",
              "这是每个版本单个随机种子的结果，不能据此估计种子稳定性。配置沿用此前assay探索结果；本轮测试不参与检查点选择。", "", "![训练曲线](training_curves.png)", "", "![测试ROC](test_roc_curves.png)", ""]
    for r in results:
        v = r["variant"]
        lines += [f"- {v}：[逐轮日志]({v}/TRAINING_LOG.md) · [原始日志]({v}/train.log) · [配置]({v}/config.yaml) · [最优模型]({v}/best.pt) · [完整指标]({v}/metrics.json) · [数据核对]({v}/data_audit.json)"]
    lines += ["", "[结果CSV](results.csv) · [核验记录](result_audit.json)"]
    (root / "TRAINING_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def coordinator(args):
    root = Path(args.output).resolve()
    assert not root.exists(), "Use a fresh output directory"
    root.mkdir(parents=True)
    base = load_config(args.base_config)
    snapshot = root / "source_snapshot"
    shutil.copytree(ROOT / "src/a2f_ood", snapshot / "a2f_ood", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    shutil.copy2(args.base_config, snapshot / "base_config.yaml")
    plan = {"created_at": now(), "base_config": str(Path(args.base_config).resolve()), "seed": 42,
            "epochs": 50, "early_stopping_enabled": False, "training_source_hashes": sources(),
            "input_csv_sha256": {}, "test_policy": "one test evaluation after validation selection per variant"}
    state = {"status": "running", "started_at": now(), "runs": []}
    for variant in args.variants:
        folder = root / variant
        folder.mkdir()
        config = {k: copy.deepcopy(v) for k, v in base.items() if not k.startswith("_")}
        config["experiment"].update(name=f"drugood_ec50_{variant}_lr_warmup_seed42", seed=42, output_dir=str(folder))
        config["data"]["csv_path"] = f"data/processed/drugood/lbap_core_ec50_{variant}.csv"
        config["loss"]["binary_class_weights"] = None
        assert config["training"]["epochs"] == 50 and config["training"]["patience"] == 51
        assert config["training"]["learning_rate"] == 7.5e-5 and config["training"]["learning_rate_warmup_epochs"] == 5
        assert config["data"]["split_strategy"] == "preassigned"
        csv_path = ROOT / config["data"]["csv_path"]
        metadata = read(csv_path.with_suffix(".metadata.json"))
        plan["input_csv_sha256"][variant] = digest(csv_path)
        assert digest(csv_path) == metadata["csv_sha256"]
        path = folder / "requested_config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        load_config(path)
        state["runs"].append({"variant": variant, "status": "pending", "directory": str(folder), "config": str(path)})
    save(root / "experiment_plan.json", plan)
    save(root / "experiment_status.json", state)
    for run in state["runs"]:
        folder = Path(run["directory"])
        run.update(status="running", started_at=now())
        started = time.monotonic()
        print(f"{now()} START {run['variant']}", flush=True)
        try:
            with (folder / "training_console.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker-config", run["config"]], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                run["worker_pid"] = process.pid
                save(root / "experiment_status.json", state)
                code = process.wait()
                if code:
                    raise RuntimeError(f"worker exit code {code}")
            metrics = read(folder / "metrics.json")
            run.update(status="completed", best_epoch=metrics["best_epoch"], valid_auc=metrics["best_validation_score"], test_auc=metrics["test"]["roc_auc"])
        except Exception as exc:
            run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (folder / "runner_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
        run.update(finished_at=now(), seconds=round(time.monotonic()-started, 2))
        save(root / "experiment_status.json", state)
        report(root, state)
        print(f"{now()} END {run['variant']} status={run['status']} test_auc={run.get('test_auc')}", flush=True)
    state.update(status="completed" if all(r["status"] == "completed" for r in state["runs"]) else "failed", finished_at=now())
    save(root / "experiment_status.json", state)
    report(root, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config")
    parser.add_argument("--output")
    parser.add_argument("--variants", nargs="+", choices=["scaffold", "size"], default=["scaffold", "size"])
    parser.add_argument("--worker-config")
    args = parser.parse_args()
    if args.worker_config:
        worker(args)
    else:
        if not args.base_config or not args.output:
            parser.error("--base-config and --output are required")
        coordinator(args)


if __name__ == "__main__":
    main()
