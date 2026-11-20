"""Validation-only, staged hyperparameter search with one frozen test evaluation.

All trials retain train-fitted balanced BCE and the official molecular splits.
Preprocessing is shared in memory; each trial receives fresh dataset objects.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import gc
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import random
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
from a2f_ood.config import load_config, validate_config
from a2f_ood.data import MoleculeDataset
from a2f_ood.engine import (evaluate_model, get_device, load_checkpoint, make_loader,
                           prepare_data, seed_everything, write_predictions, json_safe)
from a2f_ood.metrics import compute_metrics
from a2f_ood.train import main as train_main


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def set_values(config, changes):
    for dotted, value in changes.items():
        section, key = dotted.split(".")
        config[section][key] = value


def parameter_signature(changes):
    return json.dumps(changes, sort_keys=True)


def candidate_space(count, seed):
    """Cover scale, architecture, optimization and objective choices explicitly."""
    candidates = []
    for mode, hidden, depth, lr, dropout, aux, dro in [
        ("adaptive", 64, 2, 3e-4, .2, .1, 0),
        ("concat", 96, 2, 3e-4, .2, 0, 0),
        ("graph", 96, 4, 1e-3, .2, 0, 0),
        ("smiles", 96, 2, 1e-4, .2, 0, 0),
        ("adaptive", 128, 2, 1e-4, .3, 0, 0),
        ("graph", 64, 3, 3e-4, .1, 0, 0),
        ("concat", 128, 3, 1e-4, .3, .1, .1),
        ("smiles", 64, 2, 3e-4, .1, .1, 0),
        ("adaptive", 192, 2, 5e-5, .2, .1, 0),
        ("graph", 128, 5, 3e-4, .3, 0, .1),
        ("concat", 64, 2, 1e-3, .1, 0, 0),
        ("adaptive", 256, 2, 1e-4, .2, .25, .1),
    ]:
        candidates.append(dict(mode=mode, hidden=hidden, graph_depth=depth, smiles_depth=min(depth, 3),
                               lr=lr, dropout=dropout, aux=aux, dro=dro, decay=1e-4, batch=32,
                               cross=True, gate=True))
    rng = random.Random(seed)
    for index in range(max(0, count - len(candidates))):
        hidden = rng.choice([64, 96, 128, 192, 256, 384])
        candidates.append(dict(mode=rng.choice(["adaptive", "adaptive", "concat", "graph", "smiles"]),
                               hidden=hidden, graph_depth=rng.choice([2, 3, 4, 5]), smiles_depth=rng.choice([1, 2, 3]),
                               lr=rng.choice([3e-5, 1e-4, 3e-4, 7e-4]), dropout=rng.choice([.0, .1, .2, .35]),
                               aux=rng.choice([0, .05, .2, .5]), dro=rng.choice([0, 0, .1, .25]),
                               decay=rng.choice([1e-5, 1e-4, 1e-3, 1e-2]),
                               batch=8 if hidden == 384 else 16 if hidden >= 192 else rng.choice([16, 32, 64]),
                               cross=rng.choice([True, False]), gate=rng.choice([True, False])))
    results = []
    for item in candidates[:count]:
        changes = {
            "model.hidden_dim": item["hidden"], "model.graph_layers": item["graph_depth"],
            "model.smiles_layers": item["smiles_depth"], "model.attention_heads": 4,
            "model.dropout": item["dropout"], "model.use_cross_modal_decoder": item["aux"] > 0,
            "fusion.mode": item["mode"], "fusion.bidirectional_cross_attention": item["cross"],
            "fusion.reliability_gate": item["gate"],
            "training.learning_rate": item["lr"], "training.weight_decay": item["decay"],
            "training.batch_size": item["batch"], "training.learning_rate_warmup_epochs": 2,
            "loss.use_group_dro": item["dro"] > 0, "loss.group_dro_strength": item["dro"],
        }
        for key, value in {"lambda_atom_alignment": .02, "lambda_serialization_consistency": .1,
                           "lambda_cross_modal": .1, "lambda_generation": .01, "lambda_unimodal": .05}.items():
            changes["loss." + key] = value * item["aux"]
        results.append(changes)
    return results


class Search:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        status_path = self.root / "search_status.json"
        if status_path.exists() and not args.resume:
            raise ValueError("Select a fresh output directory")
        if args.resume:
            self.state = read(status_path)
            if self.state.get("test_evaluation_count", 0) or (self.root / "frozen_selection.json").exists():
                raise ValueError("Cannot resume search after a test selection has been frozen")
            if not (self.root / "finalists.json").exists():
                raise ValueError("Resume currently requires the completed screening/refinement and finalists.json")
            # Restore the original fixed data/training settings, independently of CLI defaults.
            first_config = Path(self.state["trials"][0]["directory"]) / "requested_config.yaml"
            self.base = load_config(first_config)
        else:
            self.base = load_config(args.config)
        if self.base["data"]["task_type"] != "binary":
            raise ValueError("This search reports binary ROC-AUC and requires a binary dataset")
        self.base["training"]["device"] = args.device
        self.base["training"].update(monitor="roc_auc", monitor_mode="max")
        self.base["loss"]["classification_weighting"] = "balanced"
        self.base["experiment"]["seed"] = 42
        self.audit_dir = self.root / "shared_preprocessing"
        self.audit_dir.mkdir(exist_ok=args.resume)
        self.fingerprint = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src/a2f_ood").glob("*.py")}
        if args.resume and self.fingerprint != self.state["source_hashes"]:
            raise RuntimeError("Training source changed since the saved search")
        seed_everything(42, self.base["training"]["cpu_threads"])
        self.prepared = prepare_data(self.base, self.audit_dir)
        if args.resume:
            for trial in self.state["trials"]:
                if trial["status"] == "running":
                    trial.update(status="interrupted", finished_at=now(),
                                 error="Search process restarted with isolated trial processes; checkpoint retained")
            self.state.update(status="running", resumed_at=now(), isolated_trials=args.isolated)
        else:
            self.state = {"status": "running", "started_at": now(), "seed": 42, "search_seed": args.search_seed,
                      "selection": "validation ROC-AUC only", "target_test_auc": .75,
                      "test_used_for_search": False, "test_evaluation_count": 0,
                      "source_hashes": self.fingerprint, "trials": []}
        self.update()

    def update(self):
        self.state["updated_at"] = now()
        save(self.root / "search_status.json", self.state)
        rows = []
        for trial in sorted(self.state["trials"], key=lambda t: t.get("valid_auc", -1), reverse=True):
            changes = trial["changes"]
            rows.append({"trial": trial["id"], "stage": trial["stage"], "status": trial["status"],
                         "mode": changes["fusion.mode"], "hidden": changes["model.hidden_dim"],
                         "graph_layers": changes["model.graph_layers"], "smiles_layers": changes["model.smiles_layers"],
                         "lr": changes["training.learning_rate"], "dropout": changes["model.dropout"],
                         "batch": changes["training.batch_size"], "decay": changes["training.weight_decay"],
                         "valid_auc_5views": trial.get("valid_auc"), "best_epoch": trial.get("best_epoch"),
                         "seconds": trial.get("seconds"), "error": trial.get("error", "")})
        if rows:
            with (self.root / "leaderboard.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        columns = ["trial", "stage", "status", "mode", "hidden", "graph_layers", "smiles_layers", "lr", "dropout", "valid_auc_5views", "best_epoch", "seconds"]
        table = "<tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>"
        for row in rows:
            table += "<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in columns) + "</tr>"
        page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>DrugOOD EC50 assay hyperparameter search</title><style>body{{font:14px/1.6 Arial;margin:32px;color:#183042}}table{{border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #ddd}}img{{max-width:100%}}</style>
<h1>DrugOOD EC50 assay 超参数搜索</h1><p>状态：{self.state['status']}；更新时间：{self.state['updated_at']}</p>
<p>仅按验证 AUC 搜索，保留训练集拟合的类别加权 BCE。目标测试 AUC 约 0.75；测试集评估次数 {self.state['test_evaluation_count']}。</p>
<p><a href="leaderboard.csv">参数结果 CSV</a> · <a href="search_status.json">完整配置和状态</a> · <a href="SEARCH_REPORT.md">最终报告</a></p>
<table>{table}</table>'''
        if (self.root / "search_curves.png").exists():
            page += '<img src="search_curves.png">'
        if (self.root / "final_metrics.json").exists():
            page += '<p><a href="final_metrics.json">最终测试指标</a></p>'
        (self.root / "index.html").write_text(page, encoding="utf-8")

    def datasets(self, config):
        original, vocab, groups = self.prepared
        # Every trial starts at epoch zero with fresh evaluation view counts.
        datasets = {name: MoleculeDataset(dataset.molecules, vocab,
                    num_views=config["data"]["random_smiles_per_molecule"] if name == "train" else
                              config["training"]["validation_smiles_views"] if name == "valid" else 1,
                    randomize=name == "train", seed=config["experiment"]["seed"], group_to_id=groups)
                    for name, dataset in original.items()}
        return datasets, vocab, groups

    def trial(self, changes, stage, epochs, patience, *, parent=None):
        for p in (ROOT / "src/a2f_ood").glob("*.py"):
            if hashlib.sha256(p.read_bytes()).hexdigest() != self.fingerprint[p.name]:
                raise RuntimeError("Training source changed during search")
        identity = f"{len(self.state['trials']) + 1:03d}_{stage}"
        folder = self.root / "trials" / identity
        folder.mkdir(parents=True)
        config = copy.deepcopy(self.base)
        set_values(config, changes)
        config["experiment"].update(name=identity, output_dir=str(folder))
        config["training"].update(epochs=epochs, patience=patience, monitor_min_delta=.001)
        validate_config(config)
        for path in self.audit_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, folder / path.name)
        config_path = folder / "requested_config.yaml"
        config_path.write_text(yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")}, sort_keys=False), encoding="utf-8")
        trial = {"id": identity, "stage": stage, "status": "running", "changes": changes,
                 "parent": parent, "started_at": now(), "directory": str(folder), "max_epochs": epochs}
        self.state["trials"].append(trial)
        self.update()
        print(f"{now()} START {identity} {json.dumps(changes, sort_keys=True)}", flush=True)
        started = time.monotonic()
        try:
            if self.args.isolated:
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
                with (folder / "training_console.log").open("w", encoding="utf-8") as stream:
                    command = [sys.executable, "-u", "-m", "a2f_ood.train", "--config", str(config_path), "--selection-only"]
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    trial.update(process_id=process.pid, execution="isolated subprocess")
                    self.update()
                    returncode = process.wait()
                    if returncode:
                        raise subprocess.CalledProcessError(returncode, command)
            else:
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                train_main(["--config", str(config_path), "--selection-only"], prepared_data=self.datasets(config))
            metrics = read(folder / "selection_metrics.json")
            assert "test" not in metrics and not (folder / "test_predictions.csv").exists()
            history = read(folder / "history.json")
            trial.update(status="completed", valid_auc=metrics["valid"]["roc_auc"],
                         selection_valid_auc=metrics["best_validation_score"], best_epoch=metrics["best_epoch"],
                         epochs=len(history), peak_cuda_mib=None if self.args.isolated else
                         torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0)
            # Last checkpoints are disposable in screening; best checkpoints remain reviewable.
            (folder / "last.pt").unlink(missing_ok=True)
        except Exception as exc:
            trial.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (folder / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
        finally:
            gc.collect()
            if not self.args.isolated and torch.cuda.is_available():
                torch.cuda.empty_cache()
            trial.update(seconds=round(time.monotonic() - started, 2), finished_at=now())
            self.update()
        print(f"{now()} END {identity} {trial['status']} valid={trial.get('valid_auc')} seconds={trial['seconds']}", flush=True)
        return trial

    def ranked(self, stages=None):
        return sorted([t for t in self.state["trials"] if t["status"] == "completed"
                       and (stages is None or t["stage"] in stages)], key=lambda t: t["valid_auc"], reverse=True)

    def run(self):
        if self.args.resume:
            finalists = read(self.root / "finalists.json")
            for finalist in finalists:
                completed = any(t["status"] == "completed" and t["stage"] == "extended"
                                and t["parent"] == finalist["id"] for t in self.state["trials"])
                if not completed:
                    self.trial(finalist["changes"], "extended", self.args.final_epochs, 10, parent=finalist["id"])
            self.final_evaluation(self.ranked()[0])
            return
        initial = candidate_space(self.args.screen_trials, self.args.search_seed)
        save(self.root / "search_plan.json", {"screen": initial, "screen_epochs": self.args.screen_epochs,
             "refinement_rule": "For top four screening trials: 0.5x LR, 2x LR, or stronger dropout/weight decay",
             "refine_epochs": self.args.refine_epochs, "finalists": self.args.finalists,
             "final_epochs": self.args.final_epochs, "test_policy": "Freeze the highest validation AUC candidate before one test evaluation"})
        for changes in initial:
            self.trial(changes, "screen", self.args.screen_epochs, 4)
        anchors = self.ranked({"screen"})[:4]
        used = {parameter_signature(t["changes"]) for t in self.state["trials"]}
        for anchor in anchors:
            for variant in ("lower_lr", "higher_lr", "regularize"):
                changes = copy.deepcopy(anchor["changes"])
                if variant == "lower_lr":
                    changes["training.learning_rate"] *= .5
                elif variant == "higher_lr":
                    changes["training.learning_rate"] *= 2
                else:
                    changes["model.dropout"] = min(.5, changes["model.dropout"] + .1)
                    changes["training.weight_decay"] = max(1e-3, changes["training.weight_decay"] * 10)
                if parameter_signature(changes) not in used:
                    used.add(parameter_signature(changes))
                    self.trial(changes, "refine", self.args.refine_epochs, 5, parent=anchor["id"])
        finalists = self.ranked()[:self.args.finalists]
        save(self.root / "finalists.json", [{"id": t["id"], "valid_auc": t["valid_auc"], "changes": t["changes"]} for t in finalists])
        for finalist in finalists:
            self.trial(finalist["changes"], "extended", self.args.final_epochs, 10, parent=finalist["id"])
        winner = self.ranked()[0]
        self.final_evaluation(winner)

    def final_evaluation(self, winner):
        if (self.root / "frozen_selection.json").exists() or self.state.get("test_evaluation_count", 0):
            raise RuntimeError("Final model already frozen; refusing another test selection/evaluation")
        checkpoint_path = Path(winner["directory"]) / "best.pt"
        frozen = {"frozen_at": now(), "trial": winner["id"], "valid_auc": winner["valid_auc"],
                  "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  "selection": "maximum validation ROC-AUC (5 fixed views)", "test_evaluated_before_selection": False,
                  "changes": winner["changes"]}
        save(self.root / "frozen_selection.json", frozen)
        self.state.update(status="final_test_evaluation", frozen_selection=frozen)
        self.update()
        device = get_device(self.args.device)
        model, vocabulary, checkpoint = load_checkpoint(checkpoint_path, device)
        config = checkpoint["config"]
        datasets, _, _ = self.datasets(config)
        report = {"trial": winner["id"], "best_epoch": checkpoint["epoch"], "seed": 42,
                  "selection_validation_auc": winner["valid_auc"], "target_test_auc": .75}
        for split in ("valid", "test"):
            datasets[split].num_views = 5
            values, rows = evaluate_model(model, make_loader(datasets[split], config), config, device, vocabulary=vocabulary)
            report[split] = values
            write_predictions(self.root / f"final_{split}_predictions.csv", rows)
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef
        with (self.root / "final_test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        y = np.array([float(r["label"]) for r in records])
        p = np.array([float(r["prediction"]) for r in records])
        report["test"].update(balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
                             mcc=float(matthews_corrcoef(y, p >= .5)),
                             confusion_matrix=confusion_matrix(y, p >= .5, labels=[0, 1]).tolist())
        report["target_reached"] = bool(report["test"]["roc_auc"] >= .745)
        save(self.root / "final_metrics.json", report)
        shutil.copy2(checkpoint_path, self.root / "best.pt")
        (self.root / "best_config.yaml").write_text(yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")}, sort_keys=False), encoding="utf-8")
        self.state.update(status="completed", finished_at=now(), test_evaluation_count=1,
                          final_test_auc=report["test"]["roc_auc"], target_reached=report["target_reached"])
        self.make_report(report, winner)
        self.update()

    def make_report(self, report, winner):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
        ranked = self.ranked()
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
        colors = {"adaptive": "#17628a", "concat": "#489d73", "graph": "#b08d3c", "smiles": "#bb6b52"}
        for mode, color in colors.items():
            items = [t for t in ranked if t["changes"]["fusion.mode"] == mode]
            axes[0].scatter([int(t["id"].split("_")[0]) for t in items], [t["valid_auc"] for t in items], label=mode, color=color)
        axes[0].set(title="All trials: validation ROC-AUC only", xlabel="Trial", ylabel="Validation AUC (5 views)")
        axes[0].legend()
        for trial in ranked[:5]:
            history = read(Path(trial["directory"]) / "history.json")
            axes[1].plot([h["epoch"] for h in history], [h["valid"]["roc_auc"] for h in history], label=trial["id"])
        axes[1].set(title="Top candidates: validation histories", xlabel="Epoch", ylabel="Validation AUC (3 views)")
        axes[1].legend(fontsize=8)
        fig.savefig(self.root / "search_curves.png", dpi=160)
        plt.close(fig)
        with (self.root / "final_test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        y = np.array([float(r["label"]) for r in records]); p = np.array([float(r["prediction"]) for r in records])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"ROC-AUC {report['test']['roc_auc']:.4f}")
        axes[0].plot([0, 1], [0, 1], "--", color="#aaa")
        axes[0].set(title="Final OOD test ROC", xlabel="False positive rate", ylabel="True positive rate")
        precision, recall, _ = precision_recall_curve(y, p)
        axes[1].plot(recall, precision, label=f"AP {report['test']['average_precision']:.4f}")
        axes[1].axhline(y.mean(), ls="--", color="#aaa", label=f"Prevalence {y.mean():.4f}")
        axes[1].set(title="Final OOD test precision-recall", xlabel="Recall", ylabel="Precision")
        for ax in axes: ax.legend()
        fig.savefig(self.root / "final_test_curves.png", dpi=160); plt.close(fig)
        lines = ["# DrugOOD EC50 assay 超参数搜索结果", "",
                 f"共运行 {len(self.state['trials'])} 个训练任务，成功 {len(ranked)} 个。所有候选保留按训练集比例加权的 BCE；按验证集 AUC 筛选后，冻结一个模型并评估测试集一次。",
                 f"最优候选 `{winner['id']}`，最优 checkpoint 第 {report['best_epoch']} 轮，验证 AUC={report['valid']['roc_auc']:.6f}，测试 AUC={report['test']['roc_auc']:.6f}。",
                 f"目标测试 AUC 约 0.75：{'达到' if report['target_reached'] else '未达到'}（报告采用 >=0.745 的近似判断）。", "",
                 "```json", json.dumps(winner["changes"], ensure_ascii=False, indent=2), "```", "",
                 "| 指标 | 验证集 | 测试集 |", "|---|---:|---:|"]
        for key in ("roc_auc", "average_precision", "accuracy", "log_loss", "ece"):
            lines.append(f"| {key} | {report['valid'][key]:.6f} | {report['test'][key]:.6f} |")
        lines += ["", f"测试平衡准确率={report['test']['balanced_accuracy']:.6f}，MCC={report['test']['mcc']:.6f}；固定阈值 0.5。",
                  "", "筛选和优化使用训练/验证集；测试集仅在最终配置冻结后评估。既有历史测试结果已知，因此新结果仍属于本数据集的探索结果。单随机种子和大量验证集选择会带来乐观偏差，需要额外数据或多种子验证。",
                  "", "![搜索对比](search_curves.png)", "", "![最终测试曲线](final_test_curves.png)", "",
                  "[完整排行榜](leaderboard.csv) · [冻结选择记录](frozen_selection.json) · [最终指标](final_metrics.json) · [最佳配置](best_config.yaml)", ""]
        (self.root / "SEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/datasets/drugood_ec50_assay.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--screen-trials", type=int, default=24)
    parser.add_argument("--screen-epochs", type=int, default=7)
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=40)
    parser.add_argument("--search-seed", type=int, default=20261120)
    parser.add_argument("--resume", action="store_true", help="Resume saved finalists after screening/refinement")
    parser.add_argument("--isolated", action="store_true", help="Run each training trial in a fresh Python process")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    Search(args).run()


if __name__ == "__main__":
    main()
