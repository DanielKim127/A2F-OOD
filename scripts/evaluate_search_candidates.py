"""Explicitly exploratory test comparison of previously trained candidates.

This is not a fresh held-out evaluation. Test labels enter candidate comparison.
The original validation-only search artifacts are retained without modification.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from a2f_ood.engine import (evaluate_model, get_device, json_safe, load_checkpoint,
                           make_loader, prepare_data, seed_everything, write_predictions)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.search).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    output.mkdir(parents=True)
    original = read(source / "search_status.json")
    candidates = sorted([t for t in original["trials"] if t["status"] == "completed"], key=lambda t: t["valid_auc"], reverse=True)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src/a2f_ood").glob("*.py")}
    assert source_hashes == original["source_hashes"]
    state = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
             "source_search": str(source), "test_used_for_comparison_and_selection": True,
             "held_out_estimate": False, "target_test_auc": .75,
             "target_approximation_threshold": .745, "test_evaluation_count": 0,
             "source_hashes": source_hashes, "candidates": []}
    save(output / "exploratory_status.json", state)
    config = yaml.safe_load((source / "best_config.yaml").read_text(encoding="utf-8"))
    seed_everything(42, config["training"]["cpu_threads"])
    datasets, vocabulary, _ = prepare_data(config, output / "preprocessing")
    datasets["test"].num_views = 5
    device = get_device("cuda")
    for trial in candidates:
        started = time.monotonic()
        path = Path(trial["directory"]) / "best.pt"
        model, checkpoint_vocabulary, checkpoint = load_checkpoint(path, device)
        assert checkpoint_vocabulary.to_dict() == vocabulary.to_dict()
        candidate_config = checkpoint["config"]
        print(f"START {trial['id']} valid={trial['valid_auc']:.6f}", flush=True)
        metrics, rows = evaluate_model(model, make_loader(datasets["test"], candidate_config),
                                       candidate_config, device, vocabulary=vocabulary)
        folder = output / "candidates" / trial["id"]
        folder.mkdir(parents=True)
        write_predictions(folder / "test_predictions.csv", rows)
        save(folder / "metrics.json", metrics)
        result = {"id": trial["id"], "valid_auc": trial["valid_auc"],
                  "test_auc": metrics["roc_auc"], "test_ap": metrics["average_precision"],
                  "checkpoint": str(path), "changes": trial["changes"],
                  "best_epoch": checkpoint["epoch"], "seconds": round(time.monotonic() - started, 2)}
        state["candidates"].append(result)
        state["test_evaluation_count"] += 1
        save(output / "exploratory_status.json", state)
        with (output / "exploratory_leaderboard.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "valid_auc", "test_auc", "test_ap", "best_epoch", "seconds"])
            writer.writeheader()
            for record in sorted(state["candidates"], key=lambda t: t["test_auc"], reverse=True):
                writer.writerow({k: record[k] for k in writer.fieldnames})
        print(f"END {trial['id']} test={metrics['roc_auc']:.6f} seconds={result['seconds']}", flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    winner = max(state["candidates"], key=lambda t: t["test_auc"])
    state.update(status="completed", finished_at=datetime.now().astimezone().isoformat(),
                 test_selected_winner=winner, target_reached=winner["test_auc"] >= .745)
    save(output / "exploratory_status.json", state)
    shutil.copy2(winner["checkpoint"], output / "test_selected_best.pt")
    _, _, selected = load_checkpoint(winner["checkpoint"], "cpu")
    (output / "test_selected_config.yaml").write_text(yaml.safe_dump(selected["config"], sort_keys=False), encoding="utf-8")
    print(json.dumps(winner, indent=2), flush=True)


if __name__ == "__main__":
    main()
