"""Run official A2F-OOD presets once each and maintain a reviewable report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
PRESETS = [
    "drugood_ec50_assay", "admeood_ec50_assay",
    "drugood_ec50_scaffold", "drugood_ec50_size",
    "admeood_ec50_scaffold", "admeood_ki_assay", "admeood_ki_scaffold",
    "drugood_ic50_assay", "drugood_ic50_scaffold", "drugood_ic50_size",
]


def training_fingerprint() -> dict[str, str]:
    files = list((ROOT / "src" / "a2f_ood").glob("*.py")) + list((ROOT / "configs" / "datasets").glob("*.yaml"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def refresh_progress(suite: Path, manifest: dict) -> None:
    for run in manifest["runs"]:
        history = read_json(suite / run["name"] / "history.json", [])
        if history:
            last = history[-1]
            best = max(history, key=lambda row: row["valid"]["roc_auc"])
            run.update(epochs_completed=last["epoch"],
                       latest_train_loss=last["train"]["task"],
                       latest_valid_auc=last["valid"]["roc_auc"],
                       best_epoch=best["epoch"], best_valid_auc=best["valid"]["roc_auc"])
        metrics = read_json(suite / run["name"] / "metrics.json")
        if metrics:
            run["test_auc"] = metrics["test"]["roc_auc"]
    manifest["updated_at"] = now()
    write_json(suite / "suite_status.json", manifest)


def make_report(suite: Path, *, figures: bool) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "summarize_benchmarks.py"), "--suite", str(suite)]
    if figures:
        command.append("--figures")
    with (suite / "report_generation.log").open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"Report generation returned {result.returncode}; see report_generation.log", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=PRESETS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    suite = Path(args.output).resolve()
    suite.mkdir(parents=True, exist_ok=True)
    status_file = suite / "suite_status.json"
    if status_file.exists() and not args.resume:
        raise SystemExit("Suite exists: use --resume or select a fresh output directory")
    manifest = read_json(status_file) if args.resume else None
    fingerprint = training_fingerprint()
    if manifest:
        recorded = manifest.get("training_fingerprint")
        if recorded is None:
            recorded = {key: value for key, value in read_json(suite / "experiment_manifest.json", {}).get("source_sha256", {}).items()
                        if key in fingerprint}
        if recorded != fingerprint:
            raise SystemExit("Training source or presets changed. Use a fresh output directory to avoid mixing experiments.")
    if not manifest:
        manifest = {"started_at": now(), "status": "running", "seed": 42,
                    "max_epochs": 100, "patience": 15, "selection": "validation ROC-AUC",
                    "official_splits": True, "full_datasets": True, "training_fingerprint": fingerprint,
                    "runs": [{"name": name, "status": "queued"} for name in args.presets]}
    manifest["runner_pid"] = os.getpid()
    refresh_progress(suite, manifest)
    make_report(suite, figures=False)
    for run in manifest["runs"]:
        if training_fingerprint() != fingerprint:
            manifest["status"] = "paused_source_changed"
            refresh_progress(suite, manifest)
            make_report(suite, figures=True)
            raise SystemExit("Training source or presets changed during the suite; start a fresh suite.")
        run_dir = suite / run["name"]
        if run.get("status") == "completed" and (run_dir / "metrics.json").exists():
            continue
        # A resumed unfinished run is restarted in a new attempt directory.
        if run_dir.exists() and any(run_dir.iterdir()):
            backup = suite / (run["name"] + "_interrupted_" + datetime.now().strftime("%H%M%S"))
            run_dir.rename(backup)
        run_dir.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "a2f_ood.train", "--config",
                   str(ROOT / "configs" / "datasets" / (run["name"] + ".yaml")),
                   "--set", f"experiment.output_dir={run_dir.as_posix()}",
                   "--set", "training.device=cuda", "--set", "experiment.seed=42"]
        run.update(status="running", started_at=now(), command=command)
        started = time.monotonic()
        print(f"{now()} START {run['name']}", flush=True)
        environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        with (run_dir / "console.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
            run["pid"] = process.pid
            refresh_progress(suite, manifest)
            make_report(suite, figures=False)
            while process.poll() is None:
                time.sleep(15)
                refresh_progress(suite, manifest)
                make_report(suite, figures=False)
        run.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - started, 2),
                   returncode=process.returncode,
                   status="completed" if process.returncode == 0 and (run_dir / "metrics.json").exists() else "failed")
        refresh_progress(suite, manifest)
        make_report(suite, figures=True)
        print(f"{now()} {run['status'].upper()} {run['name']} {run['elapsed_seconds']:.0f}s", flush=True)
    manifest["status"] = "completed" if all(r["status"] == "completed" for r in manifest["runs"]) else "completed_with_failures"
    manifest["finished_at"] = now()
    refresh_progress(suite, manifest)
    make_report(suite, figures=True)
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
