"""Evaluate probabilities/predictions already exported to CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from a2f_ood.engine import json_safe, prediction_risk, write_json
from a2f_ood.metrics import compute_group_metrics, compute_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate A2F-OOD prediction CSVs")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task-type", required=True, choices=("binary", "multiclass", "multilabel", "regression"))
    parser.add_argument("--label-columns", nargs="+", required=True)
    parser.add_argument("--prediction-columns", nargs="+", required=True,
                        help="Probabilities, or regression values; multiclass needs one column per class")
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task_type != "multiclass" and len(args.label_columns) != len(args.prediction_columns):
        raise ValueError("Each label column must have a matching prediction column")
    if args.task_type == "multiclass" and (len(args.label_columns) != 1 or len(args.prediction_columns) < 2):
        raise ValueError("Multiclass requires one class-index label column and >= 2 probability columns")
    with Path(args.predictions).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = set(args.label_columns + args.prediction_columns + ([args.group_column] if args.group_column else []))
        if needed - set(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(needed - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Prediction CSV is empty")
    labels = np.array([[float(r[c]) if r[c].strip() else np.nan for c in args.label_columns] for r in rows])
    predictions = np.array([[float(r[c]) for c in args.prediction_columns] for r in rows])
    report = compute_metrics(labels, predictions, task_type=args.task_type)
    report["task_loss"] = prediction_risk(labels, predictions, args.task_type)
    if args.group_column:
        report["group_metrics"] = compute_group_metrics(labels, predictions, [r[args.group_column] for r in rows], task_type=args.task_type)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
