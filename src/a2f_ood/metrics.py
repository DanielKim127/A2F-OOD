"""Probability-based classification metrics and direction-aware group reports."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


_MINIMIZE = {"rmse", "mae", "mse", "log_loss", "brier", "ece"}


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, *, num_bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    if p.ndim == 2:
        confidence = p.max(-1)
        outcome = (p.argmax(-1) == y).astype(float)
    else:
        confidence, outcome = p, y
    bins = np.minimum((confidence * num_bins).astype(int), num_bins - 1)
    return float(sum(
        np.mean(bins == index)
        * abs(float(confidence[bins == index].mean()) - float(outcome[bins == index].mean()))
        for index in range(num_bins) if (bins == index).any()
    ))


def _check_probabilities(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite predictions for observed labels")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classification metrics require probabilities in [0,1], not logits")


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    names = ["roc_auc", "average_precision", "accuracy", "log_loss", "brier", "ece"]
    if len(y) == 0:
        return {**{key: float("nan") for key in names}, "n_labeled": 0.0}
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels must be 0/1 or NaN")
    _check_probabilities(p)
    both_classes = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": float(np.square(p - y).mean()),
        "ece": _expected_calibration_error(y, p), "n_labeled": float(len(y)),
    }


def _multiclass_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y)
    y, p = y[mask], p[mask]
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("Multiclass predictions must be an [N,num_classes] probability matrix")
    names = ["roc_auc", "average_precision", "accuracy", "log_loss", "brier", "ece"]
    if len(y) == 0:
        return {**{name: float("nan") for name in names}, "n_labeled": 0.0}
    _check_probabilities(p)
    if not np.allclose(p.sum(-1), 1, atol=1e-4):
        raise ValueError("Multiclass probability rows must sum to one")
    # Float32 softmax can sum to 1 +/- 1e-7; normalize in float64 for sklearn.
    p = p / p.sum(-1, keepdims=True)
    if not ((y == np.floor(y)) & (y >= 0) & (y < p.shape[1])).all():
        raise ValueError("Multiclass labels must be valid integer class indices")
    y = y.astype(int)
    targets = np.eye(p.shape[1])[y]
    aucs, aps = [], []
    for class_id in range(p.shape[1]):
        target = targets[:, class_id]
        if len(np.unique(target)) == 2:
            aucs.append(float(roc_auc_score(target, p[:, class_id])))
            aps.append(float(average_precision_score(target, p[:, class_id])))
    return {
        "roc_auc": _finite_mean(aucs), "average_precision": _finite_mean(aps),
        "accuracy": float(accuracy_score(y, p.argmax(-1))),
        "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)).mean()),
        "brier": float(np.square(p - targets).sum(-1).mean()),
        "ece": _expected_calibration_error(y, p), "n_labeled": float(len(y)),
    }


def compute_metrics(
    labels: Any,
    predictions: Any,
    *,
    task_type: str,
) -> dict[str, float]:
    """Compute macros and ``task_i/metric`` keys for multitask data.

    Missing labels are NaN. Macros average finite task values; undefined AUCs
    and wholly unlabelled tasks are excluded. Classification inputs are
    probabilities, never logits. ``n_labeled`` counts observed label entries.
    """
    y, p = _array(labels), _array(predictions)
    if y.ndim == 1:
        y = y[:, None]
    if task_type == "multiclass":
        if y.ndim != 2 or y.shape[1] != 1 or p.shape[0] != y.shape[0]:
            raise ValueError("Multiclass labels must have one class-index column")
        return _multiclass_metrics(y[:, 0], p)
    if p.ndim == 1:
        p = p[:, None]
    if y.ndim != 2 or y.shape != p.shape:
        raise ValueError("labels and predictions must have matching [N,tasks] shapes")
    per_task = []
    for task in range(y.shape[1]):
        mask = np.isfinite(y[:, task])
        target, prediction = y[mask, task], p[mask, task]
        if task_type in {"binary", "multilabel"}:
            scores = _binary_metrics(target, prediction)
        elif task_type == "regression":
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite predictions for observed labels")
            if len(target):
                mse = float(mean_squared_error(target, prediction))
                scores = {"mse": mse, "rmse": float(np.sqrt(mse)),
                          "mae": float(mean_absolute_error(target, prediction)), "n_labeled": float(len(target))}
            else:
                scores = {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n_labeled": 0.0}
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        per_task.append(scores)
    if not per_task:
        raise ValueError("At least one label column is required")
    aggregate = {key: _finite_mean([scores[key] for scores in per_task])
                 for key in per_task[0] if key != "n_labeled"}
    aggregate["n_labeled"] = sum(scores["n_labeled"] for scores in per_task)
    if len(per_task) > 1:
        for task, scores in enumerate(per_task):
            aggregate.update({f"task_{task}/{key}": value for key, value in scores.items()})
    return aggregate


def compute_group_metrics(
    labels: Any,
    predictions: Any,
    groups: Any,
    *,
    task_type: str,
) -> dict[str, Any]:
    """Return per-group, equal-weight mean-group, and worst-group metrics.

    Worst is maximum for errors/calibration scores and minimum for predictive
    scores. Undefined AUCs remain NaN per group and are excluded from summaries.
    Counts are not summarized as model-quality scores.
    """
    y, p = _array(labels), _array(predictions)
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or len(group_array) != len(y) or len(p) != len(y):
        raise ValueError("groups must have one entry per prediction row")
    group_strings = np.array(["<missing>" if group is None else str(group) for group in group_array])
    reports = {group: compute_metrics(y[group_strings == group], p[group_strings == group], task_type=task_type)
               for group in sorted(set(group_strings.tolist()))}
    metric_names = sorted({key for report in reports.values() for key in report if not key.endswith("n_labeled")})
    mean, worst = {}, {}
    for metric in metric_names:
        values = [float(report[metric]) for report in reports.values() if np.isfinite(report.get(metric, np.nan))]
        mean[metric] = _finite_mean(values)
        base_metric = metric.rsplit("/", 1)[-1]
        worst[metric] = float((max if base_metric in _MINIMIZE else min)(values)) if values else float("nan")
    return {"groups": reports, "mean": mean, "worst": worst}
