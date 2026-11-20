import numpy as np
import pytest

from a2f_ood.metrics import compute_group_metrics, compute_metrics


def test_binary_perfect_ranking_and_missing_labels():
    result = compute_metrics([0, 1, np.nan, 1], [0.1, 0.8, np.nan, 0.9], task_type="binary")
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["n_labeled"] == 3
    assert result["accuracy"] == 1


def test_single_class_auc_is_undefined_and_invalid_predictions_raise():
    result = compute_metrics([1, 1], [0.1, 0.5], task_type="binary")
    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["average_precision"])
    with pytest.raises(ValueError, match="probabilities"):
        compute_metrics([0, 1], [-2, 3], task_type="binary")


def test_missing_tasks_do_not_dilute_macro_scores():
    result = compute_metrics([[0, np.nan], [1, np.nan]], [[0.1, 0.8], [0.9, 0.3]], task_type="multilabel")
    assert result["roc_auc"] == 1
    assert np.isnan(result["task_1/roc_auc"])
    assert result["n_labeled"] == 2


def test_group_reports_choose_worst_metric_in_correct_direction():
    classification = compute_group_metrics([0, 1, 0, 1], [0.1, 0.9, 0.9, 0.1], ["a", "a", "b", "b"], task_type="binary")
    assert classification["worst"]["roc_auc"] == 0
    assert classification["mean"]["roc_auc"] == 0.5
    regression = compute_group_metrics([0, 0, 0], [1, 1, 3], ["a", "a", "b"], task_type="regression")
    assert regression["worst"]["rmse"] == 3
    assert regression["mean"]["rmse"] == 2


def test_multiclass_probabilities_and_missing_label():
    result = compute_metrics([0, 1, 2, np.nan], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]], task_type="multiclass")
    assert result["accuracy"] == 1
    assert result["roc_auc"] == 1
    assert result["n_labeled"] == 3


def test_multitask_regression_uses_per_task_finite_masks():
    result = compute_metrics([[1, np.nan], [3, 1]], [[1, 5], [1, 4]], task_type="regression")
    assert result["task_0/rmse"] == pytest.approx(np.sqrt(2))
    assert result["task_1/rmse"] == 3
    assert result["rmse"] == pytest.approx((np.sqrt(2) + 3) / 2)
