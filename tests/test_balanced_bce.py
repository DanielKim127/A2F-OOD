"""Class balancing must use train-only counts and reach every supervised head."""
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from a2f_ood.engine import prediction_risk
from a2f_ood.losses import GroupDRO, compute_losses, fit_binary_class_weights, task_sample_losses


def test_training_counts_balance_each_observed_task_with_missing_class_fallback():
    report = fit_binary_class_weights(
        [(0, 1, None), (1, 1, math.nan), (1, None, None), (1, 1, None)],
        ["a", "single_class", "unlabelled"],
    )
    assert report["weights"][0] == pytest.approx([2, 2 / 3])
    assert report["tasks"][0]["negative_count"] == 1
    assert report["tasks"][0]["positive_count"] == 3
    assert report["tasks"][0]["positive_to_negative_weight"] == pytest.approx(1 / 3)
    assert report["weights"][1:] == [[1, 1], [1, 1]]
    assert all(t["neutral_fallback"] for t in report["tasks"][1:])


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_balanced_bce_matches_normalized_pos_weight_and_equalizes_class_gradients(device):
    labels = torch.tensor([[0.], [1.], [1.], [1.]], device=device)
    logits = torch.zeros(4, 2, 1, device=device, requires_grad=True)
    risk, mask = task_sample_losses(logits, labels, torch.ones_like(labels).bool(), task_type="binary",
                                    binary_class_weights=[[2., 2 / 3]])
    expected = 2 * F.binary_cross_entropy_with_logits(
        logits, labels[:, None].expand_as(logits), pos_weight=torch.tensor([1 / 3], device=device))
    assert mask.all()
    torch.testing.assert_close(risk.mean(), expected)
    assert risk.mean().item() == pytest.approx(math.log(2))
    risk.mean().backward()
    assert logits.grad[0].sum().item() == pytest.approx(-logits.grad[1:].sum().item())
    assert logits.grad.sum().item() == pytest.approx(0, abs=1e-7)


def test_missing_labels_have_no_gradient_and_weighted_evaluation_uses_saved_training_weights():
    labels = torch.tensor([[0., 1.], [1., float("nan")], [float("nan"), float("nan")]])
    mask = torch.isfinite(labels)
    logits = torch.tensor([[[1., -1.]], [[-2., 100.]], [[10., 10.]]], requires_grad=True)
    weights = [[2., 2 / 3], [.75, 1.5]]
    risk, valid = task_sample_losses(logits, labels, mask, task_type="multilabel", binary_class_weights=weights)
    loss = risk[valid].mean()
    expected = prediction_risk(labels.numpy(), logits.detach().sigmoid().squeeze(1).numpy(),
                               "multilabel", binary_class_weights=weights)
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert logits.grad[1, 0, 1].item() == 0
    assert torch.equal(logits.grad[2], torch.zeros_like(logits.grad[2]))


def test_main_auxiliary_heads_and_group_dro_use_the_same_class_weights():
    labels = torch.tensor([[0.], [1.]])
    mask = torch.ones_like(labels).bool()
    logits = torch.tensor([[[1.], [2.]], [[-1.], [-2.]]], requires_grad=True)
    graph = logits[:, 0].detach().clone().requires_grad_()
    smiles = logits.detach().clone().requires_grad_()
    config = {"data": {"task_type": "binary"}, "loss": {
        "classification_weighting": "balanced", "binary_class_weights": [[2., 2 / 3]],
        "use_group_dro": True, "group_dro_strength": .25, "lambda_unimodal": .05,
    }}
    batch = {"labels": labels, "label_mask": mask, "group_ids": torch.tensor([0, 1])}
    dro = GroupDRO(2, update_mode="epoch", group_counts=[1, 1])
    dro.group_weights.copy_(torch.tensor([.75, .25]))
    result = compute_losses({"logits": logits, "graph_logits": graph, "smiles_logits": smiles},
                            batch, config, dro, update_dro=False)
    main, _ = task_sample_losses(logits, labels, mask, task_type="binary", binary_class_weights=[[2., 2 / 3]])
    aux, _ = task_sample_losses(graph, labels, mask, task_type="binary", binary_class_weights=[[2., 2 / 3]])
    torch.testing.assert_close(result["task"], main.mean())
    torch.testing.assert_close(result["dro"], .75 * main[0] + .25 * main[1])
    torch.testing.assert_close(result["unimodal"], .5 * (aux.mean() + main.mean()))
    result["total"].backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (logits, graph, smiles))


def test_unfitted_balancing_cannot_silently_use_plain_bce():
    with pytest.raises(ValueError, match="fitted on training"):
        compute_losses({"logits": torch.zeros(1, 1)},
                       {"labels": torch.ones(1, 1), "label_mask": torch.ones(1, 1).bool()},
                       {"data": {"task_type": "binary"}, "loss": {"classification_weighting": "balanced"}})
