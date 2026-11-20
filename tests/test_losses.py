"""Scientific edge cases: missing labels, group state, and view consistency."""

import pytest
import torch
from torch.nn import functional as F

from a2f_ood.losses import (
    GroupDRO,
    atom_alignment_loss,
    compute_losses,
    serialization_consistency_loss,
    task_sample_losses,
)


def test_missing_labels_do_not_dilute_supervised_risk_or_gradients():
    logits = torch.tensor([[[0.0, 100.0]], [[-10.0, 20.0]]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, False], [False, False]])
    losses, observed = task_sample_losses(logits, labels, mask, task_type="multilabel")
    assert observed.tolist() == [True, False]
    assert losses[0].item() == pytest.approx(float(F.softplus(torch.tensor(0.0))))
    assert losses[1].item() == 0
    losses.sum().backward()
    assert logits.grad[0, 0, 1].item() == 0
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))


@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
def test_all_unlabelled_batches_still_allow_backpropagation(task_type):
    outputs = 3 if task_type == "multiclass" else 1
    logits = torch.randn(2, 2, outputs, requires_grad=True)
    losses, observed = task_sample_losses(
        logits, torch.full((2, 1), float("nan")), torch.zeros(2, 1, dtype=torch.bool), task_type=task_type
    )
    assert not observed.any()
    losses.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() == 0


def test_multiclass_uses_class_indices_and_averages_views():
    logits = torch.tensor([[[2.0, 0.0, -1.0], [1.0, 0.0, -1.0]]], requires_grad=True)
    loss, mask = task_sample_losses(
        logits, torch.tensor([[0.0]]), torch.tensor([[True]]), task_type="multiclass"
    )
    expected = F.cross_entropy(logits.flatten(0, 1), torch.tensor([0, 0]))
    assert mask.item()
    assert torch.allclose(loss[0], expected)
    with pytest.raises(ValueError, match="class indices"):
        task_sample_losses(logits, torch.tensor([[0.5]]), torch.tensor([[True]]), task_type="multiclass")


def test_group_dro_increases_hard_group_weight_without_diluting_absent_groups():
    dro = GroupDRO(3, step_size=1.0)
    losses = torch.tensor([1.0, 3.0], requires_grad=True)
    result = dro(losses, torch.tensor([0, 1]))
    assert dro.group_weights[1] > dro.group_weights[0] > dro.group_weights[2]
    assert result.item() > 2
    result.backward()
    assert losses.grad[1] > losses.grad[0]
    single_group = dro(torch.tensor([5.0]), torch.tensor([2]), update=False)
    assert single_group.item() == pytest.approx(5.0)
    weights = dro.group_weights.clone()
    dro.eval()
    dro(torch.tensor([20.0, 1.0]), torch.tensor([0, 1]))
    assert torch.equal(weights, dro.group_weights)
    assert dro(torch.tensor([2.0, 4.0]), torch.tensor([-1, 1])).item() == pytest.approx(3.0)
    clone = GroupDRO(3)
    clone.load_state_dict(dro.state_dict())
    assert torch.equal(clone.group_weights, weights)


def test_atom_alignment_masks_padding_and_rewards_correct_correspondence():
    graph = torch.eye(4).reshape(1, 4, 4).requires_grad_()
    smiles = graph.detach().clone().unsqueeze(1).requires_grad_()
    mask = torch.tensor([[True, True, True, False]])
    good = atom_alignment_loss(graph, smiles, mask)
    incorrect = smiles.detach().clone()
    incorrect[:, :, :3] = incorrect[:, :, [1, 2, 0]]
    assert good < atom_alignment_loss(graph, incorrect, mask)
    changed_padding = smiles.detach().clone()
    changed_padding[:, :, 3] = 1000
    assert torch.allclose(good, atom_alignment_loss(graph, changed_padding, mask))
    good.backward()
    assert torch.isfinite(graph.grad).all()
    assert smiles.grad[:, :, 3].abs().sum().item() == 0


@pytest.mark.parametrize("task_type,outputs", [("binary", 1), ("multiclass", 3), ("regression", 1)])
def test_identical_serialization_views_have_zero_consistency_loss(task_type, outputs):
    logits = torch.randn(3, 1, outputs).expand(-1, 2, -1)
    embeddings = torch.randn(3, 1, 8).expand(-1, 2, -1)
    assert serialization_consistency_loss(logits, embeddings, task_type=task_type).item() == pytest.approx(0, abs=1e-7)


def test_combined_auxiliary_losses_are_finite_and_targets_stop_gradient():
    torch.manual_seed(2)
    outputs = {
        "logits": torch.randn(2, 2, 1, requires_grad=True),
        "graph_logits": torch.randn(2, 1, requires_grad=True),
        "smiles_logits": torch.randn(2, 2, 1, requires_grad=True),
        "graph_embedding": torch.randn(2, 8, requires_grad=True),
        "smiles_embedding": torch.randn(2, 2, 8, requires_grad=True),
        "fused_embedding": torch.randn(2, 2, 8, requires_grad=True),
        "graph_atoms": torch.randn(2, 3, 8, requires_grad=True),
        "smiles_atoms": torch.randn(2, 2, 3, 8, requires_grad=True),
        "predicted_smiles_embedding": torch.randn(2, 8, requires_grad=True),
        "predicted_graph_embedding": torch.randn(2, 2, 8, requires_grad=True),
        "generation_logits": torch.randn(2, 3, 10, requires_grad=True),
    }
    batch = {
        "labels": torch.tensor([[0.0], [1.0]]), "label_mask": torch.ones(2, 1, dtype=torch.bool),
        "atom_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "target_ids": torch.tensor([[3, 5, 4, 0], [3, 6, 7, 4]]),
    }
    config = {
        "data": {"task_type": "binary"}, "fusion": {},
        "loss": {"lambda_atom_alignment": 0.1, "lambda_serialization_consistency": 0.1,
                 "lambda_cross_modal": 0.1, "lambda_generation": 0.1, "lambda_unimodal": 0.1},
    }
    losses = compute_losses(outputs, batch, config)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert outputs["graph_embedding"].grad is None
    assert outputs["smiles_embedding"].grad is None
    assert outputs["predicted_graph_embedding"].grad.abs().sum() > 0
    assert outputs["generation_logits"].grad[0, 2].abs().sum().item() == 0
