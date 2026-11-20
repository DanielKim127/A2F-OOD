"""Regression checks for group sampling, staged objectives and unsafe updates."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from a2f_ood.config import DEFAULTS
from a2f_ood.engine import train_epoch
from a2f_ood.losses import GroupDRO, compute_losses
from a2f_ood.train import epoch_objectives


def test_equal_group_risks_do_not_reward_more_frequent_batches():
    counts = [90, 10]
    dro = GroupDRO(2, step_size=1, update_mode="epoch", group_counts=counts)
    initial = dro.group_weights.clone()
    dro.begin_epoch()
    for group, repeats in [(0, 9), (1, 1)]:
        for _ in range(repeats):
            dro(torch.ones(10), torch.full((10,), group))
    assert torch.equal(dro.group_weights, initial)
    dro.end_epoch()
    torch.testing.assert_close(dro.group_weights, initial)
    assert dro.epoch_sample_counts.tolist() == counts


def test_epoch_adversary_is_invariant_to_batch_partition_and_order():
    losses = torch.tensor([0.4, 1.2, 2.0, 0.8, 1.7, 0.3])
    groups = torch.tensor([0, 1, 1, 0, 1, 0])
    whole = GroupDRO(2, step_size=.4, update_mode="epoch", group_counts=[3, 3])
    split = GroupDRO(2, step_size=.4, update_mode="epoch", group_counts=[3, 3])
    whole(losses, groups)
    for index in [4, 0, 5, 2, 1, 3]:
        split(losses[index:index + 1], groups[index:index + 1])
    whole.end_epoch(); split.end_epoch()
    torch.testing.assert_close(whole.group_weights, split.group_weights)


def test_importance_risk_matches_weighted_group_risk_under_imbalanced_sampling():
    dro = GroupDRO(2, update_mode="epoch", group_counts=[90, 10], max_weight_ratio=10)
    dro.group_weights.copy_(torch.tensor([.5, .5]))
    losses = torch.cat([torch.ones(90), torch.full((10,), 3.0)]).requires_grad_()
    groups = torch.tensor([0] * 90 + [1] * 10)
    risk = dro(losses, groups, update=False)
    assert risk.item() == pytest.approx(2.0)
    risk.backward()
    assert losses.grad[:90].sum().item() == pytest.approx(.5)
    assert losses.grad[90:].sum().item() == pytest.approx(.5)


def test_repeated_hard_rare_group_updates_have_bounded_sample_influence():
    dro = GroupDRO(2, step_size=1, update_mode="epoch", group_counts=[99, 1], max_weight_ratio=3)
    for _ in range(30):
        dro.begin_epoch()
        dro(torch.cat([torch.ones(99), torch.tensor([100.0])]), torch.tensor([0] * 99 + [1]))
        diagnostics = dro.end_epoch()
        assert dro.group_weights.sum().item() == pytest.approx(1)
        assert diagnostics["group_dro_max_importance"] <= 3.00001
    assert dro.group_weights[1].item() == pytest.approx(.03, abs=1e-6)
    weights = dro.group_weights.clone()
    dro.eval()
    dro(torch.tensor([1000.0]), torch.tensor([0]))
    dro.end_epoch()
    assert torch.equal(weights, dro.group_weights)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_unlabelled_groups_have_no_adversarial_mass_or_gradient_on_cpu_and_cuda(device):
    dro = GroupDRO(3, update_mode="epoch", group_counts=[2, 0, 1]).to(device)
    losses = torch.tensor([1.0, float("nan"), 4.0], device=device, requires_grad=True)
    risk = dro(losses, torch.tensor([0, 1, 2], device=device), torch.tensor([True, False, True], device=device))
    risk.backward()
    diagnostics = dro.end_epoch()
    assert torch.isfinite(risk)
    assert losses.grad[1].item() == 0
    assert dro.group_weights[1].item() == 0
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())


def test_supervised_risk_keeps_erm_anchor_and_bounded_per_sample_gradients():
    dro = GroupDRO(2, update_mode="epoch", group_counts=[9, 1], max_weight_ratio=3)
    dro.group_weights.copy_(torch.tensor([.7, .3]))
    logits = torch.tensor([[[0.0]], [[-2.0]]], requires_grad=True)
    labels = torch.tensor([[0.0], [1.0]])
    batch = {"labels": labels, "label_mask": torch.ones(2, 1, dtype=torch.bool), "group_ids": torch.tensor([0, 1])}
    config = {"data": {"task_type": "binary"}, "loss": {"use_group_dro": True, "group_dro_strength": .25}}
    result = compute_losses({"logits": logits}, batch, config, dro, update_dro=False)
    baseline = F.binary_cross_entropy_with_logits(logits, labels[:, None])
    base_gradient = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
    result["total"].backward()
    ratios = logits.grad / base_gradient
    assert ratios.min().item() >= .75
    assert ratios.max().item() <= 1.50001
    assert result["robust"].item() == pytest.approx(.75 * result["task"].item() + .25 * result["dro"].item())


def test_supervised_warmup_cannot_train_auxiliary_heads_or_mutate_base_config():
    config = copy.deepcopy(DEFAULTS)
    config["loss"]["use_group_dro"] = False
    original = copy.deepcopy(config)
    warmup, scale = epoch_objectives(config, 1)
    assert scale == 0
    assert warmup["loss"]["group_dro_strength"] == 0
    logits = torch.randn(2, 2, 1, requires_grad=True)
    graph_head = torch.randn(2, 1, requires_grad=True)
    smiles_head = torch.randn(2, 2, 1, requires_grad=True)
    outputs = {"logits": logits, "graph_logits": graph_head, "smiles_logits": smiles_head}
    batch = {"labels": torch.tensor([[0.0], [1.0]]), "label_mask": torch.ones(2, 1, dtype=torch.bool)}
    losses = compute_losses(outputs, batch, warmup)
    losses["total"].backward()
    assert graph_head.grad is None and smiles_head.grad is None
    assert logits.grad.abs().sum() > 0
    assert config == original
    complete, scale = epoch_objectives(config, 12)
    assert scale == 1
    assert complete["loss"]["group_dro_strength"] == pytest.approx(.25)


def test_finite_loss_with_nonfinite_gradient_is_rejected_before_optimizer_step():
    class BadGradientModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            return {"logits": self.value.sqrt().expand(1, 1, 1)}

    model = BadGradientModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    batch = {"labels": torch.ones(1, 1), "label_mask": torch.ones(1, 1, dtype=torch.bool), "row_ids": [0]}
    config = {"data": {"task_type": "binary"}, "loss": {}, "training": {"gradient_clip": 1.0}}
    with pytest.raises(RuntimeError, match="non-finite"):
        train_epoch(model, [batch], optimizer, config, torch.device("cpu"))
    assert model.value.item() == 0
    assert not optimizer.state
