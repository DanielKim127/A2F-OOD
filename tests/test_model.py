"""Behavioral checks for masking, target leakage, and optimization."""

from __future__ import annotations

import copy

import pytest
import torch

from a2f_ood.model import A2FModel, AtomTokenAligner


def tiny_config():
    return {
        "data": {"task_type": "binary", "label_columns": ["y"], "max_smiles_length": 32},
        "model": {
            "hidden_dim": 16, "attention_heads": 2, "graph_layers": 2,
            "smiles_layers": 1, "dropout": 0.0, "decoder_layers": 1,
            "use_cross_modal_decoder": True,
        },
        "fusion": {},
    }


def tiny_batch():
    torch.manual_seed(3)
    # Two molecules with different lengths; molecule 2 is an isolated atom.
    return {
        "atom_features": torch.randn(2, 3, 5),
        "bond_features": torch.randn(2, 3, 3, 3),
        "adjacency": torch.tensor([
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ], dtype=torch.bool),
        "atom_mask": torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool),
        "token_ids": torch.tensor([
            [[2, 5, 6, 7, 4], [2, 7, 6, 5, 4]],
            [[2, 8, 4, 0, 0], [2, 8, 4, 0, 0]],
        ]),
        "token_mask": torch.tensor([
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]],
        ], dtype=torch.bool),
        "atom_token_positions": torch.tensor([
            [[1, 2, 3], [3, 2, 1]], [[1, 0, 0], [1, 0, 0]],
        ]),
        "target_ids": torch.tensor([[3, 5, 6, 7, 4], [3, 8, 4, 0, 0]]),
    }


def test_gathers_corresponding_atoms_and_masks_padding():
    tokens = torch.arange(20.0).reshape(1, 5, 4)
    result = AtomTokenAligner()(tokens, torch.tensor([[3, 1, 0]]), torch.tensor([[1, 1, 0]]).bool())
    torch.testing.assert_close(result[0, 0], tokens[0, 3])
    torch.testing.assert_close(result[0, 1], tokens[0, 1])
    assert result[0, 2].count_nonzero() == 0


def test_prediction_cannot_read_teacher_forced_targets():
    model = A2FModel(tiny_config(), 12, 5, 3).eval()
    batch = tiny_batch()
    with torch.no_grad():
        first = model(batch)
        changed = dict(batch)
        changed["target_ids"] = torch.tensor([[3, 9, 9, 9, 4], [3, 9, 9, 4, 0]])
        second = model(changed)
    torch.testing.assert_close(first["logits"], second["logits"])
    assert not torch.equal(first["generation_logits"], second["generation_logits"])


def test_padding_does_not_change_predictions():
    model = A2FModel(tiny_config(), 12, 5, 3).eval()
    batch = tiny_batch()
    changed = copy.deepcopy(batch)
    changed["atom_features"][1, 1:] = 1000
    changed["bond_features"][1] = 1000
    # Hidden padded token IDs may change; their mask must keep them invisible.
    changed["token_ids"][1, :, 3:] = 11
    with torch.no_grad():
        original = model(batch)
        padded = model(changed)
    torch.testing.assert_close(original["logits"], padded["logits"], atol=1e-6, rtol=1e-5)
    assert original["atom_gate"][1, :, 1:].count_nonzero() == 0


def test_atom_index_permutation_does_not_change_prediction():
    model = A2FModel(tiny_config(), 12, 5, 3).eval()
    batch = tiny_batch()
    changed = copy.deepcopy(batch)
    permutation = torch.tensor([2, 0, 1])
    changed["atom_features"] = batch["atom_features"][:, permutation]
    changed["atom_mask"] = batch["atom_mask"][:, permutation]
    changed["bond_features"] = batch["bond_features"][:, permutation][:, :, permutation]
    changed["adjacency"] = batch["adjacency"][:, permutation][:, :, permutation]
    changed["atom_token_positions"] = batch["atom_token_positions"][:, :, permutation]
    with torch.no_grad():
        original = model(batch)
        permuted = model(changed)
    torch.testing.assert_close(original["logits"], permuted["logits"], atol=1e-6, rtol=1e-5)


def test_autoregressive_decoder_cannot_read_future_tokens():
    model = A2FModel(tiny_config(), 12, 5, 3).eval()
    batch = tiny_batch()
    changed = copy.deepcopy(batch)
    changed["target_ids"][:, 3] = 11
    with torch.no_grad():
        first = model(batch)["generation_logits"]
        second = model(changed)["generation_logits"]
    torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["adaptive", "concat", "graph", "smiles"])
def test_modes_have_finite_predictions_and_gradients(mode):
    config = tiny_config()
    config["fusion"]["mode"] = mode
    model = A2FModel(config, 12, 5, 3)
    outputs = model(tiny_batch())
    assert outputs["logits"].shape == (2, 2, 1)
    assert outputs["generation_logits"].shape == (2, 4, 12)
    assert outputs["cross_modal_error"].shape == (2, 2, 2)
    loss = outputs["logits"].square().mean() + outputs["generation_logits"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_singleton_isolated_atom_without_decoder_or_alignment():
    config = tiny_config()
    config["model"]["use_cross_modal_decoder"] = False
    config["fusion"].update({
        "atom_alignment": False, "bidirectional_cross_attention": False,
        "reliability_gate": False,
    })
    batch = tiny_batch()
    batch = {name: value[1:, :1] if name in {"token_ids", "token_mask", "atom_token_positions"} else value[1:] for name, value in batch.items()}
    model = A2FModel(config, 12, 5, 3)
    outputs = model(batch)
    assert outputs["logits"].shape == (1, 1, 1)
    assert torch.isfinite(outputs["logits"]).all()
    assert "generation_logits" not in outputs
    with pytest.raises(RuntimeError, match="disabled"):
        model.generate_smiles(batch)


def test_generation_reads_only_graph_and_stops_at_eos():
    model = A2FModel(tiny_config(), 12, 5, 3).eval()
    # Force EOS as the first generated token for a deterministic stop check.
    with torch.no_grad():
        model.decoder.output.weight.zero_()
        model.decoder.output.bias.fill_(-20)
        model.decoder.output.bias[model.eos_id] = 20
    batch = tiny_batch()
    graph_only = {key: batch[key] for key in ["atom_features", "bond_features", "adjacency", "atom_mask"]}
    generated = model.generate_smiles(graph_only, max_length=10)
    assert generated.tolist() == [[3, 4], [3, 4]]
