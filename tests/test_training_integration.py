"""Tiny, synthetic end-to-end tests; these labels have no scientific meaning."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from a2f_ood.config import DEFAULTS
from a2f_ood.engine import load_checkpoint
from a2f_ood.engine import build_model, evaluate_model, make_loader, seed_everything
from a2f_ood.data import MoleculeDataset, MoleculeRow, SmilesVocabulary, preprocess_rows
from a2f_ood.train import main


@pytest.mark.parametrize("task", ["binary", "multilabel", "regression", "multiclass"])
def test_train_checkpoint_and_heldout_prediction_pipeline(task, tmp_path):
    csv_path = tmp_path / "molecules.csv"
    columns = ["smiles", "label", "group", "split"]
    if task == "multilabel":
        columns.append("aux")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index in range(20):
            label = index % 3 if task == "multiclass" else index / 10 if task == "regression" else index % 2
            row = {"smiles": "C" * (index + 1), "label": label, "group": str(index % 3),
                   "split": "train" if index < 12 else "valid" if index < 16 else "test"}
            if task == "multilabel":
                row["aux"] = "" if index % 3 == 0 else (index + 1) % 2
            writer.writerow(row)
    config = copy.deepcopy(DEFAULTS)
    config["experiment"].update(output_dir=str(tmp_path / "run"), seed=17)
    config["data"].update(csv_path=str(csv_path), task_type=task, group_column="group", split_column="split",
                          max_smiles_length=64, label_columns=["label", "aux"] if task == "multilabel" else ["label"])
    if task == "multiclass":
        config["data"]["num_classes"] = 3
    config["model"].update(hidden_dim=16, graph_layers=1, smiles_layers=1, attention_heads=2,
                           dropout=0.0, max_atom_count=32, use_cross_modal_decoder=True)
    config["training"].update(epochs=2, patience=2, batch_size=6, device="cpu", cpu_threads=1,
                              learning_rate_warmup_epochs=1, lr_scheduler_patience=0,
                              monitor_min_delta=100.0)
    config["loss"].update(use_group_dro=True, auxiliary_warmup_epochs=1,
                         supervised_warmup_epochs=0, group_dro_warmup_epochs=0, group_dro_ramp_epochs=1,
                         classification_weighting="balanced")
    config["evaluation"].update(random_smiles_views=2, generate_smiles=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    selection_only = task == "multilabel"
    assert main(["--config", str(config_path), *(["--selection-only"] if selection_only else [])]) == 0
    run = tmp_path / "run"
    report = json.loads((run / ("selection_metrics.json" if selection_only else "metrics.json")).read_text(encoding="utf-8"))
    if selection_only:
        assert report["test_evaluated"] is False
        assert "test" not in report
        assert not (run / "test_predictions.csv").exists()
        assert not (run / "metrics.json").exists()
    else:
        assert report["test_evaluated"] is True
    assert report["best_epoch"] in (1, 2)
    assert report["valid"]["task_loss"] >= 0
    assert len(json.loads((run / "history.json").read_text(encoding="utf-8"))) == 2
    for split in (("valid",) if selection_only else ("valid", "test")):
        with (run / f"{split}_predictions.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4
        assert "cross_modal_error" in rows[0]
        assert "smiles_prediction_variance" in rows[0]
    model, vocabulary, checkpoint = load_checkpoint(run / "best.pt", torch.device("cpu"))
    assert len(vocabulary) > 5
    assert checkpoint["config"]["data"]["task_type"] == task
    if task in {"binary", "multilabel"}:
        weights = json.loads((run / "class_weighting.json").read_text(encoding="utf-8"))
        assert weights["fitted_on"] == "retained_training_labels_only"
        assert weights["tasks"][0]["positive_count"] == 6
        assert weights["tasks"][0]["negative_count"] == 6
        assert checkpoint["config"]["loss"]["binary_class_weights"] == weights["weights"]
        assert "class_weighted_task_loss" in report["valid"]
    assert checkpoint["group_dro_state"] is not None
    assert checkpoint["feature_schema"]["atom_dim"] == 157
    assert checkpoint["feature_schema"]["bond_dim"] == 14
    assert model.config["model"]["hidden_dim"] == 16
    last = torch.load(run / "last.pt", map_location="cpu")
    assert last["epoch"] == 2
    assert last["scheduler_state"] is not None
    assert last["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(config["training"]["learning_rate"] * .5)
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    assert history[-1]["train"]["group_dro_max_importance"] <= 3.00001
    assert history[-1]["train"]["gradient_norm_mean"] >= 0
    assert history[-1]["train"]["num_labeled"] == 12
    assert history[-1]["train"]["auxiliary_weighted"] > 0


@pytest.mark.parametrize("generated", ["empty", "valid", "unknown"])
def test_generation_evaluation_omits_targets_and_does_not_count_empty_molecules(monkeypatch, generated):
    seed_everything(9, cpu_threads=1)
    config = copy.deepcopy(DEFAULTS)
    config["data"].update(csv_path="unused.csv", max_smiles_length=32)
    config["model"].update(hidden_dim=16, graph_layers=1, smiles_layers=1, attention_heads=2,
                           decoder_layers=1, dropout=0.0, use_cross_modal_decoder=True)
    config["evaluation"]["generate_smiles"] = True
    molecules = preprocess_rows([MoleculeRow(0, "CCO", (1.0,), "a", "test")])
    vocabulary = SmilesVocabulary.build(molecules)
    dataset = MoleculeDataset(molecules, vocabulary)
    model = build_model(config, vocabulary)
    original_forward = model.forward
    seen = []

    def property_forward(batch):
        assert "target_ids" not in batch
        seen.append(True)
        return original_forward(batch)

    monkeypatch.setattr(model, "forward", property_forward)
    tokens = [] if generated == "empty" else [vocabulary.unk_id] if generated == "unknown" else vocabulary.encode(["C", "C", "O"])
    sequence = torch.tensor([[vocabulary.bos_id, *tokens, vocabulary.eos_id]])
    monkeypatch.setattr(model, "generate_smiles", lambda batch, max_length: sequence)
    metrics, predictions = evaluate_model(model, make_loader(dataset, config), config,
                                          torch.device("cpu"), vocabulary=vocabulary)
    assert seen and len(predictions) == 1
    assert metrics["generation_validity"] == (1.0 if generated == "valid" else 0.0)
    assert metrics["generation_exact_match"] == (1.0 if generated == "valid" else 0.0)
    if generated == "unknown":
        assert "<unk>" in predictions[0]["generated_smiles"]
