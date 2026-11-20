"""Prediction/evaluation CLI contracts including entirely unlabelled CSVs."""
from __future__ import annotations

import copy
import csv
import json

import numpy as np
import torch

from a2f_ood.config import DEFAULTS
from a2f_ood.data import MoleculeRow, SmilesVocabulary, preprocess_rows
from a2f_ood.engine import build_model, seed_everything
from a2f_ood.evaluate import main as evaluate_main
from a2f_ood.predict import main as predict_main


def test_unlabelled_prediction_and_group_preservation(tmp_path):
    seed_everything(11, 1)
    config = copy.deepcopy(DEFAULTS)
    config["data"].update(csv_path="unused.csv", group_column="group")
    config["model"].update(hidden_dim=16, graph_layers=1, smiles_layers=1,
                           attention_heads=2, dropout=0.0, use_cross_modal_decoder=False)
    prepared = preprocess_rows([MoleculeRow(0, "CCO", (0.0,), "a"), MoleculeRow(1, "CCN", (1.0,), "b")])
    vocab = SmilesVocabulary.build(prepared)
    model = build_model(config, vocab)
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"config": config, "vocabulary": vocab.to_dict(),
                "model_state": model.state_dict(), "group_to_id": {"a": 0, "b": 1}}, checkpoint_path)
    source = tmp_path / "unlabelled.csv"
    source.write_text("smiles,group\nCCO,a\nCCN,b\n", encoding="utf-8")
    output = tmp_path / "predictions.csv"
    assert predict_main(["--checkpoint", str(checkpoint_path), "--csv", str(source),
                         "--output", str(output), "--views", "2", "--device", "cpu"]) == 0
    with output.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["group"] for row in rows] == ["a", "b"]
    assert all(row["label"] == "" and 0 <= float(row["prediction"]) <= 1 for row in rows)
    metrics = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_labeled"] == 0 and metrics["roc_auc"] is None


def test_multiclass_evaluation_accepts_one_label_and_three_probability_columns(tmp_path):
    source = tmp_path / "multiclass.csv"
    source.write_text("label,p0,p1,p2\n0,0.8,0.1,0.1\n1,0.1,0.8,0.1\n2,0.1,0.1,0.8\n", encoding="utf-8")
    output = tmp_path / "report.json"
    assert evaluate_main(["--predictions", str(source), "--task-type", "multiclass",
                          "--label-columns", "label", "--prediction-columns", "p0", "p1", "p2",
                          "--output", str(output)]) == 0
    assert np.isclose(json.loads(output.read_text())["accuracy"], 1.0)
