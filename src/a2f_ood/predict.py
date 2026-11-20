"""Run a saved checkpoint on labelled or unlabelled molecules."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from a2f_ood.data import MoleculeDataset, load_molecule_table, preprocess_rows
from a2f_ood.engine import (evaluate_model, get_device, load_checkpoint, make_loader,
                           seed_everything, write_json, write_predictions)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict molecular properties with a saved A2F-OOD model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--generate-smiles", action="store_true")
    args = parser.parse_args(argv)
    if args.views < 1:
        raise ValueError("--views must be >= 1")
    device = get_device(args.device)
    model, vocabulary, checkpoint = load_checkpoint(args.checkpoint, device)
    config = copy.deepcopy(checkpoint["config"])
    seed_everything(config["experiment"]["seed"], config["training"].get("cpu_threads", 4))
    dc = config["data"]
    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as stream:
        columns = set(csv.DictReader(stream).fieldnames or [])
    # Preserve individual missing target columns instead of discarding all labels.
    labels_present = [c for c in dc["label_columns"] if c in columns]
    group_column = args.group_column or (dc.get("group_column") if dc.get("group_column") in columns else None)
    rows = load_molecule_table(args.csv, smiles_column=args.smiles_column or dc["smiles_column"],
                               label_columns=labels_present, group_column=group_column)
    position = {c: i for i, c in enumerate(labels_present)}
    rows = [replace(r, labels=tuple(r.labels[position[c]] if c in position else None for c in dc["label_columns"])) for r in rows]
    prepared = preprocess_rows(rows, max_atom_count=config["model"]["max_atom_count"],
                               max_smiles_length=dc["max_smiles_length"], invalid_smiles_policy=dc["invalid_smiles_policy"])
    if not prepared:
        raise ValueError("Input contains no valid molecules")
    valid_ids = {m.row.row_id for m in prepared}
    write_json(Path(args.output).with_suffix(".skipped.json"), [
        {"row_id": r.row_id, "smiles": r.smiles} for r in rows if r.row_id not in valid_ids
    ])
    dataset = MoleculeDataset(prepared, vocabulary, num_views=args.views, randomize=False,
                               seed=config["experiment"]["seed"], group_to_id=checkpoint["group_to_id"])
    config["evaluation"]["generate_smiles"] = args.generate_smiles
    if args.generate_smiles and not config["model"]["use_cross_modal_decoder"]:
        raise ValueError("Checkpoint was trained without a SMILES decoder")
    metrics, output_rows = evaluate_model(model, make_loader(dataset, config), config, device, vocabulary=vocabulary)
    write_predictions(Path(args.output), output_rows)
    write_json(Path(args.output).with_suffix(".metrics.json"), metrics)
    print(f"Predicted {len(output_rows)} molecules -> {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
