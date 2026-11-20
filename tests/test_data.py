from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest
import numpy as np
import torch
from rdkit import Chem

from a2f_ood.data import (
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM, MoleculeRow, MoleculeDataset,
    SmilesVocabulary, canonicalize_and_map_smiles, collate_molecules,
    load_molecule_table, preprocess_rows, tokenize_mapped_smiles,
)


def rows(smiles):
    return [MoleculeRow(i, value, (1.0, None), "g", "train") for i, value in enumerate(smiles)]


@pytest.mark.parametrize("smiles", ["CC(Cl)Br", "Cc1ccccc1", "N[C@@H](C)C(=O)O", "[NH4+].[Cl-]", "c1cc[nH]c1", "[13CH3]CO", "[Na+]", "C%12CCCCC%12"])
def test_atom_mapping_is_bijective_and_never_enters_model(smiles):
    parsed = canonicalize_and_map_smiles(smiles)
    tokens, maps = tokenize_mapped_smiles(parsed["mapped_smiles"])
    assert sorted(index for index in maps if index is not None) == list(range(1, parsed["num_atoms"] + 1))
    assert all(not re.search(r":\d+\]", token) for token in tokens)
    restored = Chem.MolFromSmiles("".join(tokens))
    assert restored is not None
    assert Chem.MolToSmiles(restored) == parsed["canonical_smiles"]


def test_input_atom_maps_are_not_part_of_molecule_identity():
    parsed = canonicalize_and_map_smiles("[CH3:300][OH:100]")
    assert parsed["canonical_smiles"] == "CO"
    assert parsed["num_atoms"] == 2


def test_equivalent_chiral_input_enumerations_have_identical_graph_features():
    first, second = preprocess_rows(rows(["N[C@@H](C)C(=O)O", "C[C@H](N)C(=O)O"]))
    assert first.canonical_smiles == second.canonical_smiles
    np.testing.assert_array_equal(first.atom_features, second.atom_features)
    np.testing.assert_array_equal(first.edge_index, second.edge_index)
    np.testing.assert_array_equal(first.edge_features, second.edge_features)
    assert first.mapped_smiles == second.mapped_smiles


def test_tokenizer_preserves_chemical_symbols_and_errors_on_garbage():
    tokens, maps = tokenize_mapped_smiles("ClC(Br)C%12CCCCC%12")
    assert tokens[:4] == ["Cl", "C", "(", "Br"]
    assert tokens.count("%12") == 2
    assert tokenize_mapped_smiles("Cc1ccccc1")[0][:2] == ["C", "c"]
    assert all(value is None for value in maps)
    with pytest.raises(ValueError, match="Unsupported"):
        tokenize_mapped_smiles("CC?O")


def test_random_views_are_deterministic_aligned_and_stereochemically_valid():
    molecules = preprocess_rows(rows(["N[C@@H](C)C(=O)O", "CC(Cl)Br"]))
    vocabulary = SmilesVocabulary.build(molecules, random_views=4)
    dataset = MoleculeDataset(molecules, vocabulary, num_views=3, randomize=True, seed=12)
    first = dataset[0]
    assert first["token_ids"] == dataset[0]["token_ids"]
    for token_ids, positions in zip(first["token_ids"], first["atom_token_positions"]):
        assert sorted(positions) == sorted(set(positions))
        assert len(positions) == molecules[0].num_atoms
        assert all(token_ids[position] != vocabulary.pad_id for position in positions)
        decoded = vocabulary.decode(token_ids)
        assert Chem.MolToSmiles(Chem.MolFromSmiles(decoded)) == molecules[0].canonical_smiles
    dataset.set_epoch(1)
    again = MoleculeDataset(molecules, vocabulary, num_views=3, randomize=True, seed=12)
    again.set_epoch(1)
    assert dataset[0]["token_ids"] == again[0]["token_ids"]


def test_batch_padding_missing_labels_and_isolated_atom():
    molecules = preprocess_rows(rows(["CCO", "[Na+]"]))
    vocabulary = SmilesVocabulary.build(molecules)
    dataset = MoleculeDataset(molecules, vocabulary, num_views=2)
    batch = collate_molecules([dataset[0], dataset[1]])
    assert batch["atom_features"].shape == (2, 3, ATOM_FEATURE_DIM)
    assert batch["bond_features"].shape == (2, 3, 3, BOND_FEATURE_DIM)
    assert batch["atom_mask"].sum(dim=1).tolist() == [3, 1]
    assert batch["label_mask"].tolist() == [[True, False], [True, False]]
    assert torch.isfinite(batch["labels"]).all()
    assert not batch["adjacency"][1].any()
    assert not batch["atom_features"][1, 1:].any()
    assert batch["row_ids"] == [0, 1]


def test_prepared_graphs_store_edges_without_quadratic_cache():
    molecule = preprocess_rows(rows(["CCCCCCCCCCCCCCCCCCCC"] ))[0]
    assert molecule.edge_index.shape == (2, 38)
    assert molecule.edge_features.shape == (38, BOND_FEATURE_DIM)
    assert "adjacency" not in vars(molecule) and "bond_features" not in vars(molecule)
    assert molecule.bond_features.shape == (20, 20, BOND_FEATURE_DIM)
    assert molecule.adjacency.sum() == 38


def test_invalid_and_too_long_molecules_are_reported(caplog):
    molecules = preprocess_rows(rows(["C", "not valid smiles", ""]), max_smiles_length=5)
    assert len(molecules) == 1
    assert "row_id=1" in caplog.text and "row_id=2" in caplog.text
    with pytest.raises(ValueError, match="exceeds max_atom_count"):
        preprocess_rows(rows(["CCC"]), max_atom_count=2, invalid_smiles_policy="error")
    with pytest.raises(ValueError, match="max_smiles_length"):
        preprocess_rows(rows(["C1CCCCC1"]), max_smiles_length=5, invalid_smiles_policy="error")


def test_vocab_round_trip_and_train_only_unknown():
    training = preprocess_rows(rows(["CCO"]))
    vocabulary = SmilesVocabulary.build(training)
    clone = SmilesVocabulary.from_dict(vocabulary.to_dict())
    assert clone.tokens == vocabulary.tokens
    assert clone.encode(["[U]"]) == [vocabulary.unk_id]


def test_csv_retains_invalid_rows_and_handles_missing_labels(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("smiles,a,b\nCCO,1,nan\n,0,\nC,nope,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-numeric label"):
        load_molecule_table(path, smiles_column="smiles", label_columns=["a", "b"])
    path.write_text("smiles,a,b\nCCO,1,nan\n,0,\n", encoding="utf-8")
    loaded = load_molecule_table(path, smiles_column="smiles", label_columns=["a", "b"])
    assert len(loaded) == 2 and loaded[1].smiles == ""
    assert loaded[0].labels == (1.0, None)
