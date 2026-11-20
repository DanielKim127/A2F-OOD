"""Chemical attribute semantics from the user-supplied feature table."""
import copy
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from rdkit import Chem

from a2f_ood.config import DEFAULTS
from a2f_ood.data import (
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM, MoleculeDataset, MoleculeRow, SmilesVocabulary,
    _atom_features, _bond_features, collate_molecules, molecular_feature_schema, preprocess_rows,
)
from a2f_ood.engine import build_model, load_checkpoint


def atom(smiles):
    return Chem.MolFromSmiles(smiles).GetAtomWithIdx(0)


def test_methane_uses_eight_one_hot_fields_and_implicit_hydrogens():
    features = np.array(_atom_features(atom("C")))
    assert ATOM_FEATURE_DIM == 157
    assert BOND_FEATURE_DIM == 14
    assert features.shape == (157,)
    # Carbon, degree zero, neutral, no radicals, unspecified chirality,
    # four implicit H, SP3, and aromatic=False, in table order.
    assert np.flatnonzero(features).tolist() == [5, 119, 130, 132, 138, 147, 151, 156]
    explicit = np.array(_atom_features(atom("[CH4]")))
    assert explicit[143:149].argmax() == 0
    np.testing.assert_array_equal(explicit, _atom_features(atom("[13CH4]")))


@pytest.mark.parametrize("smiles,index", [("[Cl-]", 0), ("[O-2]", 1), ("[Na+]", 2), ("[Mg+2]", 3), ("C", 4)])
def test_formal_charge_category_order_matches_table(smiles, index):
    assert np.array(_atom_features(atom(smiles)))[126:132].argmax() == index


@pytest.mark.parametrize("method,value,start,stop", [
    ("GetAtomicNum", 0, 0, 119),
    ("GetDegree", 6, 119, 126),
    ("GetFormalCharge", 3, 126, 132),
    ("GetNumRadicalElectrons", 5, 132, 138),
    ("GetChiralTag", 999, 138, 143),
    ("GetNumImplicitHs", 5, 143, 149),
    ("GetHybridization", Chem.rdchem.HybridizationType.UNSPECIFIED, 149, 155),
])
def test_unlisted_atom_attributes_use_their_own_unknown_category(method, value, start, stop):
    value_atom = Mock(wraps=atom("C"))
    getattr(value_atom, method).return_value = value
    field = np.array(_atom_features(value_atom))[start:stop]
    assert field.sum() == 1
    assert field[-1] == 1


@pytest.mark.parametrize("smiles,index", [("CC", 0), ("C=C", 1), ("C#C", 2), ("c1ccccc1", 3)])
def test_bond_type_category_order_matches_table(smiles, index):
    features = np.array(_bond_features(Chem.MolFromSmiles(smiles).GetBondWithIdx(0)))
    assert features.shape == (14,)
    assert features[:5].argmax() == index
    for start, stop in [(0, 5), (5, 10), (10, 12), (12, 14)]:
        assert features[start:stop].sum() == 1


def test_bond_stereo_conjugation_and_ring_fields_are_separate():
    trans = Chem.MolFromSmiles("C/C=C/C").GetBondWithIdx(1)
    cis = Chem.MolFromSmiles(r"C/C=C\C").GetBondWithIdx(1)
    assert np.array(_bond_features(trans))[5:10].argmax() == 3
    assert np.array(_bond_features(cis))[5:10].argmax() == 2
    aromatic = np.array(_bond_features(Chem.MolFromSmiles("c1ccccc1").GetBondWithIdx(0)))
    assert aromatic[10:].tolist() == [1, 0, 1, 0]
    single = np.array(_bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))
    assert single[10:].tolist() == [0, 1, 0, 1]
    unknown = Mock(wraps=trans)
    unknown.GetBondType.return_value = 999
    unknown.GetStereo.return_value = 999
    encoded = np.array(_bond_features(unknown))
    assert encoded[4] == encoded[9] == 1


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_formal_hidden_width_accepts_table_features_and_backpropagates(device):
    torch.set_num_threads(1)
    molecules = preprocess_rows([MoleculeRow(0, "CCO", (1.0,)), MoleculeRow(1, "c1ccccc1", (0.0,))])
    vocabulary = SmilesVocabulary.build(molecules)
    dataset = MoleculeDataset(molecules, vocabulary, num_views=2)
    batch = collate_molecules([dataset[0], dataset[1]])
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    config = copy.deepcopy(DEFAULTS)
    model = build_model(config, vocabulary).to(device)
    assert model.hidden_dim == 256
    assert model.graph_encoder.input_projection.in_features == 157
    assert model.graph_encoder.layers[0].edge_encoder.in_features == 14
    outputs = model(batch)
    assert outputs["fused_embedding"].shape == (2, 2, 256)
    assert outputs["graph_atoms"].shape == (2, 6, 256)
    outputs["logits"].square().mean().backward()
    assert model.graph_encoder.input_projection.weight.grad.shape == (256, 157)
    assert torch.isfinite(model.graph_encoder.layers[0].edge_encoder.weight.grad).all()


@pytest.mark.parametrize("old_width,wrong_schema", [(42, False), (157, True)])
def test_incompatible_checkpoint_attributes_fail_with_a_clear_message(tmp_path, old_width, wrong_schema):
    checkpoint = {"model_state": {"graph_encoder.input_projection.weight": torch.zeros(16, old_width)}}
    if wrong_schema:
        checkpoint["feature_schema"] = {**molecular_feature_schema(), "id": "different_bond_semantics"}
    path = tmp_path / "old.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="feature schema.*incompatible"):
        load_checkpoint(path, torch.device("cpu"))
