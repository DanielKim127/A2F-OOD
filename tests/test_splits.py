from __future__ import annotations

from dataclasses import replace

import pytest
from rdkit import Chem

from a2f_ood.data import MoleculeRow
from a2f_ood.splits import assign_scaffold_groups, audit_split_overlaps, make_split_indices


def make_rows(smiles):
    return [MoleculeRow(i, value, (float(i % 2),), "g", None) for i, value in enumerate(smiles)]


def test_preassigned_rejects_duplicate_canonical_molecule_leakage():
    rows = [MoleculeRow(0, "CCO", (0,), split="train"), MoleculeRow(1, "OCC", (0,), split="valid"), MoleculeRow(2, "C", (1,), split="test")]
    with pytest.raises(ValueError, match="leakage"):
        make_split_indices(rows, strategy="preassigned")


def test_random_split_keeps_duplicates_together_and_is_reproducible():
    rows = make_rows(["CCO", "OCC", "CCN", "CC", "C", "CCC", "CCCC", "CO"])
    first = make_split_indices(rows, strategy="random", seed=4, fractions=(.6, .2, .2))
    assert first == make_split_indices(rows, strategy="random", seed=4, fractions=(.6, .2, .2))
    assert any(0 in indices and 1 in indices for indices in first.values())
    assert sorted(index for indices in first.values() for index in indices) == list(range(len(rows)))


def test_scaffold_split_has_disjoint_scaffolds_and_explicit_acyclic_group():
    rows = make_rows(["CCO", "CCN", "c1ccccc1", "Oc1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1"])
    groups = assign_scaffold_groups(rows)
    assert groups[0] == groups[1] == "<acyclic>"
    split = make_split_indices(rows, strategy="scaffold", seed=3)
    sets = [set(groups[index] for index in indices) for indices in split.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    with pytest.raises(ValueError, match="three distinct"):
        make_split_indices(rows[:2], strategy="scaffold")


def test_size_split_holds_out_largest_molecules():
    rows = make_rows(["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"])
    split = make_split_indices(rows, strategy="size", fractions=(.5, 1/3, 1/6))
    sizes = {name: [Chem.MolFromSmiles(rows[index].smiles).GetNumAtoms() for index in indices] for name, indices in split.items()}
    assert max(sizes["train"]) <= min(sizes["valid"])
    assert max(sizes["valid"]) <= min(sizes["test"])


def test_environment_split_never_uses_labels_and_requires_group_names():
    rows = make_rows(["CCO", "CCN", "CC", "C"])
    rows = [replace(row, group=group) for row, group in zip(rows, ["a", "a", "b", "c"])]
    expected = {"train": [0, 1], "valid": [2], "test": [3]}
    assert make_split_indices(rows, strategy="environment", valid_groups=["b"], test_groups=["c"]) == expected
    with pytest.raises(ValueError, match="disjoint nonempty"):
        make_split_indices(rows, strategy="environment")


def test_official_overlap_policy_retains_every_record_and_reports_overlap(caplog):
    rows = [MoleculeRow(10, "CCO", (0,), group="a", split="train"),
            MoleculeRow(11, "OCC", (1,), group="b", split="valid"),
            MoleculeRow(12, "CCO", (0,), group="c", split="test"),
            MoleculeRow(13, "C", (1,), group="a", split="train")]
    splits = make_split_indices(rows, strategy="preassigned", overlap_policy="allow_official")
    assert splits == {"train": [0, 3], "valid": [1], "test": [2]}
    assert "1 overlapping canonical molecules" in caplog.text
    assert "3 rows" in caplog.text and "not a molecule-disjoint" in caplog.text
    audit = audit_split_overlaps(rows, splits)
    assert audit["overlapping_molecules"] == 1 and audit["overlapping_rows"] == 3
    assert audit["pair_counts"] == {"test__train": 1, "test__valid": 1, "train__valid": 1}
    assert audit["examples"][0]["row_ids_by_split"] == {"train": [10], "valid": [11], "test": [12]}


@pytest.mark.parametrize("strategy", ["random", "scaffold", "size", "environment"])
def test_official_overlap_permission_cannot_apply_to_generated_splits(strategy):
    rows = make_rows(["C", "CC", "CCC"])
    with pytest.raises(ValueError, match="only supported for preassigned"):
        make_split_indices(rows, strategy=strategy, overlap_policy="allow_official")


def test_unknown_overlap_policy_rejected():
    with pytest.raises(ValueError, match="overlap_policy"):
        make_split_indices(make_rows(["C", "CC", "CCC"]), strategy="random", overlap_policy="ignore")
