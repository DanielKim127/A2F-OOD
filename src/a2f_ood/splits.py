"""Reproducible OOD splits with canonical-molecule leakage checks."""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Iterable

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


LOGGER = logging.getLogger(__name__)


def _row(item: Any) -> Any:
    return getattr(item, "row", item)


def _identity(item: Any) -> str:
    if hasattr(item, "canonical_smiles"):
        return item.canonical_smiles
    molecule = Chem.MolFromSmiles(_row(item).smiles)
    if molecule is None:
        raise ValueError(f"Cannot split invalid molecule: {_row(item).smiles}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def assign_scaffold_groups(rows: list[Any]) -> list[str]:
    """Murcko scaffolds; all acyclic molecules share one explicit group."""
    groups = []
    for item in rows:
        molecule = getattr(item, "molecule", None)
        if molecule is None:
            molecule = Chem.MolFromSmiles(_row(item).smiles)
        if molecule is None:
            raise ValueError(f"Cannot determine scaffold for {_row(item).smiles}")
        molecule = Chem.Mol(molecule)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        groups.append(scaffold or "<acyclic>")
    return groups


def audit_split_overlaps(rows: list[Any], splits: dict[str, list[int]], *, max_examples: int = 10) -> dict[str, Any]:
    """Summarize exact canonical-molecule overlap without modifying any records.

    This audit is label-free. Assay records of the same ligand can legitimately
    belong to different supplied official subsets, but those subsets are then
    not molecule-disjoint. Report both the benchmark protocol and this overlap.
    """
    membership: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for name, indices in splits.items():
        for index in indices:
            membership[_identity(rows[index])][name].append(_row(rows[index]).row_id)
    overlap = {identity: groups for identity, groups in membership.items() if len(groups) > 1}
    pairs: dict[str, int] = defaultdict(int)
    for groups in overlap.values():
        names = sorted(groups)
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                pairs[f"{names[first]}__{names[second]}"] += 1
    return {
        "overlapping_molecules": len(overlap),
        "overlapping_rows": sum(len(ids) for groups in overlap.values() for ids in groups.values()),
        "pair_counts": dict(pairs),
        "examples": [{"canonical_smiles": identity, "row_ids_by_split": dict(groups)}
                     for identity, groups in list(overlap.items())[:max_examples]],
    }


def _validate_splits(rows: list[Any], splits: dict[str, list[int]], *, overlap_policy: str = "error") -> dict[str, list[int]]:
    identities = {}
    seen_indices = set()
    for name, indices in splits.items():
        if not indices:
            raise ValueError(f"Split '{name}' is empty; supply more molecules/groups or a preassigned split")
        for index in indices:
            if index in seen_indices:
                raise ValueError(f"Row index appears in multiple splits: {index}")
            seen_indices.add(index)
            identity = _identity(rows[index])
            previous_name, previous_row = identities.setdefault(identity, (name, _row(rows[index]).row_id))
            if previous_name != name and overlap_policy == "error":
                raise ValueError(
                    f"Molecule leakage across {previous_name}/{name}: {identity} "
                    f"(row_id={previous_row} and row_id={_row(rows[index]).row_id}). "
                    "Strict molecule-disjoint evaluation rejects this overlap without "
                    "silently dropping records; inspect repeated compounds in the supplied "
                    "assay/source split and choose an explicit evaluation protocol."
                )
    if seen_indices != set(range(len(rows))):
        raise ValueError("Split indices must cover every input row")
    if overlap_policy == "allow_official":
        audit = audit_split_overlaps(rows, splits)
        if audit["overlapping_molecules"]:
            example = audit["examples"][0]
            LOGGER.warning(
                "Official preassigned split retained %d overlapping canonical molecules "
                "across subsets (%d rows; pairs=%s). Example: %s, row IDs %s. "
                "No records were dropped; this is not a molecule-disjoint evaluation.",
                audit["overlapping_molecules"], audit["overlapping_rows"], audit["pair_counts"],
                example["canonical_smiles"], example["row_ids_by_split"],
            )
    return {name: sorted(indices) for name, indices in splits.items()}


def _allocate_groups(grouped: dict[str, list[int]], fractions: tuple[float, float, float], seed: int) -> dict[str, list[int]]:
    if len(grouped) < 3:
        raise ValueError("At least three distinct molecule/scaffold groups are needed for a three-way split")
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    splits = {"train": [], "valid": [], "test": []}
    total = sum(len(group) for group in groups)
    targets = (total * fractions[0], total * fractions[1])
    cursor = 0
    for position, (name, target) in enumerate(zip(("train", "valid"), targets)):
        reserve = 2 - position
        while cursor < len(groups) - reserve and (len(splits[name]) < target or not splits[name]):
            splits[name].extend(groups[cursor])
            cursor += 1
    for group in groups[cursor:]:
        splits["test"].extend(group)
    return splits


def make_split_indices(rows: list[Any], *, strategy: str = "preassigned", seed: int = 42, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1), valid_groups: Iterable[str] | None = None, test_groups: Iterable[str] | None = None, overlap_policy: str = "error") -> dict[str, list[int]]:
    """Split without labels; size sends largest molecules to test.

    Environment splits require explicit held-out groups. Scaffold splits keep
    acyclic molecules together and fail when fewer than three scaffolds exist.
    ``allow_official`` retains cross-subset exact-molecule overlap only for
    supplied preassigned splits and reports it; ``error`` is the strict default.
    """
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    if overlap_policy not in {"error", "allow_official"}:
        raise ValueError("overlap_policy must be 'error' or 'allow_official'")
    if overlap_policy == "allow_official" and strategy not in {"preassigned", "predefined", "column"}:
        raise ValueError("allow_official overlap_policy is only supported for preassigned official splits")
    if len(fractions) != 3 or any(value <= 0 for value in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Split fractions must contain three positive values summing to one")
    splits = {name: [] for name in ("train", "valid", "test")}
    if strategy in {"preassigned", "predefined", "column"}:
        aliases = {"val": "valid", "validation": "valid", "training": "train", "testing": "test"}
        for index, item in enumerate(rows):
            split = (_row(item).split or "").strip().lower()
            split = aliases.get(split, split)
            if split not in splits:
                raise ValueError(f"Unknown/missing split {split!r} at row_id={_row(item).row_id}")
            splits[split].append(index)
    elif strategy in {"environment", "assay"}:
        valid, test = set(map(str, valid_groups or [])), set(map(str, test_groups or []))
        if not valid or not test or valid & test:
            raise ValueError("Environment split needs disjoint nonempty valid_groups and test_groups")
        for index, item in enumerate(rows):
            group = _row(item).group
            if group is None:
                raise ValueError("Environment split requires group metadata for every molecule")
            name = "valid" if group in valid else "test" if group in test else "train"
            splits[name].append(index)
    elif strategy in {"random", "scaffold"}:
        keys = assign_scaffold_groups(rows) if strategy == "scaffold" else [_identity(item) for item in rows]
        grouped = defaultdict(list)
        for index, key in enumerate(keys):
            grouped[key].append(index)
        splits = _allocate_groups(dict(grouped), fractions, seed)
    elif strategy == "size":
        grouped, sizes = defaultdict(list), {}
        for index, item in enumerate(rows):
            key = _identity(item)
            grouped[key].append(index)
            sizes[key] = Chem.MolFromSmiles(key).GetNumAtoms()
        if len(grouped) < 3:
            raise ValueError("Size split needs at least three distinct molecules")
        keys = sorted(grouped, key=lambda key: (sizes[key], key))
        n_train = min(max(round(len(rows) * fractions[0]), 1), len(rows) - 2)
        n_valid = max(round(len(rows) * fractions[1]), 1)
        for position, key in enumerate(keys):
            remaining = len(keys) - position
            if len(splits["train"]) < n_train and remaining > 2:
                name = "train"
            elif len(splits["valid"]) < n_valid and remaining > 1:
                name = "valid"
            else:
                name = "test"
            splits[name].extend(grouped[key])
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    return _validate_splits(rows, splits, overlap_policy=overlap_policy)
