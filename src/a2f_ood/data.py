"""RDKit preprocessing, atom-aligned SMILES views, and padded molecule batches."""
from __future__ import annotations

import csv
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem

LOGGER = logging.getLogger(__name__)
SPECIAL_TOKENS = ("<pad>", "<unk>", "<cls>", "<bos>", "<eos>")
_TOKEN_PATTERN = re.compile(r"\[[^\[\]]+\]|Br|Cl|Si|Se|Na|Li|Ca|Mg|Al|se|as|[A-Z]|[bcnops]|%\([0-9]+\)|%[0-9]{2}|[0-9]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\*|\$")
_MAP_PATTERN = re.compile(r":([0-9]+)(?=\]$)")


@dataclass(frozen=True)
class MoleculeRow:
    row_id: int
    smiles: str
    labels: tuple[float | None, ...]
    group: str | None = None
    split: str | None = None


@dataclass
class PreparedMolecule:
    row: MoleculeRow
    molecule: Any
    canonical_smiles: str
    mapped_smiles: str
    canonical_tokens: list[str]
    atom_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    max_smiles_length: int = 256

    @property
    def num_atoms(self) -> int:
        return int(self.atom_features.shape[0])

    @property
    def bond_features(self) -> np.ndarray:
        """Materialize a dense graph only for a fetched minibatch item."""
        bonds = np.zeros((self.num_atoms, self.num_atoms, BOND_FEATURE_DIM), dtype=np.float32)
        bonds[self.edge_index[0], self.edge_index[1]] = self.edge_features
        return bonds

    @property
    def adjacency(self) -> np.ndarray:
        adjacency = np.zeros((self.num_atoms, self.num_atoms), dtype=bool)
        adjacency[self.edge_index[0], self.edge_index[1]] = True
        return adjacency


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip().lower() in {"", "nan", "na", "none", "null"}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def load_molecule_table(csv_path: str | Path, *, smiles_column: str, label_columns: list[str], group_column: str | None = None, split_column: str | None = None) -> list[MoleculeRow]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {smiles_column, *label_columns}
        required.update(column for column in (group_column, split_column) if column)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row_id, row in enumerate(reader):
            smiles = (row.get(smiles_column) or "").strip()
            try:
                labels = tuple(_optional_float(row.get(name)) for name in label_columns)
            except ValueError as exc:
                raise ValueError(f"Non-numeric label in CSV row {row_id + 2}") from exc
            rows.append(MoleculeRow(row_id, smiles, labels, (row.get(group_column) or None) if group_column else None, (row.get(split_column) or None) if split_column else None))
    return rows


def tokenize_mapped_smiles(mapped_smiles: str) -> tuple[list[str], list[int | None]]:
    """Lex SMILES with one-based atom maps in a separate side channel.

    Map numbers never enter token text. Unrecognized characters are errors.
    Unmapped SMILES are supported for canonical decoder targets.
    """
    tokens, atom_maps = [], []
    position = 0
    while position < len(mapped_smiles):
        match = _TOKEN_PATTERN.match(mapped_smiles, position)
        if match is None:
            raise ValueError(f"Unsupported SMILES token at position {position}: {mapped_smiles!r}")
        token = match.group(0)
        map_match = _MAP_PATTERN.search(token) if token.startswith("[") else None
        atom_maps.append(int(map_match.group(1)) if map_match else None)
        tokens.append(_MAP_PATTERN.sub("", token) if map_match else token)
        position = match.end()
    if not tokens:
        raise ValueError("SMILES cannot be empty")
    return tokens, atom_maps


def canonicalize_and_map_smiles(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"RDKit could not parse nonempty SMILES: {smiles!r}")
    molecule = Chem.Mol(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    # RDKit's local CW/CCW chiral tag depends on neighbor traversal order.
    # Reparse a canonical serialization before featurization so equivalent
    # input enumerations produce the same graph features and atom-map IDs.
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise ValueError(f"Canonical SMILES unexpectedly failed to parse: {canonical_smiles!r}")
    for index, atom in enumerate(molecule.GetAtoms()):
        atom.SetAtomMapNum(index + 1)
    return {"molecule": molecule, "canonical_smiles": canonical_smiles, "mapped_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True), "num_atoms": molecule.GetNumAtoms()}


def _one_hot(value: Any, choices: tuple[Any, ...]) -> list[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def _atom_features(atom: Any) -> list[float]:
    """Concatenated one-hot attributes in the order of the supplied table."""
    chirality = Chem.rdchem.ChiralType
    hybridization = Chem.rdchem.HybridizationType
    aromatic = bool(atom.GetIsAromatic())
    return (
        _one_hot(atom.GetAtomicNum(), tuple(range(1, 119)))
        + _one_hot(atom.GetDegree(), (0, 1, 2, 3, 4, 5))
        + _one_hot(atom.GetFormalCharge(), (-1, -2, 1, 2, 0))
        + _one_hot(atom.GetNumRadicalElectrons(), (0, 1, 2, 3, 4))
        + _one_hot(atom.GetChiralTag(), (
            chirality.CHI_UNSPECIFIED, chirality.CHI_TETRAHEDRAL_CW,
            chirality.CHI_TETRAHEDRAL_CCW, chirality.CHI_OTHER,
        ))
        + _one_hot(atom.GetNumImplicitHs(), (0, 1, 2, 3, 4))
        + _one_hot(atom.GetHybridization(), (
            hybridization.SP, hybridization.SP2, hybridization.SP3,
            hybridization.SP3D, hybridization.SP3D2,
        ))
        + [float(aromatic), float(not aromatic)]
    )


def _bond_features(bond: Any) -> list[float]:
    bond_type, stereo = Chem.rdchem.BondType, Chem.rdchem.BondStereo
    conjugated, in_ring = bool(bond.GetIsConjugated()), bool(bond.IsInRing())
    return (
        _one_hot(bond.GetBondType(), (bond_type.SINGLE, bond_type.DOUBLE, bond_type.TRIPLE, bond_type.AROMATIC))
        + _one_hot(bond.GetStereo(), (stereo.STEREONONE, stereo.STEREOANY, stereo.STEREOZ, stereo.STEREOE))
        + [float(conjugated), float(not conjugated)]
        + [float(in_ring), float(not in_ring)]
    )


ATOM_FEATURE_SIZES = {
    "atom_type": 119, "degree": 7, "formal_charge": 6, "radical_electrons": 6,
    "chirality_tag": 5, "implicit_hydrogens": 6, "hybridization": 6, "aromatic": 2,
}
BOND_FEATURE_SIZES = {"bond_type": 5, "bond_stereo": 5, "conjugated": 2, "in_ring": 2}
ATOM_FEATURE_DIM = sum(ATOM_FEATURE_SIZES.values())
BOND_FEATURE_DIM = sum(BOND_FEATURE_SIZES.values())
FEATURE_SCHEMA_ID = "table_atom157_bond14_v1"


def molecular_feature_schema() -> dict[str, Any]:
    """Version field semantics as well as widths, including the 14-wide bonds."""
    return {"id": FEATURE_SCHEMA_ID, "atom_dim": ATOM_FEATURE_DIM, "bond_dim": BOND_FEATURE_DIM,
            "atom_groups": dict(ATOM_FEATURE_SIZES), "bond_groups": dict(BOND_FEATURE_SIZES)}


def preprocess_rows(rows: Iterable[MoleculeRow], *, max_atom_count: int = 256, max_smiles_length: int = 256, invalid_smiles_policy: str = "skip") -> list[PreparedMolecule]:
    """Validate and tensorize graphs without truncating atoms or aligned tokens."""
    if invalid_smiles_policy not in {"skip", "error"}:
        raise ValueError("invalid_smiles_policy must be 'skip' or 'error'")
    prepared = []
    for row in rows:
        try:
            parsed = canonicalize_and_map_smiles(row.smiles)
            molecule, num_atoms = parsed["molecule"], parsed["num_atoms"]
            if num_atoms > max_atom_count:
                raise ValueError(f"{num_atoms} atoms exceeds max_atom_count={max_atom_count}")
            encoder_tokens, maps = tokenize_mapped_smiles(parsed["mapped_smiles"])
            target_tokens, _ = tokenize_mapped_smiles(parsed["canonical_smiles"])
            if max(len(encoder_tokens), len(target_tokens)) + 2 > max_smiles_length:
                raise ValueError(f"SMILES tokens exceed max_smiles_length={max_smiles_length}")
            if sorted(x for x in maps if x is not None) != list(range(1, num_atoms + 1)):
                raise ValueError("Atom mapping is not a bijection")
            atoms = np.asarray([_atom_features(atom) for atom in molecule.GetAtoms()], dtype=np.float32)
            edge_indices, edge_features = [], []
            for bond in molecule.GetBonds():
                first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                features = _bond_features(bond)
                edge_indices.extend(((first, second), (second, first)))
                edge_features.extend((features, features))
            edges = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 2).T
            features = np.asarray(edge_features, dtype=np.float32).reshape(-1, BOND_FEATURE_DIM)
            prepared.append(PreparedMolecule(row, molecule, parsed["canonical_smiles"], parsed["mapped_smiles"], target_tokens, atoms, edges, features, max_smiles_length))
        except (ValueError, RuntimeError) as exc:
            preview = row.smiles if len(row.smiles) <= 120 else row.smiles[:117] + "..."
            message = f"Skipping molecule row_id={row.row_id}, smiles={preview!r}: {exc}"
            if invalid_smiles_policy == "error":
                raise ValueError(message) from exc
            LOGGER.warning(message)
    if not prepared:
        raise ValueError("No valid molecules remain after preprocessing")
    return prepared


def _view_tokens(molecule: PreparedMolecule, rng: random.Random | None = None) -> tuple[list[str], list[int | None]]:
    mapped = molecule.mapped_smiles
    if rng is not None:
        permutation = list(range(molecule.num_atoms))
        rng.shuffle(permutation)
        reordered = Chem.RenumberAtoms(molecule.molecule, permutation)
        mapped = Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True)
    tokens, maps = tokenize_mapped_smiles(mapped)
    if len(tokens) + 2 > molecule.max_smiles_length:
        tokens, maps = tokenize_mapped_smiles(molecule.mapped_smiles)
    return tokens, maps


class SmilesVocabulary:
    """Train-only vocabulary serialized together with each model checkpoint."""
    def __init__(self, tokens: Iterable[str]):
        self.tokens = list(SPECIAL_TOKENS) + sorted(set(tokens) - set(SPECIAL_TOKENS))
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id, self.unk_id, self.cls_id, self.bos_id, self.eos_id = range(5)

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def build(cls, molecules: Iterable[PreparedMolecule], *, random_views: int = 2, seed: int = 42) -> SmilesVocabulary:
        tokens = set()
        for molecule in molecules:
            tokens.update(molecule.canonical_tokens)
            tokens.update(_view_tokens(molecule)[0])
            rng = random.Random(seed + molecule.row.row_id * 1009)
            for _ in range(random_views):
                tokens.update(_view_tokens(molecule, rng)[0])
        # A new traversal can invert SMILES-local @/@@ notation without
        # introducing a new molecule or a new chemical atom type.
        for token in list(tokens):
            if "@@" in token:
                tokens.add(token.replace("@@", "@"))
            elif "@" in token:
                tokens.add(token.replace("@", "@@"))
        return cls(tokens)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        result = []
        for index in ids:
            token = self.tokens[int(index)]
            if token == "<eos>":
                break
            if skip_special and token in SPECIAL_TOKENS:
                continue
            result.append(token)
        return "".join(result)

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmilesVocabulary:
        vocabulary = cls(payload["tokens"])
        if vocabulary.tokens != payload["tokens"]:
            raise ValueError("Vocabulary token order is incompatible")
        return vocabulary


class MoleculeDataset(Dataset):
    def __init__(self, molecules: list[PreparedMolecule], vocabulary: SmilesVocabulary, *, num_views: int = 1, randomize: bool = False, seed: int = 42, group_to_id: dict[str, int] | None = None) -> None:
        if num_views < 1:
            raise ValueError("num_views must be positive")
        self.molecules, self.vocabulary = molecules, vocabulary
        self.num_views, self.randomize, self.seed, self.epoch = num_views, randomize, seed, 0
        self.group_to_id = group_to_id or {group: index for index, group in enumerate(sorted({m.row.group or "ungrouped" for m in molecules}))}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, index: int) -> dict[str, Any]:
        molecule = self.molecules[index]
        row = molecule.row
        views, atom_positions = [], []
        for view in range(self.num_views):
            # Canonical first view at evaluation; other evaluation views are deterministic.
            random_view = self.randomize or view > 0
            rng = random.Random(self.seed + row.row_id * 1000003 + self.epoch * 9176 + view * 101)
            tokens, maps = _view_tokens(molecule, rng if random_view else None)
            positions = [0] * molecule.num_atoms
            for token_position, atom_map in enumerate(maps, start=1):
                if atom_map is not None:
                    positions[atom_map - 1] = token_position
            if any(position == 0 for position in positions):
                raise ValueError(f"Missing atom-token alignment for row_id={row.row_id}")
            views.append([self.vocabulary.cls_id, *self.vocabulary.encode(tokens), self.vocabulary.eos_id])
            atom_positions.append(positions)
        group = row.group or "ungrouped"
        return {"atom_features": molecule.atom_features, "bond_features": molecule.bond_features, "adjacency": molecule.adjacency, "token_ids": views, "atom_token_positions": atom_positions, "target_ids": [self.vocabulary.bos_id, *self.vocabulary.encode(molecule.canonical_tokens), self.vocabulary.eos_id], "labels": [value if value is not None else 0.0 for value in row.labels], "label_mask": [value is not None for value in row.labels], "group_id": self.group_to_id.get(group, -1), "row_id": row.row_id, "smiles": molecule.canonical_smiles, "group": group, "pad_id": self.vocabulary.pad_id}


def collate_molecules(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch_size, num_views = len(items), len(items[0]["token_ids"])
    if any(len(item["token_ids"]) != num_views for item in items):
        raise ValueError("All items in a batch must have the same number of views")
    atoms = max(item["atom_features"].shape[0] for item in items)
    length = max(len(view) for item in items for view in item["token_ids"])
    target_length = max(len(item["target_ids"]) for item in items)
    pad_id = items[0]["pad_id"]
    batch = {"atom_features": torch.zeros(batch_size, atoms, ATOM_FEATURE_DIM), "bond_features": torch.zeros(batch_size, atoms, atoms, BOND_FEATURE_DIM), "adjacency": torch.zeros(batch_size, atoms, atoms, dtype=torch.bool), "atom_mask": torch.zeros(batch_size, atoms, dtype=torch.bool), "token_ids": torch.full((batch_size, num_views, length), pad_id, dtype=torch.long), "token_mask": torch.zeros(batch_size, num_views, length, dtype=torch.bool), "atom_token_positions": torch.zeros(batch_size, num_views, atoms, dtype=torch.long), "target_ids": torch.full((batch_size, target_length), pad_id, dtype=torch.long), "labels": torch.tensor([item["labels"] for item in items], dtype=torch.float32), "label_mask": torch.tensor([item["label_mask"] for item in items], dtype=torch.bool), "group_ids": torch.tensor([item["group_id"] for item in items], dtype=torch.long), "row_ids": [item["row_id"] for item in items], "smiles": [item["smiles"] for item in items], "groups": [item["group"] for item in items]}
    for batch_index, item in enumerate(items):
        count = item["atom_features"].shape[0]
        batch["atom_features"][batch_index, :count] = torch.from_numpy(item["atom_features"])
        batch["bond_features"][batch_index, :count, :count] = torch.from_numpy(item["bond_features"])
        batch["adjacency"][batch_index, :count, :count] = torch.from_numpy(item["adjacency"])
        batch["atom_mask"][batch_index, :count] = True
        for view_index, ids in enumerate(item["token_ids"]):
            batch["token_ids"][batch_index, view_index, :len(ids)] = torch.tensor(ids)
            batch["token_mask"][batch_index, view_index, :len(ids)] = True
            batch["atom_token_positions"][batch_index, view_index, :count] = torch.tensor(item["atom_token_positions"][view_index])
        batch["target_ids"][batch_index, :len(item["target_ids"])] = torch.tensor(item["target_ids"])
    return batch
