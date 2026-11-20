"""Download provenance and lossless adaptation of official OOD split JSONs.

No label thresholds or data splits are inferred here. Original raw files are
retained; the training CSV selects official train/ood_val/ood_test only.
"""

from __future__ import annotations

import csv
import hashlib
import html
from html.parser import HTMLParser
import http.cookiejar
import json
import math
from pathlib import Path
import ssl
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, HTTPSHandler, Request, build_opener
from datetime import datetime, timezone


OFFICIAL_SPLITS = ("train", "iid_val", "iid_test", "ood_val", "ood_test")
TRAINING_SPLITS = {"train": "train", "ood_val": "valid", "ood_test": "test"}
CSV_FIELDS = ("molecule_id", "smiles", "label", "group", "split", "official_split", "source_index", "assay_id", "domain_id")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _html_response(prefix: bytes, content_type: str = "") -> bool:
    lowered = prefix.lstrip().lower()
    return "text/html" in content_type.lower() or lowered.startswith((b"<!doctype html", b"<html"))


class _DownloadForm(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.fields: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "download-form":
            self.action = attributes.get("action")
        if tag == "input" and attributes.get("name") and attributes.get("value") is not None:
            self.fields[str(attributes["name"])] = str(attributes["value"])


def _drive_download_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname not in {"drive.google.com", "drive.usercontent.google.com", "docs.google.com"}:
        return url
    file_id = parse_qs(parsed.query).get("id", [None])[0]
    if file_id is None and "/file/d/" in parsed.path:
        file_id = parsed.path.split("/file/d/", 1)[1].split("/", 1)[0]
    if file_id:
        return "https://drive.usercontent.google.com/download?" + urlencode({"id": file_id, "export": "download", "confirm": "t"})
    return url


def download_file(
    url: str,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    overwrite: bool = False,
    timeout: int = 90,
) -> dict[str, Any]:
    """Stream a verified HTTPS file, handling public Drive confirmation forms.

    HTML landing/error pages, incomplete files, and hash mismatches are errors.
    TLS certificate verification is always enabled. A caller behind a custom
    trusted proxy can set SSL_CERT_FILE to its certificate bundle.
    """
    destination = Path(destination)
    if urlparse(url).scheme != "https":
        raise ValueError("Dataset downloads require HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_name(destination.name + ".download.json")
    if destination.exists() and not overwrite:
        with destination.open("rb") as stream:
            if _html_response(stream.read(512)):
                raise ValueError(f"Existing file contains HTML, not dataset content: {destination}")
        size = destination.stat().st_size
        digest = file_sha256(destination)
        if expected_bytes is not None and size != expected_bytes:
            raise ValueError(f"Existing file size mismatch: {size} != {expected_bytes}")
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError("Existing dataset SHA256 does not match the expected checksum")
        existing = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        return {**existing, "source_url": url, "path": str(destination.resolve()), "sha256": digest, "bytes": size, "reused": True}

    context = ssl.create_default_context()
    opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()), HTTPSHandler(context=context))
    request_url = _drive_download_url(url)
    temporary = destination.with_name(destination.name + ".part")
    try:
        for attempt in range(3):
            request = Request(request_url, headers={"User-Agent": "A2F-OOD/0.1 dataset downloader"})
            with opener.open(request, timeout=timeout) as response:
                final_url = response.geturl()
                if urlparse(final_url).scheme != "https":
                    raise ValueError("Download redirected away from HTTPS")
                prefix = response.read(4096)
                content_type = response.headers.get("Content-Type", "")
                if _html_response(prefix, content_type):
                    document = prefix + response.read(1024 * 1024)
                    parser = _DownloadForm()
                    parser.feed(document.decode("utf-8", errors="replace"))
                    if parser.action and attempt < 2:
                        action = urljoin(final_url, html.unescape(parser.action))
                        if urlparse(action).scheme != "https" or urlparse(action).hostname not in {"drive.google.com", "drive.usercontent.google.com", "docs.google.com"}:
                            raise ValueError("Untrusted download confirmation form")
                        request_url = action + ("&" if "?" in action else "?") + urlencode(parser.fields)
                        continue
                    raise ValueError("Server returned an HTML page instead of dataset content; download may require access or quota renewal")
                digest = hashlib.sha256()
                size = 0
                with temporary.open("wb") as stream:
                    chunk = prefix
                    while chunk:
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        chunk = response.read(1024 * 1024)
                if size == 0:
                    raise ValueError("Server returned an empty file")
                content_length = response.headers.get("Content-Length")
                if content_length and size != int(content_length):
                    raise ValueError("Download size differs from the HTTP Content-Length")
                if expected_bytes is not None and size != expected_bytes:
                    raise ValueError(f"Downloaded file size mismatch: {size} != {expected_bytes}")
                checksum = digest.hexdigest()
                if expected_sha256 and checksum.lower() != expected_sha256.lower():
                    raise ValueError("Downloaded dataset SHA256 does not match the expected checksum")
                temporary.replace(destination)
                metadata = {
                    "source_url": url, "resolved_url": final_url,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "path": str(destination.resolve()), "sha256": checksum, "bytes": size,
                    "content_type": content_type, "expected_sha256": expected_sha256,
                    "expected_bytes": expected_bytes,
                }
                sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return metadata
        raise RuntimeError("Download confirmation attempts exhausted")
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_canonical_overlap(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report exact structural overlap and label conflicts without removing rows.

    Assay benchmarks may contain the same structure in several environments;
    preserving this fact is necessary for reproducing an official split.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return {"status": "unavailable", "reason": "RDKit is not installed"}
    canonical_by_smiles: dict[str, str | None] = {}
    molecule_info: dict[str, dict[str, Any]] = {}
    invalid = []
    for record in records:
        smiles = record["smiles"]
        if smiles not in canonical_by_smiles:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                canonical_by_smiles[smiles] = None
            else:
                for atom in molecule.GetAtoms():
                    atom.SetAtomMapNum(0)
                canonical_by_smiles[smiles] = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        canonical = canonical_by_smiles[smiles]
        if canonical is None:
            invalid.append(record["molecule_id"])
            continue
        info = molecule_info.setdefault(canonical, {"splits": set(), "labels": set(), "rows": 0})
        info["splits"].add(record["official_split"])
        info["labels"].add(record["label"])
        info["rows"] += 1
    overlaps: dict[str, dict[str, Any]] = {}
    for first_index, first in enumerate(OFFICIAL_SPLITS):
        for second in OFFICIAL_SPLITS[first_index + 1:]:
            common = [smiles for smiles, info in molecule_info.items() if {first, second} <= info["splits"]]
            if common:
                overlaps[f"{first}/{second}"] = {"count": len(common), "examples": sorted(common)[:10]}
    conflicts = [smiles for smiles, info in molecule_info.items() if len(info["labels"]) > 1]
    cross_split_conflicts = [smiles for smiles in conflicts if len(molecule_info[smiles]["splits"]) > 1]
    return {
        "status": "complete", "unique_canonical_molecules": len(molecule_info),
        "invalid_smiles_rows": len(invalid), "invalid_row_examples": invalid[:10],
        "overlap": overlaps, "conflicting_label_molecules": len(conflicts),
        "cross_split_conflicting_label_molecules": len(cross_split_conflicts),
        "conflicting_label_examples": [
            {"smiles": smiles, "labels": sorted(molecule_info[smiles]["labels"]),
             "official_splits": sorted(molecule_info[smiles]["splits"])}
            for smiles in sorted(conflicts)[:10]
        ],
        "policy": "Audit only; no rows, labels, or official split memberships were changed.",
    }


def convert_official_json(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str | None = None,
    label_key: str = "cls_label",
    group_key: str = "domain_id",
    smiles_key: str = "smiles",
    canonical_audit: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a DrugOOD/ADMEOOD official JSON without relabeling/resplitting.

    The primary CSV contains train/ood_val/ood_test as train/valid/test.
    An accompanying ``.all_splits.csv`` retains all five original subsets.
    Unsupported schemas fail rather than creating new split assignments.
    """
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError("The raw source file must not be overwritten")
    metadata_path = target.with_suffix(".metadata.json")
    all_path = target.with_name(target.stem + ".all_splits.csv")
    for output in (target, metadata_path, all_path):
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite explicitly: {output}")
    with source.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("split"), dict):
        raise ValueError("Expected official JSON object with a 'split' mapping")
    split_data = payload["split"]
    missing = set(TRAINING_SPLITS) - split_data.keys()
    if missing:
        raise ValueError(f"Official OOD subsets are missing: {sorted(missing)}; refusing to create replacement splits")
    unknown = set(split_data) - set(OFFICIAL_SPLITS)
    if unknown:
        raise ValueError(f"Unrecognized official split names: {sorted(unknown)}")
    dataset_name = dataset_name or source.stem
    all_records: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    split_counts, domain_counts, positives, negatives, overlap_sets = {}, {}, {}, {}, {}
    for split_name in OFFICIAL_SPLITS:
        if split_name not in split_data:
            continue
        records = split_data[split_name]
        if not isinstance(records, list):
            raise ValueError(f"Official split {split_name} must be a list of records")
        if split_name in TRAINING_SPLITS and not records:
            raise ValueError(f"Required official split {split_name} is empty")
        domains, smiles_seen = set(), set()
        positive_count, negative_count = 0, 0
        for index, original in enumerate(records):
            if not isinstance(original, dict):
                raise ValueError(f"Non-object record in {split_name}[{index}]")
            for field in (smiles_key, label_key, group_key):
                if field not in original or original[field] is None or str(original[field]).strip() == "":
                    raise ValueError(f"Missing official field {field!r} in {split_name}[{index}]")
            smiles = str(original[smiles_key]).strip()
            label = float(original[label_key])
            if not math.isfinite(label):
                raise ValueError(f"Nonfinite official label in {split_name}[{index}]")
            if label_key == "cls_label" and label not in (0, 1):
                raise ValueError(f"Expected official binary cls_label in {split_name}[{index}], got {label}")
            group = str(original[group_key])
            domains.add(group)
            smiles_seen.add(smiles)
            positive_count += label == 1
            negative_count += label == 0
            identifier = original.get("molecule_id", original.get("id", f"{dataset_name}:{split_name}:{index}"))
            record = {
                "molecule_id": identifier, "smiles": smiles,
                "label": int(label) if label.is_integer() else label,
                "group": group, "split": TRAINING_SPLITS.get(split_name, split_name),
                "official_split": split_name, "source_index": index,
                "assay_id": original.get("assay_id", ""), "domain_id": original.get("domain_id", ""),
            }
            all_records.append(record)
            if split_name in TRAINING_SPLITS:
                selected.append(record)
        split_counts[split_name] = len(records)
        domain_counts[split_name] = len(domains)
        positives[split_name], negatives[split_name] = positive_count, negative_count
        overlap_sets[split_name] = smiles_seen
    overlaps = {}
    for first_index, first in enumerate(OFFICIAL_SPLITS):
        for second in OFFICIAL_SPLITS[first_index + 1:]:
            if first in overlap_sets and second in overlap_sets:
                common = overlap_sets[first] & overlap_sets[second]
                if common:
                    overlaps[f"{first}/{second}"] = {"count": len(common), "examples": sorted(common)[:10]}
    source_sidecar = next((candidate for candidate in (
        source.with_name(source.name + ".download.json"),
        source.with_name(source.name + ".source.json"),
        source.with_name(source.name + ".provenance.json"),
    ) if candidate.exists()), None)
    metadata = {
        "dataset_name": dataset_name, "source_path": str(source.resolve()),
        "source_bytes": source.stat().st_size, "source_sha256": file_sha256(source),
        "source_download": json.loads(source_sidecar.read_text(encoding="utf-8")) if source_sidecar else None,
        "label_key": label_key, "group_key": group_key, "smiles_key": smiles_key,
        "split_mapping": TRAINING_SPLITS, "official_split_counts": split_counts,
        "official_domain_counts": domain_counts,
        "binary_positive_counts": positives if label_key == "cls_label" else None,
        "binary_negative_counts": negatives if label_key == "cls_label" else None,
        "literal_smiles_overlap": overlaps,
        "overlap_note": "Original official rows are preserved. This checks literal strings, not RDKit-normalized structures; downstream preprocessing must audit canonical overlaps.",
        "canonical_audit": audit_canonical_overlap(all_records) if canonical_audit else {"status": "disabled"},
        "original_cfg": payload.get("cfg", payload.get("config")),
        "original_statistics": payload.get("statistics"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_rows": len(selected), "all_rows": len(all_records),
        "csv_path": str(target.resolve()), "all_splits_csv": str(all_path.resolve()),
    }
    _write_csv(target, selected)
    _write_csv(all_path, all_records)
    metadata["csv_sha256"] = file_sha256(target)
    metadata["all_splits_csv_sha256"] = file_sha256(all_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
