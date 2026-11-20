"""Official labels/splits remain unchanged and downloads never accept HTML."""
import csv
import hashlib
import io
import json

import pytest

from a2f_ood.datasets import convert_official_json, download_file, file_sha256


@pytest.fixture
def official_payload():
    return {
        "cfg": {"domain": {"domain_name": "assay"}},
        "statistics": {"thr_for_cls": 5},
        "split": {
            "train": [{"smiles": "CCO", "cls_label": 0, "domain_id": 0, "assay_id": 123, "reg_label": 7.04}],
            "iid_val": [{"smiles": "CCN", "cls_label": 1, "domain_id": 0, "assay_id": 123, "reg_label": 2.5}],
            "iid_test": [{"smiles": "CCC", "cls_label": 0, "domain_id": 0, "assay_id": 123, "reg_label": 3.0}],
            "ood_val": [{"smiles": "C1CC1", "cls_label": 1, "domain_id": 1, "assay_id": 456, "reg_label": 8.0}],
            "ood_test": [{"smiles": "OCC", "cls_label": 1, "domain_id": 2, "assay_id": 789, "reg_label": 4.0}],
        },
    }


def test_preserves_official_splits_labels_domains_and_audits_conflicts(tmp_path, official_payload):
    raw = tmp_path / "official.json"
    raw.write_text(json.dumps(official_payload), encoding="utf-8")
    raw.with_name(raw.name + ".source.json").write_text(json.dumps({"source_url": "https://example.org/official.json"}), encoding="utf-8")
    original_hash = file_sha256(raw)
    target = tmp_path / "data.csv"
    metadata = convert_official_json(raw, target, dataset_name="test_benchmark")
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    assert len(rows) == 3
    assert [row["split"] for row in rows] == ["train", "valid", "test"]
    assert [row["official_split"] for row in rows] == ["train", "ood_val", "ood_test"]
    assert [row["group"] for row in rows] == ["0", "1", "2"]
    assert rows[0]["label"] == "0"  # Never recompute from reg_label or threshold.
    assert metadata["source_sha256"] == original_hash == file_sha256(raw)
    assert metadata["source_download"]["source_url"] == "https://example.org/official.json"
    assert metadata["original_cfg"] == official_payload["cfg"]
    audit = metadata["canonical_audit"]
    assert audit["status"] == "complete"
    assert audit["overlap"]["train/ood_test"]["count"] == 1
    assert audit["cross_split_conflicting_label_molecules"] == 1
    all_rows = list(csv.DictReader(target.with_name("data.all_splits.csv").open(encoding="utf-8")))
    assert len(all_rows) == 5
    assert {row["official_split"] for row in all_rows} == set(official_payload["split"])
    assert json.loads(target.with_suffix(".metadata.json").read_text(encoding="utf-8"))["csv_sha256"] == file_sha256(target)
    with pytest.raises(FileExistsError):
        convert_official_json(raw, target)


@pytest.mark.parametrize("mutation,match", [
    (lambda data: data["split"].pop("ood_test"), "missing"),
    (lambda data: data["split"]["train"][0].pop("domain_id"), "domain_id"),
    (lambda data: data["split"]["train"][0].update(cls_label=2), "binary"),
    (lambda data: data["split"].update(ood_val=[]), "empty"),
])
def test_refuses_to_invent_missing_official_fields_or_splits(tmp_path, official_payload, mutation, match):
    mutation(official_payload)
    raw = tmp_path / "official.json"
    raw.write_text(json.dumps(official_payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        convert_official_json(raw, tmp_path / "data.csv")
    assert not (tmp_path / "data.csv").exists()


def test_explicit_regression_conversion_retains_continuous_labels(tmp_path, official_payload):
    raw = tmp_path / "official.json"
    raw.write_text(json.dumps(official_payload), encoding="utf-8")
    target = tmp_path / "regression.csv"
    convert_official_json(raw, target, label_key="reg_label", canonical_audit=False)
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    assert rows[0]["label"] == "7.04"


class FakeResponse(io.BytesIO):
    def __init__(self, body, *, content_type="application/json", url="https://example.org/data.json"):
        super().__init__(body)
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        self.url = url

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, body, content_type="application/json"):
        self.body, self.content_type = body, content_type

    def open(self, request, timeout):
        return FakeResponse(self.body, content_type=self.content_type)


def test_downloader_checks_bytes_hash_and_writes_provenance(tmp_path, monkeypatch):
    body = b'{"split":{}}'
    monkeypatch.setattr("a2f_ood.datasets.build_opener", lambda *args: FakeOpener(body))
    target = tmp_path / "data.json"
    digest = hashlib.sha256(body).hexdigest()
    metadata = download_file("https://example.org/data.json", target, expected_bytes=len(body), expected_sha256=digest)
    assert metadata["sha256"] == digest
    assert target.read_bytes() == body
    assert target.with_name("data.json.download.json").exists()
    assert download_file("https://example.org/data.json", target, expected_sha256=digest)["reused"]


@pytest.mark.parametrize("body,kind,kwargs,match", [
    (b"<html>quota exceeded</html>", "text/html", {}, "HTML"),
    (b"{}", "application/json", {"expected_bytes": 99}, "size mismatch"),
    (b"{}", "application/json", {"expected_sha256": "0" * 64}, "SHA256"),
])
def test_downloader_rejects_landing_pages_and_incomplete_content(tmp_path, monkeypatch, body, kind, kwargs, match):
    monkeypatch.setattr("a2f_ood.datasets.build_opener", lambda *args: FakeOpener(body, kind))
    target = tmp_path / "data.json"
    with pytest.raises(ValueError, match=match):
        download_file("https://example.org/data.json", target, **kwargs)
    assert not target.exists()
    assert not target.with_name("data.json.part").exists()


def test_downloader_requires_tls(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        download_file("http://example.org/data.json", tmp_path / "data.json")
