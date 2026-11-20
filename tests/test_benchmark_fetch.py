"""Offline integrity checks for selective benchmark archive downloads."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "benchmark_fetch", Path(__file__).resolve().parents[1] / "scripts" / "fetch_benchmarks.py"
)
fetch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch)


class FakeResponse:
    def __init__(self, data, start, end, *, status=206):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
        self.content = data[start:end + 1]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass


def test_zipfile_extracts_only_selected_member_with_range_requests(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("chosen.json", json.dumps({"split": {"train": []}}))
        archive.writestr("other.dat", b"do not extract me")
    payload = buffer.getvalue()
    reader = fetch.HTTPRangeArchive("https://example.org/archive", len(payload))
    requests = []

    def get(url, *, headers, **kwargs):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        requests.append((start, end))
        return FakeResponse(payload, start, end)

    monkeypatch.setattr(reader.session, "get", get)
    with zipfile.ZipFile(reader) as archive:
        assert json.loads(archive.read("chosen.json")) == {"split": {"train": []}}
    assert len(requests) > 1
    assert reader.downloaded_bytes == sum(end - start + 1 for start, end in requests)
    reader.close()


def test_server_ignoring_range_is_rejected(monkeypatch):
    reader = fetch.HTTPRangeArchive("https://example.org/archive", 100)
    monkeypatch.setattr(reader.session, "get", lambda *args, **kwargs: FakeResponse(b"x" * 100, 0, 9, status=200))
    with pytest.raises(RuntimeError, match="byte range"):
        reader.read(10)
    assert reader.position == 0
    reader.close()


def test_corrupt_payload_cannot_replace_verified_file(tmp_path):
    path = tmp_path / "dataset.json"
    payload = json.dumps({"split": {"train": []}}).encode()
    specification = {"expected_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "url": "https://example.org/data"}
    fetch._write_json_payload(path, payload, specification)
    assert fetch.verified(path, specification)
    with pytest.raises(RuntimeError, match="SHA256"):
        fetch._write_json_payload(path, payload[:-1] + b"x", specification)
    assert path.read_bytes() == payload
