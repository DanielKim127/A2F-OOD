"""Fetch the ten pinned official DrugOOD/ADMEOOD raw datasets.

DrugOOD files are selectively extracted from the official 1.16 GB ZIP using
HTTP byte ranges. Only the central directory and selected members are fetched.
The server must honor Range; falling back to a full archive download is never
implicit. Downloaded JSONs are checked against pinned sizes and SHA256 hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "benchmark_downloads.json"


class HTTPRangeArchive:
    """Minimal seekable file interface for zipfile backed by strict HTTP206."""

    def __init__(self, url: str, size: int, *, timeout: int = 60) -> None:
        if not url.startswith("https://"):
            raise ValueError("Archive URLs must use HTTPS")
        self.url, self.size, self.timeout = url, int(size), timeout
        self.position = 0
        self.session = requests.Session()
        self.downloaded_bytes = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError("Invalid seek origin")
        if position < 0:
            raise ValueError("Cannot seek before the archive start")
        self.position = position
        return position

    def read(self, count: int = -1) -> bytes:
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        start, end = self.position, self.position + count - 1
        last_error = None
        for _ in range(3):
            try:
                with self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                    timeout=self.timeout, stream=True,
                ) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{end}/{self.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Server did not honor the requested archive byte range")
                    payload = response.content
                if len(payload) != count:
                    raise RuntimeError(f"Incomplete byte range: expected {count}, received {len(payload)}")
                self.position += count
                self.downloaded_bytes += count
                return payload
            except (requests.RequestException, RuntimeError) as error:
                last_error = error
        raise RuntimeError(f"Failed to read archive range {start}-{end}: {last_error}") from last_error

    def close(self) -> None:
        self.session.close()


def verified(path: Path, specification: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == specification["expected_bytes"]
        and hashlib.sha256(path.read_bytes()).hexdigest() == specification["sha256"]
    )


def _write_json_payload(path: Path, payload: bytes, specification: dict[str, Any]) -> None:
    if len(payload) != specification["expected_bytes"]:
        raise RuntimeError(f"Size mismatch for {path.name}")
    if hashlib.sha256(payload).hexdigest() != specification["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {path.name}; official source may have changed")
    document = json.loads(payload)
    if not isinstance(document, dict) or "split" not in document:
        raise RuntimeError(f"Downloaded file is not a supported benchmark JSON: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    metadata = {
        **specification, "bytes": len(payload),
        "source_url": specification.get("url", specification.get("archive_url")),
        "download_method": "selective HTTP Range ZIP extraction" if "archive_member" in specification else "public HTTPS",
    }
    path.with_suffix(path.suffix + ".source.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def fetch_selected(specifications: list[dict[str, Any]], root: Path, *, overwrite: bool = False) -> None:
    root = root.resolve()
    archives: dict[str, tuple[HTTPRangeArchive, zipfile.ZipFile]] = {}
    try:
        for specification in specifications:
            path = (root / specification["path"]).resolve()
            if root not in path.parents:
                raise ValueError("Manifest destination escapes the selected raw-data root")
            if verified(path, specification):
                print(f"Verified existing {path.name}", flush=True)
                continue
            if path.exists() and not overwrite:
                raise RuntimeError(f"Existing {path} fails integrity checks; pass --overwrite to replace it")
            if "archive_member" in specification:
                archive_url = specification["archive_url"]
                if archive_url not in archives:
                    remote = HTTPRangeArchive(archive_url, specification["archive_bytes"])
                    archives[archive_url] = remote, zipfile.ZipFile(remote)
                remote, archive = archives[archive_url]
                info = archive.getinfo(specification["archive_member"])
                if info.file_size != specification["expected_bytes"]:
                    raise RuntimeError("Archive member size changed")
                expected_crc = specification.get("archive_member_crc32")
                if expected_crc is not None and f"{info.CRC:08x}" != expected_crc:
                    raise RuntimeError("Archive member CRC changed")
                payload = archive.read(info)  # zipfile independently validates CRC.
            else:
                url = specification["url"]
                if not url.startswith("https://"):
                    raise ValueError("Dataset URLs must use HTTPS")
                with requests.get(url, timeout=120, stream=True) as response:
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise RuntimeError("Received an HTML landing page instead of a dataset")
                    payload = response.content
            _write_json_payload(path, payload, specification)
            print(f"Downloaded {path.name} ({len(payload):,} bytes)", flush=True)
        for remote, _ in archives.values():
            print(f"DrugOOD archive transfer: {remote.downloaded_bytes:,} bytes", flush=True)
    finally:
        for remote, archive in archives.values():
            archive.close()
            remote.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--dataset", action="append", help="Pinned dataset ID; repeat to select multiple; default all ten")
    parser.add_argument("--list", action="store_true", help="List names and raw sizes without downloading")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    specifications = manifest["files"]
    if args.dataset:
        selected = set(args.dataset)
        unknown = selected - {item["dataset"] for item in specifications}
        if unknown:
            parser.error(f"Unknown dataset IDs: {sorted(unknown)}")
        specifications = [item for item in specifications if item["dataset"] in selected]
    if args.list:
        for item in specifications:
            print(f"{item['dataset']}: {item['path']} ({item['expected_bytes']:,} bytes)")
        return 0
    fetch_selected(specifications, args.root, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
