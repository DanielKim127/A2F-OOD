"""Download HTTPS/Google Drive dataset files with integrity/provenance records.

Pass --url for one file or --manifest for a JSON list of download specifications.
This command rejects HTML landing pages and never disables TLS verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from a2f_ood.datasets import download_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--manifest", type=Path, help="JSON list with url/path/expected_bytes/sha256 fields")
    parser.add_argument("--output", type=Path, help="Output file for a single --url")
    parser.add_argument("--root", type=Path, default=Path("data/raw"), help="Root for relative manifest paths")
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--expected-bytes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.url:
        if args.output is None:
            parser.error("--url requires --output")
        specs = [{"url": args.url, "destination": args.output, "sha256": args.sha256, "expected_bytes": args.expected_bytes}]
    else:
        entries = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        if isinstance(entries, dict):
            entries = entries["files"]
        specs = [{**entry, "destination": args.root / entry["path"]} for entry in entries]
    for spec in specs:
        metadata = download_file(spec["url"], spec["destination"], expected_sha256=spec.get("sha256"),
                                 expected_bytes=spec.get("expected_bytes"), overwrite=args.overwrite)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
