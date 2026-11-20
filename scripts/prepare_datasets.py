"""Convert official DrugOOD/ADMEOOD JSONs into project CSVs.

Example: python scripts/prepare_datasets.py --raw-dir data/raw --output-dir data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from a2f_ood.datasets import convert_official_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="One official JSON source")
    parser.add_argument("--output", type=Path, help="Training CSV path when --input is supplied")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--label-key", default="cls_label")
    parser.add_argument("--group-key", default="domain_id")
    parser.add_argument("--smiles-key", default="smiles")
    parser.add_argument("--no-canonical-audit", action="store_true", help="Skip the RDKit audit; never changes or drops official rows")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input:
        if args.output is None:
            parser.error("--input requires --output")
        inputs = [(args.input, args.output)]
    else:
        inputs = [(path, args.output_dir / path.relative_to(args.raw_dir).with_suffix(".csv"))
                  for path in sorted(args.raw_dir.rglob("*.json"))
                  if not path.name.endswith((".download.json", ".metadata.json", ".manifest.json", ".provenance.json", ".source.json"))]
    if not inputs:
        parser.error("No official raw JSON files found")
    for source, target in inputs:
        metadata = convert_official_json(
            source, target, dataset_name=args.dataset_name,
            label_key=args.label_key, group_key=args.group_key,
            smiles_key=args.smiles_key, canonical_audit=not args.no_canonical_audit, overwrite=args.overwrite,
        )
        print(json.dumps({"source": str(source), "output": str(target),
                          "selected_rows": metadata["selected_rows"],
                          "official_split_counts": metadata["official_split_counts"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
