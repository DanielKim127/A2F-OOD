"""Create reproducible ablation configs; use --run to execute them sequentially."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from a2f_ood.config import load_config, resolve_path


def variants(config: dict) -> dict[str, dict]:
    result = {"full": copy.deepcopy(config)}
    for mode in ("graph", "smiles", "concat"):
        item = copy.deepcopy(config)
        item["fusion"]["mode"] = mode
        for key in item["loss"]:
            if key.startswith("lambda_"):
                item["loss"][key] = 0.0
        item["model"]["use_cross_modal_decoder"] = False
        item["loss"]["use_group_dro"] = False
        result[mode] = item
    for name, section, key, value in (
        ("no_atom_alignment", "fusion", "atom_alignment", False),
        ("no_cross_attention", "fusion", "bidirectional_cross_attention", False),
        ("no_reliability_gate", "fusion", "reliability_gate", False),
        ("no_xmodal", "loss", "lambda_cross_modal", 0.0),
        ("no_serial_consistency", "loss", "lambda_serialization_consistency", 0.0),
        ("no_group_dro", "loss", "use_group_dro", False),
    ):
        item = copy.deepcopy(config)
        item[section][key] = value
        result[name] = item
    item = copy.deepcopy(config)
    item["model"]["use_cross_modal_decoder"] = False
    item["loss"]["lambda_generation"] = 0.0
    result["no_decoder"] = item
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    available = variants(base)
    selected = args.variants or list(available)
    if set(selected) - available.keys():
        parser.error(f"Unknown variants; choose from {list(available)}")
    for name in selected:
        for seed in args.seeds:
            item = copy.deepcopy(available[name])
            item["experiment"].update(name=f"{name}_{seed}", seed=seed,
                                      output_dir=str(output_dir / f"{name}_{seed}"))
            item["data"]["csv_path"] = str(resolve_path(base, item["data"]["csv_path"]))
            item = {k: v for k, v in item.items() if not k.startswith("_")}
            path = output_dir / f"{name}_{seed}.yaml"
            path.write_text(yaml.safe_dump(item, allow_unicode=True, sort_keys=False), encoding="utf-8")
            command = [sys.executable, "-m", "a2f_ood.train", "--config", str(path)]
            runs.append({"name": name, "seed": seed, "command": command})
            if args.run:
                subprocess.run(command, check=True)
    (output_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"{'Executed' if args.run else 'Prepared'} {len(runs)} runs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
