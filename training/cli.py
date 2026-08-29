"""Inspect, preflight, and explicitly delegate supported runs to OpenPI."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from training.configs.experiments import EXPERIMENTS, get_experiment
from training.openpi.adapter import DEFAULT_OPENPI_ROOT, OpenPIUnavailable, build_openpi_train_config
from training.validation.preflight import preflight


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _dataset_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Dataset path must be DATASET_ID=PATH, got {value!r}")
        dataset_id, path = value.split("=", 1)
        result[dataset_id] = Path(path).resolve()
    return result


def _load_train_module(openpi_root: Path) -> Any:
    script = openpi_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("project_openpi_train", script)
    if spec is None or spec.loader is None:
        raise OpenPIUnavailable(f"Cannot load OpenPI training entrypoint: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list canonical experiment names")
    show = subparsers.add_parser("show", help="print one fully resolved project config")
    show.add_argument("config")
    check = subparsers.add_parser("preflight", help="run non-training validation")
    check.add_argument("config")
    check.add_argument("--dataset-path", action="append", default=[], metavar="ID=PATH")
    check.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    check.add_argument("--skip-openpi", action="store_true")
    train = subparsers.add_parser("train", help="delegate a supported config to OpenPI")
    train.add_argument("config")
    train.add_argument("--exp-name", required=True)
    train.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    train.add_argument("--assets-base-dir", default="./assets")
    train.add_argument("--checkpoint-base-dir", default="./checkpoints")
    train.add_argument("--execute", action="store_true", help="required acknowledgement; launches training")
    args = parser.parse_args(argv)
    if args.command == "list":
        for name in sorted(EXPERIMENTS):
            print(name)
        return 0
    config = get_experiment(args.config)
    if args.command == "show":
        print(_json(config.as_dict()), end="")
        return 0
    if args.command == "preflight":
        try:
            paths = _dataset_paths(args.dataset_path)
        except ValueError as exc:
            parser.error(str(exc))
        report = preflight(
            config,
            dataset_paths=paths,
            openpi_root=args.openpi_root,
            check_openpi=not args.skip_openpi,
        )
        print(_json(report.as_dict()), end="")
        return 0 if report.passed else 2
    if not args.execute:
        parser.error("train requires --execute; use preflight first")
    openpi_config = build_openpi_train_config(
        config,
        exp_name=args.exp_name,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        openpi_root=args.openpi_root,
    )
    metadata_dir = Path(args.checkpoint_base_dir) / config.name / args.exp_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "project_resolved_config.json").write_text(_json(config.as_dict()), encoding="utf-8")
    _load_train_module(args.openpi_root.resolve()).main(openpi_config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
