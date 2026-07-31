from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_isaac.asset_preparation import (
    load_asset_paths,
    prepare_assets,
    write_asset_report,
)


DEFAULT_CONFIG = PROJECT_ROOT / "sim_isaac" / "config" / "asset_import.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and prepare the xArm6 Isaac asset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--expand-xacro", action="store_true")
    parser.add_argument("--import-usd", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()

    if args.validate_only and (args.expand_xacro or args.import_usd):
        parser.error("--validate-only cannot be combined with expansion or import")
    paths = load_asset_paths(args.config, PROJECT_ROOT)
    report = prepare_assets(
        paths,
        expand=args.expand_xacro or args.import_usd,
        import_usd=args.import_usd,
        headless=args.headless,
    )
    if not args.no_write_report:
        write_asset_report(paths.validation_report, report)
    print(json.dumps(report, indent=2))
    if not report["source_validation"]["valid"]:
        return 2
    if args.expand_xacro and not report["urdf_validation"].get("valid"):
        return 3
    if args.import_usd and not report["usd"]["generated"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

