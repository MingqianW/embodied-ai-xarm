"""Data-only validation of formal A/B/C checkpoint specifications."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.formal_evaluation.models import load_model_spec  # noqa: E402
from sim_mujoco.formal_evaluation.models import validate_abc_comparison_specs  # noqa: E402
from sim_mujoco.formal_evaluation.models import validate_training_config_asset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", action="append", type=Path, required=True)
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=Path(os.environ.get("OPENPI_ROOT", "/u/mw89/repos/openpi")),
    )
    args = parser.parse_args()
    specs = tuple(load_model_spec(path) for path in args.model_spec)
    for spec in specs:
        validate_training_config_asset(spec, openpi_root=args.openpi_root)
    print(json.dumps(validate_abc_comparison_specs(specs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
