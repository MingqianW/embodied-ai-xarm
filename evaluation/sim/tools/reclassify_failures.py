"""Create schema-v2 failure diagnoses from existing formal evaluation results.

Historical ``result.json`` files are never modified.  Upgraded copies,
summaries, and video indexes are written below a separate derived-analysis
root so the original evaluation evidence remains immutable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sim.outputs import read_json  # noqa: E402
from evaluation.sim.outputs import result_fingerprint  # noqa: E402
from evaluation.sim.outputs import upgrade_result_with_failure_diagnosis  # noqa: E402
from evaluation.sim.outputs import write_json  # noqa: E402
from evaluation.sim.summary import write_comparison  # noqa: E402


def _source_result_paths(root: Path) -> list[Path]:
    models_root = root / "models"
    search_root = models_root if models_root.is_dir() else root
    paths = sorted(search_root.glob("**/tasks/*/seed_*/result.json"))
    if not paths:
        raise FileNotFoundError(f"No formal result.json files found below: {root}")
    return paths


def _derived_result_path(*, source: Path, root: Path, output_root: Path, model_id: str) -> Path:
    try:
        relative = source.relative_to(root / "models" / model_id)
    except ValueError:
        relative = source.relative_to(root)
        if relative.parts[0] == "tasks":
            return output_root / "models" / model_id / relative
    return output_root / "models" / model_id / relative


def _key_evidence(result: dict[str, object]) -> dict[str, object]:
    metrics = dict(result["metrics"])
    if "max_lift_m" in metrics:
        return {
            "max_lift_m": metrics.get("max_lift_m"),
            "final_lift_m": metrics.get("final_lift_m", metrics.get("lift_height_m")),
        }
    return {
        "release_confirmed": metrics.get("release_confirmed"),
        "containment_confirmed": metrics.get("containment_confirmed"),
        "height_confirmed": metrics.get("height_confirmed"),
        "stability_confirmed": metrics.get("stability_confirmed"),
        "pepper_linear_speed_mps": metrics.get("pepper_linear_speed_mps"),
        "pepper_angular_speed_radps": metrics.get("pepper_angular_speed_radps"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Evaluation root or one model root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Derived analysis root; defaults to <root>/derived/failure_diagnosis_v1.",
    )
    parser.add_argument(
        "--overwrite-derived",
        action="store_true",
        help="Allow replacing a non-empty derived analysis root, never source results.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output_root = (args.output_root or root / "derived" / "failure_diagnosis_v1").expanduser().resolve()
    paths = _source_result_paths(root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite_derived:
        raise FileExistsError(
            f"Derived output already exists; choose a new --output-root or pass --overwrite-derived: {output_root}"
        )

    rows = []
    for source in paths:
        original = read_json(source)
        upgraded = upgrade_result_with_failure_diagnosis(original)
        episode = upgraded["episode"]
        rows.append(
            {
                "task": episode["task"],
                "seed": episode["seed"],
                "success": episode["success"],
                "valid": episode["valid"],
                "failure_category": episode["failure_category"],
                "failure_stage": episode["failure_stage"],
                "key_evidence": _key_evidence(upgraded),
            }
        )
        if args.dry_run:
            continue
        model_id = str(upgraded["model"]["model_id"])
        destination = _derived_result_path(
            source=source, root=root, output_root=output_root, model_id=model_id
        )
        write_json(destination, upgraded)
        write_json(
            destination.with_name("source_result.json"),
            {
                "source_result_path": str(source),
                "source_result_fingerprint": result_fingerprint(original),
                "derived_schema_version": upgraded["schema_version"],
            },
        )
    if not args.dry_run:
        write_json(
            output_root / "REANALYSIS.json",
            {
                "analysis_version": "xarm-formal-failure-reanalysis-v1",
                "source_root": str(root),
                "source_result_count": len(paths),
                "write_mode": "derived_copy_only",
            },
        )
        write_comparison(output_root)
    print(json.dumps(rows, indent=2, sort_keys=True))
    if not args.dry_run:
        print(f"Wrote derived diagnosis analysis: {output_root}")


if __name__ == "__main__":
    main()
