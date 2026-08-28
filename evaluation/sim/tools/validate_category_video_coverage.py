"""Validate that every observed formal outcome category has a representative video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sim.representative_videos import validate_category_video_coverage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Also print the full machine-readable report.")
    args = parser.parse_args()
    report = validate_category_video_coverage(args.evaluation_root)
    for row in report["model_task_reports"]:
        print(f"Model {row['model']} / {row['task']}")
        videos = set(row["categories_with_videos"])
        for category in row["observed_categories"]:
            print(f"  {category:<50} {'PASS' if category in videos else 'FAIL'}")
    status = "CATEGORY VIDEO COVERAGE COMPLETE" if report["coverage_complete"] else "CATEGORY VIDEO COVERAGE INCOMPLETE"
    print(status)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if not report["coverage_complete"]:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
