"""Unblind a completed human review and report human/automated agreement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sim.human_review import load_decisions  # noqa: E402
from evaluation.sim.human_review import markdown_summary  # noqa: E402
from evaluation.sim.human_review import summarize_review_rows  # noqa: E402
from evaluation.sim.human_review import unblind_manifest  # noqa: E402
from evaluation.sim.human_review import write_unblinded_csv  # noqa: E402
from evaluation.sim.outputs import read_json  # noqa: E402
from evaluation.sim.outputs import write_json  # noqa: E402


def _write_paired_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "task",
        "seed",
        "A_human_label",
        "B_human_label",
        "C_human_label",
        "A_review_id",
        "B_review_id",
        "C_review_id",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    review_root = args.review_root.expanduser().resolve()
    output_root = (args.output_root or review_root).expanduser().resolve()
    private_manifest = read_json(review_root / "manifest_private.json")
    decisions = load_decisions(review_root / "human_review.csv")
    rows = unblind_manifest(
        private_manifest=private_manifest, decisions=decisions, allow_incomplete=args.allow_incomplete
    )
    summary = summarize_review_rows(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    write_unblinded_csv(output_root / "human_review_unblinded.csv", rows)
    write_json(output_root / "human_review_summary.json", summary)
    (output_root / "human_review_summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    _write_paired_csv(output_root / "human_review_paired.csv", summary["paired_model_analysis"]["task_seed_rows"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
