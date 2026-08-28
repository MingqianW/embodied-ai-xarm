"""Build a deterministic blinded human-video-review manifest from formal results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sim.human_review import build_manifest  # noqa: E402
from evaluation.sim.human_review import write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--review-seed", type=int, default=20260808)
    parser.add_argument("--mode", choices=("full", "representative"), default="full")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    output_root = (
        args.output_root
        or evaluation_root / "human_review" / f"{args.mode}_seed_{args.review_seed}"
    ).expanduser().resolve()
    private_manifest, reviewer_manifest, coverage = build_manifest(
        evaluation_root=evaluation_root, review_seed=args.review_seed, mode=args.mode
    )
    report = {
        "evaluation_root": str(evaluation_root),
        "output_root": str(output_root),
        "mode": args.mode,
        "review_seed": args.review_seed,
        "reviewable_item_count": len(private_manifest["items"]),
        "coverage": coverage,
        "reviewer_manifest_fields": sorted(reviewer_manifest["items"][0])
        if reviewer_manifest["items"]
        else [],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.dry_run:
        return
    write_manifest(
        output_root=output_root,
        private_manifest=private_manifest,
        reviewer_manifest=reviewer_manifest,
        coverage=coverage,
    )
    print(f"Wrote blinded human-review manifest: {output_root}")


if __name__ == "__main__":
    main()
