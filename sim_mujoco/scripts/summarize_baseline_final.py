"""Create the requested consolidated baseline-versus-final reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_and_validate(root: Path) -> tuple[dict, dict, list[int]]:
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    rows = summary["episodes"]
    seeds = [int(row["seed"]) for row in rows]
    if seeds != [int(seed) for seed in config["seeds"]]:
        raise RuntimeError(f"Summary seeds do not match evaluation config: {root}")
    if int(config["episodes"]) != len(rows) or int(summary["attempted"]) != len(rows):
        raise RuntimeError(f"Evaluation episode counts are inconsistent: {root}")
    labels = [row["label"] for row in rows]
    expected_counts = {
        "successes": labels.count("success"),
        "failures": labels.count("failure"),
        "invalid": labels.count("invalid"),
    }
    if any(int(summary[key]) != value for key, value in expected_counts.items()):
        raise RuntimeError(f"Evaluation outcome counts are inconsistent: {root}")
    return config, summary, seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline_config, baseline, baseline_seeds = _load_and_validate(args.baseline)
    final_config, final, final_seeds = _load_and_validate(args.final)
    if baseline_seeds != final_seeds:
        raise RuntimeError("Baseline and final evaluations did not use identical seeds")
    if len(baseline_seeds) < 20:
        raise RuntimeError(f"At least 20 identical evaluation seeds are required, got {len(baseline_seeds)}")
    baseline_shared = {key: value for key, value in baseline_config.items() if key != "policy_label"}
    final_shared = {key: value for key, value in final_config.items() if key != "policy_label"}
    if baseline_shared != final_shared:
        raise RuntimeError("Baseline and final evaluation configurations differ")
    report = {
        "passed": True,
        "identical_seeds": True,
        "seeds": baseline_seeds,
        "shared_settings": baseline_shared,
        "baseline": baseline,
        "final": final,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "POST_TRAINING_EVALUATION.json"
    markdown_path = args.output_dir / "POST_TRAINING_EVALUATION.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# Baseline versus Final MuJoCo Evaluation",
                "",
                f"- Identical fixed seeds: {len(baseline_seeds)}",
                (
                    f"- Baseline: {baseline['successes']} success, {baseline['failures']} failure, "
                    f"{baseline['invalid']} invalid; all-attempt rate {baseline['success_rate_all']:.3f}"
                ),
                (
                    f"- Final: {final['successes']} success, {final['failures']} failure, "
                    f"{final['invalid']} invalid; all-attempt rate {final['success_rate_all']:.3f}"
                ),
                "- The JSON report contains every per-seed termination, latency, first-action, clipping, and video field.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"passed": True, "json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
