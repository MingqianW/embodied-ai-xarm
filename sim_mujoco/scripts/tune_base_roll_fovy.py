from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from camera_calibration_lib import (
    CALIBRATION_ROOT,
    CONFIG_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    CalibrationRenderer,
    aggregate_metrics,
    geometric_loss,
    load_config,
    load_rgb,
    project_path,
    read_json,
    render_comparisons,
    write_config,
    write_json,
)


def values(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    return np.linspace(start, stop, count + 1)


def score_grid(
    renderer: CalibrationRenderer,
    samples: list[dict],
    real_images: dict[str, np.ndarray],
    anchor: dict,
    rolls: np.ndarray,
    fovys: np.ndarray,
    stage: str,
) -> list[dict]:
    results = []
    for roll in rolls:
        for fovy in fovys:
            parameters = copy.deepcopy(anchor)
            parameters["roll_deg"] = float(roll)
            parameters["fovy_deg"] = float(fovy)
            losses = [
                geometric_loss(
                    real_images[sample["sample_id"]],
                    renderer.render(sample, "base_camera", parameters),
                    "base_camera",
                )
                for sample in samples
            ]
            results.append(
                {
                    "stage": stage,
                    "roll_deg": float(roll),
                    "fovy_deg": float(fovy),
                    "mean_loss": float(np.mean(losses)),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--coarse-roll-step", type=float, default=2.0)
    parser.add_argument("--coarse-fovy-step", type=float, default=2.0)
    parser.add_argument("--fine-step", type=float, default=0.25)
    parser.add_argument("--anchor-config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--final-roll", type=float)
    parser.add_argument("--final-fovy", type=float)
    args = parser.parse_args()

    manifest = read_json(MANIFEST_PATH)
    calibration_samples = [sample for sample in manifest["samples"] if sample["split"] == "calibration"]
    if (args.final_roll is None) != (args.final_fovy is None):
        raise SystemExit("--final-roll and --final-fovy must be provided together")
    before_config = load_config(args.anchor_config)
    anchor = copy.deepcopy(before_config["base_camera"])
    fixed_base = {key: copy.deepcopy(value) for key, value in anchor.items() if key not in {"roll_deg", "fovy_deg"}}
    wrist_anchor = copy.deepcopy(before_config["wrist_camera"])
    real_images = {
        sample["sample_id"]: load_rgb(project_path(sample["base_image"]))
        for sample in calibration_samples
    }

    renderer = CalibrationRenderer(args.width, args.height)
    try:
        coarse = score_grid(
            renderer,
            calibration_samples,
            real_images,
            anchor,
            values(-16.0, 16.0, args.coarse_roll_step),
            values(34.0, 66.0, args.coarse_fovy_step),
            "coarse",
        )
        coarse_best = min(coarse, key=lambda item: item["mean_loss"])
        fine = score_grid(
            renderer,
            calibration_samples,
            real_images,
            anchor,
            values(coarse_best["roll_deg"] - 2.0, coarse_best["roll_deg"] + 2.0, args.fine_step),
            values(coarse_best["fovy_deg"] - 2.0, coarse_best["fovy_deg"] + 2.0, args.fine_step),
            "fine",
        )
    finally:
        renderer.close()

    all_results = coarse + fine
    numeric_best = min(all_results, key=lambda item: item["mean_loss"])
    selected = numeric_best
    if args.final_roll is not None:
        selected_parameters = copy.deepcopy(anchor)
        selected_parameters["roll_deg"] = args.final_roll
        selected_parameters["fovy_deg"] = args.final_fovy
        selected_renderer = CalibrationRenderer(args.width, args.height)
        try:
            selected_losses = [
                geometric_loss(
                    real_images[sample["sample_id"]],
                    selected_renderer.render(sample, "base_camera", selected_parameters),
                    "base_camera",
                )
                for sample in calibration_samples
            ]
        finally:
            selected_renderer.close()
        selected = {
            "stage": "visual_selection",
            "roll_deg": args.final_roll,
            "fovy_deg": args.final_fovy,
            "mean_loss": float(np.mean(selected_losses)),
        }
    tuned_config = copy.deepcopy(before_config)
    tuned_config["base_camera"]["roll_deg"] = selected["roll_deg"]
    tuned_config["base_camera"]["fovy_deg"] = selected["fovy_deg"]
    tuned_config["metadata"] = {
        **before_config.get("metadata", {}),
        "status": "roll_fovy_tuned_with_fixed_extrinsics",
        "roll_fovy_tuned_utc": datetime.now(timezone.utc).isoformat(),
        "roll_fovy_anchor": {
            "roll_deg": anchor["roll_deg"],
            "fovy_deg": anchor["fovy_deg"],
        },
        "roll_fovy_calibration_sample_count": len(calibration_samples),
        "roll_fovy_selection": "visual_balance" if args.final_roll is not None else "minimum_edge_loss",
    }

    tuned_fixed_base = {
        key: copy.deepcopy(value)
        for key, value in tuned_config["base_camera"].items()
        if key not in {"roll_deg", "fovy_deg"}
    }
    if tuned_fixed_base != fixed_base:
        raise RuntimeError("Base-camera settings other than roll/FOV changed")
    if tuned_config["wrist_camera"] != wrist_anchor:
        raise RuntimeError("Wrist-camera settings changed")

    write_config(tuned_config, CONFIG_PATH)
    write_config(tuned_config, CALIBRATION_ROOT / "camera_calibration.yaml")
    comparison = render_comparisons(manifest["samples"], before_config, tuned_config)
    metrics = {
        "metric_direction": "lower is better",
        "comparison": "fixed camera extrinsics; roll_deg and fovy_deg only",
        "fixed_base_camera_settings": fixed_base,
        "wrist_camera_unchanged": tuned_config["wrist_camera"] == wrist_anchor,
        "anchor": {"roll_deg": anchor["roll_deg"], "fovy_deg": anchor["fovy_deg"]},
        "numeric_best": numeric_best,
        "selected": selected,
        "selection_rationale": (
            "level table edge and closer arm scale, subject to held-out improvement"
            if args.final_roll is not None
            else "minimum calibration edge loss"
        ),
        "aggregate": aggregate_metrics(comparison["samples"]),
        "top_candidates": sorted(all_results, key=lambda item: item["mean_loss"])[:20],
        "evaluated_candidates": len(all_results),
    }
    write_json(CALIBRATION_ROOT / "roll_fovy_search.json", metrics)
    write_json(CALIBRATION_ROOT / "logs" / "roll_fovy_grid.json", all_results)
    write_json(METRICS_PATH, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
