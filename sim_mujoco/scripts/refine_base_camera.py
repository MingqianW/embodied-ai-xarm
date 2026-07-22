from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

import numpy as np

from camera_calibration_lib import (
    CALIBRATION_ROOT,
    CONFIG_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    CalibrationRenderer,
    aggregate_metrics,
    load_config,
    optimize_camera,
    parameter_vector,
    read_json,
    render_comparisons,
    write_config,
    write_json,
)


def local_bounds(anchor: dict, position_delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = parameter_vector(anchor)
    lower = vector.copy()
    upper = vector.copy()
    lower[:3] -= position_delta
    upper[:3] += position_delta
    lower[3:6] -= np.array([0.20, 0.20, 0.20])
    upper[3:6] += np.array([0.20, 0.20, 0.20])
    lower[5] = max(0.0, lower[5])
    upper[5] = min(0.35, upper[5])
    lower[6] = max(-30.0, vector[6] - 20.0)
    upper[6] = min(30.0, vector[6] + 20.0)
    lower[7] = max(35.0, vector[7] - 15.0)
    upper[7] = min(70.0, vector[7] + 15.0)

    heuristic = vector.copy()
    heuristic[4] = max(lower[4], vector[4] - 0.10)
    heuristic[5] = min(upper[5], vector[5] + 0.15)
    heuristic[6] = 0.0
    return lower, upper, heuristic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=180)
    parser.add_argument("--position-delta", type=float, default=0.01)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--optimization-width", type=int, default=160)
    parser.add_argument("--optimization-height", type=int, default=120)
    args = parser.parse_args()
    if not 0.0 <= args.position_delta <= 0.02:
        raise SystemExit("--position-delta must be between 0 and 0.02 meters")

    manifest = read_json(MANIFEST_PATH)
    calibration_samples = [sample for sample in manifest["samples"] if sample["split"] == "calibration"]
    optimization_samples = calibration_samples[: min(args.sample_count, len(calibration_samples))]
    before_config = load_config(CONFIG_PATH)
    refined_config = copy.deepcopy(before_config)
    anchor = copy.deepcopy(before_config["base_camera"])
    bounds = local_bounds(anchor, args.position_delta)

    renderer = CalibrationRenderer(args.optimization_width, args.optimization_height)
    try:
        refined, diagnostics = optimize_camera(
            renderer,
            optimization_samples,
            "base_camera",
            anchor,
            trials=args.trials,
            seed=41,
            bounds_override=bounds,
        )
    finally:
        renderer.close()

    refined_config["base_camera"].update(refined)
    anchor_position = np.asarray(anchor["position"], dtype=np.float64)
    final_position = np.asarray(refined["position"], dtype=np.float64)
    displacement = float(np.linalg.norm(final_position - anchor_position))
    refined_config["metadata"] = {
        **before_config.get("metadata", {}),
        "status": "locally_refined_from_user_anchor",
        "adjustment": "base pose refined locally with position constrained to the user anchor",
        "refined_utc": datetime.now(timezone.utc).isoformat(),
        "position_anchor_m": anchor["position"],
        "position_axis_limit_m": args.position_delta,
        "position_displacement_m": displacement,
        "local_optimization_sample_count": len(optimization_samples),
    }
    write_config(refined_config, CONFIG_PATH)
    write_config(refined_config, CALIBRATION_ROOT / "camera_calibration.yaml")

    comparison = render_comparisons(manifest["samples"], before_config, refined_config)
    metrics = {
        "metric_direction": "lower is better",
        "comparison": "current user anchor versus tightly constrained local refinement",
        "anchor_base_camera": anchor,
        "refined_base_camera": refined,
        "position_displacement_m": displacement,
        "position_axis_limit_m": args.position_delta,
        "aggregate": aggregate_metrics(comparison["samples"]),
        "optimization": diagnostics,
        "per_sample": comparison["samples"],
    }
    write_json(CALIBRATION_ROOT / "local_refinement_metrics.json", metrics)
    write_json(METRICS_PATH, metrics)
    write_json(CALIBRATION_ROOT / "logs" / "local_base_refinement_history.json", diagnostics)
    print(json.dumps({"base_camera": refined, "position_displacement_m": displacement, "metrics": metrics["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
