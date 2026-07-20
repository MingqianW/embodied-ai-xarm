from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

from camera_calibration_lib import (
    CALIBRATION_ROOT,
    BASELINE_CONFIG_PATH,
    CalibrationRenderer,
    MANIFEST_PATH,
    METRICS_PATH,
    aggregate_metrics,
    load_config,
    optimize_camera,
    read_json,
    render_comparisons,
    write_config,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--optimization-width", type=int, default=160)
    parser.add_argument("--optimization-height", type=int, default=120)
    args = parser.parse_args()
    manifest = read_json(MANIFEST_PATH)
    calibration_samples = [sample for sample in manifest["samples"] if sample["split"] == "calibration"]
    optimization_samples = calibration_samples[: min(6, len(calibration_samples))]
    before_config = load_config(BASELINE_CONFIG_PATH)
    calibrated = copy.deepcopy(before_config)
    renderer = CalibrationRenderer(args.optimization_width, args.optimization_height)
    diagnostics = {}
    try:
        for index, camera_name in enumerate(("base_camera", "wrist_camera")):
            parameters, camera_diagnostics = optimize_camera(
                renderer,
                optimization_samples,
                camera_name,
                before_config[camera_name],
                trials=args.trials,
                seed=17 + index,
            )
            calibrated[camera_name].update(parameters)
            diagnostics[camera_name] = camera_diagnostics
            print(camera_name, json.dumps(parameters), "loss=", camera_diagnostics["optimized_loss"])
    finally:
        renderer.close()
    calibrated["metadata"] = {
        "status": "calibrated",
        "calibrated_utc": datetime.now(timezone.utc).isoformat(),
        "units": {"translation": "meters", "angle_internal": "radians", "angle_serialized": "degrees"},
        "optimizer": "bounded deterministic random search plus coordinate refinement",
        "calibration_sample_count": len(calibration_samples),
        "optimization_sample_count": len(optimization_samples),
    }
    write_config(calibrated)
    write_config(calibrated, CALIBRATION_ROOT / "camera_calibration.yaml")
    comparison = render_comparisons(manifest["samples"], before_config, calibrated)
    metrics = {
        "metric_direction": "lower is better",
        "aggregate": aggregate_metrics(comparison["samples"]),
        "optimization": diagnostics,
        "per_sample": comparison["samples"],
    }
    write_json(METRICS_PATH, metrics)
    write_json(CALIBRATION_ROOT / "logs" / "optimization_history.json", diagnostics)
    print(json.dumps(metrics["aggregate"], indent=2))
    print(f"Saved config and outputs under: {CALIBRATION_ROOT}")


if __name__ == "__main__":
    main()
