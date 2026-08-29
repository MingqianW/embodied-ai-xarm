"""Single command-line interface for maintained xArm camera diagnostics."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from PIL import Image

from diagnostics.simulation.camera import calibration as camera


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("discover", help="Summarize configured raw camera data.")

    select = commands.add_parser("select", help="Select calibration frames.")
    select.add_argument("--calibration-count", type=int, default=12)
    select.add_argument("--validation-count", type=int, default=4)
    select.add_argument("--max-episodes", type=int, default=36)

    fit = commands.add_parser(
        "fit",
        help="Fit both cameras and write the canonical simulation camera config.",
    )
    fit.add_argument("--trials", type=int, default=120)
    fit.add_argument("--optimization-width", type=int, default=160)
    fit.add_argument("--optimization-height", type=int, default=120)

    commands.add_parser(
        "evaluate",
        help="Validate compiled cameras, raw snapshot, renders, and metrics.",
    )

    render = commands.add_parser("render", help="Render one selected frame.")
    render.add_argument("--sample-id", default="sample_000")
    render.add_argument(
        "--camera",
        choices=("base_camera", "wrist_camera", "both"),
        default="both",
    )
    render.add_argument(
        "--output-dir",
        type=Path,
        default=camera.CALIBRATION_ROOT / "single_frame",
    )
    return parser


def _discover() -> None:
    summary = camera.dataset_summary(camera.discover_episodes())
    camera.write_json(camera.DISCOVERY_PATH, summary)
    print(json.dumps(summary, indent=2))


def _select(args: argparse.Namespace) -> None:
    episodes = camera.discover_episodes()
    candidates = camera.sample_candidates(episodes, max_episodes=args.max_episodes)
    samples = camera.select_diverse_frames(
        candidates,
        args.calibration_count,
        args.validation_count,
    )
    manifest = {
        "schema_version": 1,
        "selection_method": (
            "sharpness/content filtering followed by farthest-point joint-pose sampling"
        ),
        "camera_mapping": {
            "base_camera": "realsense_0",
            "wrist_camera": "realsense_1",
        },
        "joint_order": [f"j{index}_rad" for index in range(1, 7)],
        "joint_units": "radians",
        "raw_snapshot": camera.verify_raw_snapshot(episodes),
        "samples": samples,
    }
    camera.write_json(camera.MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "selected": len(samples),
                "episodes": sorted({sample["episode"] for sample in samples}),
            },
            indent=2,
        )
    )


def _fit(args: argparse.Namespace) -> None:
    manifest = camera.read_json(camera.MANIFEST_PATH)
    calibration_samples = [
        sample for sample in manifest["samples"] if sample["split"] == "calibration"
    ]
    optimization_samples = calibration_samples[: min(6, len(calibration_samples))]
    before_config = camera.load_config(camera.BASELINE_CONFIG_PATH)
    calibrated = copy.deepcopy(before_config)
    renderer = camera.CalibrationRenderer(
        args.optimization_width,
        args.optimization_height,
    )
    diagnostics = {}
    try:
        for index, camera_name in enumerate(("base_camera", "wrist_camera")):
            parameters, details = camera.optimize_camera(
                renderer,
                optimization_samples,
                camera_name,
                before_config[camera_name],
                trials=args.trials,
                seed=17 + index,
            )
            calibrated[camera_name].update(parameters)
            diagnostics[camera_name] = details
    finally:
        renderer.close()
    calibrated["metadata"] = {
        "status": "calibrated",
        "calibrated_utc": datetime.now(timezone.utc).isoformat(),
        "units": {
            "translation": "meters",
            "angle_internal": "radians",
            "angle_serialized": "degrees",
        },
        "optimizer": "bounded deterministic random search plus coordinate refinement",
        "base_camera_side": "positive_x_front_view",
        "calibration_sample_count": len(calibration_samples),
        "optimization_sample_count": len(optimization_samples),
    }
    camera.write_config(calibrated)
    camera.write_config(calibrated, camera.CALIBRATION_ROOT / "camera_calibration.yaml")
    comparison = camera.render_comparisons(
        manifest["samples"], before_config, calibrated
    )
    metrics = {
        "metric_direction": "lower is better",
        "aggregate": camera.aggregate_metrics(comparison["samples"]),
        "optimization": diagnostics,
        "per_sample": comparison["samples"],
    }
    camera.write_json(camera.METRICS_PATH, metrics)
    camera.write_json(
        camera.CALIBRATION_ROOT / "logs" / "optimization_history.json",
        diagnostics,
    )
    print(json.dumps(metrics["aggregate"], indent=2))


def _evaluate() -> None:
    manifest = camera.read_json(camera.MANIFEST_PATH)
    metrics = camera.read_json(camera.METRICS_PATH)
    model = mujoco.MjModel.from_xml_path(str(camera.MODEL_PATH))
    config = camera.load_config()
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    current_raw = camera.verify_raw_snapshot(camera.discover_episodes())
    applied = {}
    for camera_name in ("base_camera", "wrist_camera"):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        expected = config[camera_name]
        applied[camera_name] = bool(
            camera_id >= 0
            and np.allclose(model.cam_pos[camera_id], expected["position"], atol=1e-8)
            and np.isclose(model.cam_fovy[camera_id], expected["fovy_deg"], atol=1e-8)
        )
    native_files = sorted(camera.CALIBRATION_ROOT.glob("after/*_sim_*.png"))
    policy_files = sorted(camera.CALIBRATION_ROOT.glob("after/*_policy_*.png"))
    native_sizes = sorted({Image.open(path).size for path in native_files})
    policy_sizes = sorted({Image.open(path).size for path in policy_files})
    report = {
        "scene_compiles": True,
        "model_path": str(camera.MODEL_PATH),
        "config_path": str(camera.CONFIG_PATH),
        "native_render_files": len(native_files),
        "native_render_sizes": [list(size) for size in native_sizes],
        "policy_render_files": len(policy_files),
        "policy_render_sizes": [list(size) for size in policy_sizes],
        "camera_config_applied": applied,
        "actuator_count": model.nu,
        "actuators": actuator_names,
        "six_arm_plus_one_gripper": model.nu == 7
        and actuator_names[-1] == "gripper_actuator",
        "raw_snapshot_unchanged": current_raw == manifest["raw_snapshot"],
        "raw_root": str(camera.RAW_ROOT),
        "metrics": metrics["aggregate"],
    }
    print(json.dumps(report, indent=2))
    camera.write_json(camera.CALIBRATION_ROOT / "validation_report.json", report)
    required = (
        report["six_arm_plus_one_gripper"],
        report["raw_snapshot_unchanged"],
        all(applied.values()),
        native_sizes == [(640, 480)] and len(native_files) == 32,
        policy_sizes == [(224, 224)] and len(policy_files) == 32,
    )
    if not all(required):
        raise SystemExit(1)


def _render(args: argparse.Namespace) -> None:
    manifest = camera.read_json(camera.MANIFEST_PATH)
    sample = next(
        item for item in manifest["samples"] if item["sample_id"] == args.sample_id
    )
    config = camera.load_config()
    render_config = config["render"]
    renderer = camera.CalibrationRenderer(
        int(render_config["native_width"]),
        int(render_config["native_height"]),
    )
    cameras = (
        ("base_camera", "wrist_camera")
        if args.camera == "both"
        else (args.camera,)
    )
    try:
        for camera_name in cameras:
            native = renderer.render(sample, camera_name, config[camera_name])
            short = camera_name.replace("_camera", "")
            camera.save_rgb(
                args.output_dir / f"{args.sample_id}_{short}_640x480.png", native
            )
            camera.save_rgb(
                args.output_dir / f"{args.sample_id}_{short}_224x224.png",
                camera.policy_image(native, config),
            )
    finally:
        renderer.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "discover":
        _discover()
    elif args.command == "select":
        _select(args)
    elif args.command == "fit":
        _fit(args)
    elif args.command == "evaluate":
        _evaluate()
    elif args.command == "render":
        _render(args)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
