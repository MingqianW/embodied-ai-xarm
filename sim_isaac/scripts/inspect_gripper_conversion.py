from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_isaac.articulation import gripper_mm_to_isaac
from sim_isaac.environment import IsaacEnvironment


DEFAULT_OUTPUT = ROOT / "sim_isaac" / "output" / "gripper_conversion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure policy gripper values against Isaac joint motion and jaw gap."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _finger_measurement(environment: IsaacEnvironment) -> dict:
    from pxr import Usd, UsdGeom

    stage = environment.scene.world.stage
    tcp_path = (
        f"{environment.mapping.articulation_prim_path}/"
        f"{environment.mapping.end_effector_frame}"
    )
    gripper_base_path = tcp_path.rsplit("/", 1)[0]
    finger_paths = {
        "left": f"{gripper_base_path}/left_outer_knuckle/left_finger",
        "right": f"{gripper_base_path}/right_outer_knuckle/right_finger",
    }
    centers = {}
    ranges = {}
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    for name, path in finger_paths.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Finger link is missing: {path}")
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        centers[name] = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        ranges[name] = (
            np.asarray(aligned.GetMin(), dtype=np.float64),
            np.asarray(aligned.GetMax(), dtype=np.float64),
        )
        if (
            not np.isfinite(ranges[name][0]).all()
            or not np.isfinite(ranges[name][1]).all()
            or np.any(ranges[name][0] > ranges[name][1])
        ):
            raise RuntimeError(f"Finger link has no finite render bound: {path}")

    separation = centers["right"] - centers["left"]
    separation_axis = int(np.argmax(np.abs(separation)))
    left_min, left_max = ranges["left"]
    right_min, right_max = ranges["right"]
    if centers["left"][separation_axis] <= centers["right"][separation_axis]:
        gap = right_min[separation_axis] - left_max[separation_axis]
    else:
        gap = left_min[separation_axis] - right_max[separation_axis]
    return {
        "finger_link_paths": finger_paths,
        "left_origin_m": centers["left"].tolist(),
        "right_origin_m": centers["right"].tolist(),
        "origin_separation_m": float(np.linalg.norm(separation)),
        "separation_axis_world": "XYZ"[separation_axis],
        "jaw_gap_world_aabb_m": float(gap),
    }


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    _write(output_path, {"status": "starting"})
    environment = None
    try:
        environment = IsaacEnvironment(headless=True, seed=args.seed)
        environment.require_safe("gripper sweep")
        home = np.concatenate(
            [
                environment.mapping.initial_arm_positions_rad,
                np.asarray(
                    [environment.mapping.initial_gripper_policy],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        policy_values = np.linspace(
            environment.mapping.gripper_policy_open,
            environment.mapping.gripper_policy_closed,
            5,
            dtype=np.float32,
        )
        gripper_names = [
            environment.mapping.gripper_joint_name,
            *environment.mapping.gripper_mimic_joint_names,
        ]
        dof_indices = {
            name: environment.scene.robot.joint_names.index(name)
            for name in gripper_names
        }
        samples = []
        for policy_value in policy_values:
            target = home.copy()
            target[6] = policy_value
            target_rad = gripper_mm_to_isaac(float(policy_value), environment.mapping)
            environment.apply_action(target)
            environment.step_physics(args.settle_seconds)
            environment.require_safe(
                f"gripper sweep sample policy_value={float(policy_value)}"
            )
            positions = np.asarray(
                environment.scene.robot.prim.get_joint_positions(),
                dtype=np.float64,
            )
            measured = {
                name: float(positions[index])
                for name, index in dof_indices.items()
            }
            drive_error = abs(
                measured[environment.mapping.gripper_joint_name] - target_rad
            )
            samples.append(
                {
                    "policy_value": float(policy_value),
                    "target_drive_rad": target_rad,
                    "actual_joint_rad": measured,
                    "absolute_drive_tracking_error_rad": drive_error,
                    **_finger_measurement(environment),
                    "safety": environment.safety_diagnostics(),
                    "is_safe": environment.is_safe(),
                }
            )
            if drive_error > 0.05:
                environment.hold_position()
                break

        samples.sort(key=lambda sample: sample["policy_value"])
        gaps = np.asarray(
            [sample["jaw_gap_world_aabb_m"] for sample in samples],
            dtype=np.float64,
        )
        origins = np.asarray(
            [sample["origin_separation_m"] for sample in samples],
            dtype=np.float64,
        )
        max_tracking_error = max(
            sample["absolute_drive_tracking_error_rad"] for sample in samples
        )
        finite = bool(
            np.isfinite(gaps).all()
            and np.isfinite(origins).all()
            and np.isfinite(max_tracking_error)
        )
        gap_monotonic = bool(np.all(np.diff(gaps) >= -1e-5))
        origin_monotonic = bool(np.all(np.diff(origins) >= -1e-5))
        safe = all(sample["is_safe"] for sample in samples)
        passed = bool(
            len(samples) == len(policy_values)
            and finite
            and gap_monotonic
            and origin_monotonic
            and max_tracking_error <= 0.05
            and safe
        )
        report = {
            "status": "passed" if passed else "failed",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_definition": (
                "Jaw gap is the separation between the inward faces of the "
                "finger-link world-aligned bounding boxes along their dominant "
                "world separation axis; it is a mesh-based estimate, not a "
                "calibrated physical aperture."
            ),
            "settle_seconds": args.settle_seconds,
            "execution_order": "open_to_closed_incremental",
            "requested_sample_count": int(len(policy_values)),
            "completed_sample_count": int(len(samples)),
            "gripper_joint_names": gripper_names,
            "max_gripper_tracking_error_rad": max_tracking_error,
            "jaw_gap_monotonic": gap_monotonic,
            "origin_separation_monotonic": origin_monotonic,
            "all_samples_safe": safe,
            "samples": samples,
        }
        _write(output_path, report)
        return 0 if passed else 3
    except Exception as exc:
        _write(
            output_path,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
