from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_isaac.dependencies import IsaacDependencyError
from sim_isaac.environment import DEFAULT_CONFIG_DIR, IsaacEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the generated xArm USD and report articulation/frame mappings."
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG_DIR / "robot.yaml")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CONFIG_DIR / "cameras.yaml")
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONFIG_DIR / "control.yaml")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_CONFIG_DIR / "tasks.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path (recommended for headless runs).",
    )
    parser.add_argument(
        "--check-motion",
        action="store_true",
        help="Run a reset-isolated, bounded direction/tracking check on each arm joint.",
    )
    parser.add_argument("--joint-delta-rad", type=float, default=0.02)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    return parser.parse_args()


def _check_arm_motion(
    environment: IsaacEnvironment,
    *,
    delta_rad: float,
    settle_seconds: float,
) -> dict:
    if not 0.0 < delta_rad <= 0.05:
        raise ValueError("--joint-delta-rad must be in (0, 0.05]")
    if not 0.0 < settle_seconds <= 1.0:
        raise ValueError("--settle-seconds must be in (0, 1.0]")

    minimum_directional_motion = max(1e-4, delta_rad * 0.02)
    samples = []
    for joint_index, joint_name in enumerate(
        environment.mapping.canonical_arm_joint_names
    ):
        baseline = environment.reset(seed=environment.seed).state.copy()
        environment.require_safe(f"{joint_name} direction-check reset")
        lower, upper = environment.joint_limits[joint_index]
        direction = (
            1.0
            if baseline[joint_index] + delta_rad <= upper
            else -1.0
        )
        target = baseline.copy()
        target[joint_index] += direction * delta_rad
        environment.apply_action(target)
        environment.step_physics(settle_seconds)
        measured = environment.scene.robot.get_policy_state()
        measured_delta = float(
            measured[joint_index] - baseline[joint_index]
        )
        target_error = float(abs(measured[joint_index] - target[joint_index]))
        diagnostics = environment.safety_diagnostics()
        safe = environment.is_safe()
        direction_ok = bool(
            direction * measured_delta >= minimum_directional_motion
        )
        tracking_ok = bool(target_error <= 0.05)
        sample = {
            "joint_name": joint_name,
            "baseline_rad": float(baseline[joint_index]),
            "target_rad": float(target[joint_index]),
            "measured_rad": float(measured[joint_index]),
            "commanded_delta_rad": float(direction * delta_rad),
            "measured_delta_rad": measured_delta,
            "absolute_target_error_rad": target_error,
            "direction_ok": direction_ok,
            "tracking_ok": tracking_ok,
            "is_safe": safe,
            "safety": diagnostics,
        }
        samples.append(sample)
        if not safe:
            environment.hold_position()
            break

    passed = bool(
        len(samples) == len(environment.mapping.canonical_arm_joint_names)
        and all(
            sample["direction_ok"]
            and sample["tracking_ok"]
            and sample["is_safe"]
            for sample in samples
        )
    )
    if passed:
        environment.reset(seed=environment.seed)
        environment.require_safe("post-motion reset")
    return {
        "status": "passed" if passed else "failed",
        "joint_delta_rad": delta_rad,
        "settle_seconds": settle_seconds,
        "minimum_directional_motion_rad": minimum_directional_motion,
        "maximum_target_error_rad": 0.05,
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve() if args.output is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"status": "starting"}, indent=2) + "\n",
            encoding="utf-8",
        )
    try:
        with IsaacEnvironment(
            robot_config_path=args.robot_config,
            camera_config_path=args.camera_config,
            control_config_path=args.control_config,
            task_config_path=args.task_config,
            headless=args.headless,
        ) as environment:
            stage = environment.scene.world.stage
            inspected_at_utc = datetime.now(timezone.utc).isoformat()
            required_frames = {
                name.rsplit("/", 1)[-1]: (
                    f"{environment.mapping.articulation_prim_path}/{name}"
                )
                for name in (
                    environment.mapping.base_frame,
                    environment.mapping.end_effector_frame,
                )
            }
            from pxr import PhysxSchema, Usd, UsdPhysics

            articulation_roots = [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ]
            robot_root_prim = stage.GetPrimAtPath(
                environment.mapping.articulation_prim_path
            )
            rigid_body_gravity = {}
            for prim in Usd.PrimRange(robot_root_prim):
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Get(
                    stage,
                    prim.GetPath(),
                )
                disable_gravity = (
                    physx_rigid_body.GetDisableGravityAttr().Get()
                    if physx_rigid_body
                    else None
                )
                rigid_body_gravity[str(prim.GetPath())] = {
                    "gravity_enabled": (
                        None
                        if disable_gravity is None
                        else not bool(disable_gravity)
                    ),
                    "matches_config": (
                        disable_gravity is not None
                        and (not bool(disable_gravity))
                        == environment.mapping.gravity_enabled
                    ),
                }
            rigid_body_gravity_matches = bool(rigid_body_gravity) and all(
                item["matches_config"] for item in rigid_body_gravity.values()
            )
            frame_candidates = {
                frame_name: [
                    str(prim.GetPath())
                    for prim in stage.Traverse()
                    if prim.GetName() == frame_name
                ]
                for frame_name in ("link_base", "link_tcp")
            }
            joint_details = {}
            for prim in stage.Traverse():
                if not prim.IsA(UsdPhysics.RevoluteJoint):
                    continue
                joint = UsdPhysics.RevoluteJoint(prim)
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                lower_deg = joint.GetLowerLimitAttr().Get()
                upper_deg = joint.GetUpperLimitAttr().Get()
                stiffness = drive.GetStiffnessAttr().Get() if drive else None
                damping = drive.GetDampingAttr().Get() if drive else None
                max_force = drive.GetMaxForceAttr().Get() if drive else None
                joint_details[prim.GetName()] = {
                    "prim_path": str(prim.GetPath()),
                    "axis": str(joint.GetAxisAttr().Get()),
                    "lower_limit_deg": lower_deg,
                    "upper_limit_deg": upper_deg,
                    "lower_limit_rad": (
                        None if lower_deg is None else math.radians(lower_deg)
                    ),
                    "upper_limit_rad": (
                        None if upper_deg is None else math.radians(upper_deg)
                    ),
                    "drive_stiffness_usd_per_deg": stiffness,
                    "drive_damping_usd_per_deg": damping,
                    "drive_max_force": max_force,
                }
            required_frame_report = {
                name: {
                    "prim_path": path,
                    "exists": bool(stage.GetPrimAtPath(path).IsValid()),
                }
                for name, path in required_frames.items()
            }
            required_frames_exist = all(
                item["exists"] for item in required_frame_report.values()
            )
            motion_check = (
                _check_arm_motion(
                    environment,
                    delta_rad=args.joint_delta_rad,
                    settle_seconds=args.settle_seconds,
                )
                if args.check_motion
                else {"status": "not_run"}
            )
            all_joint_positions = environment.scene.robot.prim.get_joint_positions()
            joint_positions = {
                name: float(all_joint_positions[index])
                for index, name in enumerate(environment.scene.robot.joint_names)
            }
            report = {
                "inspected_at_utc": inspected_at_utc,
                "asset_path": environment.mapping.asset_path,
                "articulation_prim_path": environment.mapping.articulation_prim_path,
                "available_joint_names": list(environment.scene.robot.joint_names),
                "canonical_arm_joint_names": list(
                    environment.mapping.canonical_arm_joint_names
                ),
                "isaac_arm_joint_names": list(environment.mapping.isaac_arm_joint_names),
                "gripper_joint_name": environment.mapping.gripper_joint_name,
                "gripper_mimic_joint_names": list(
                    environment.mapping.gripper_mimic_joint_names
                ),
                "state": environment.scene.robot.get_policy_state().tolist(),
                "joint_positions_rad": joint_positions,
                "articulation_root_prims": articulation_roots,
                "configured_gravity_enabled": environment.mapping.gravity_enabled,
                "rigid_body_gravity": rigid_body_gravity,
                "all_rigid_body_gravity_matches": rigid_body_gravity_matches,
                "required_frames": required_frame_report,
                "frame_candidates": frame_candidates,
                "joint_details": joint_details,
                "physical_gripper_aperture_validated": (
                    environment.mapping.physical_aperture_validated
                ),
                "arm_motion_check": motion_check,
                "safety": environment.safety_diagnostics(),
                "is_safe": environment.is_safe(),
            }
            motion_ok = (
                not args.check_motion or motion_check["status"] == "passed"
            )
            report["status"] = (
                "passed"
                if (
                    required_frames_exist
                    and rigid_body_gravity_matches
                    and report["is_safe"]
                    and motion_ok
                )
                else "failed"
            )
            report_json = json.dumps(report, indent=2)
            if output_path is not None:
                output_path.write_text(report_json + "\n", encoding="utf-8")
            print(report_json)
            if not required_frames_exist:
                print(
                    "ERROR: an expected robot frame is missing; inspect the imported USD hierarchy.",
                    file=sys.stderr,
                )
                return 3
            if not rigid_body_gravity_matches:
                print(
                    "ERROR: robot rigid-body gravity does not match robot.yaml.",
                    file=sys.stderr,
                )
                return 6
            if not report["is_safe"]:
                print(
                    "ERROR: articulation failed the bounded reset safety check; "
                    "do not continue to gripper motion.",
                    file=sys.stderr,
                )
                return 4
            if not motion_ok:
                print(
                    "ERROR: one or more arm joints failed the bounded "
                    "direction/tracking check; do not continue to gripper motion.",
                    file=sys.stderr,
                )
                return 5
    except (IsaacDependencyError, FileNotFoundError, RuntimeError, ValueError) as exc:
        if output_path is not None:
            output_path.write_text(
                json.dumps({"status": "failed", "error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
