from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_runtime.image_preprocessing import image_diagnostics
from policy_runtime.schemas import POLICY_SCHEMA_VERSION
from sim_isaac.dependencies import IsaacDependencyError
from sim_isaac.environment import DEFAULT_CONFIG_DIR, IsaacEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture raw and policy-preprocessed Isaac camera frames."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG_DIR / "robot.yaml")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CONFIG_DIR / "cameras.yaml")
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONFIG_DIR / "control.yaml")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_CONFIG_DIR / "tasks.yaml")
    return parser.parse_args()


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _camera_stage_report(environment: IsaacEnvironment) -> dict:
    from pxr import UsdGeom

    stage = environment.scene.world.stage
    report = {}
    for name, config in environment.camera_configs.items():
        prim = stage.GetPrimAtPath(config.prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"{name} camera prim is missing: {config.prim_path}")
        camera = UsdGeom.Camera(prim)
        focal = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        horizontal_fov = math.degrees(
            2.0 * math.atan(horizontal_aperture / (2.0 * focal))
        )
        vertical_fov = math.degrees(
            2.0 * math.atan(vertical_aperture / (2.0 * focal))
        )
        expected_parent = (
            "/World"
            if config.parent_frame == "world"
            else (
                f"{environment.mapping.articulation_prim_path}/"
                f"{config.parent_frame}"
            )
        )
        actual_parent = str(prim.GetParent().GetPath())
        width, height = config.resolution
        expected_horizontal_fov = math.degrees(
            2.0
            * math.atan(
                math.tan(math.radians(config.vertical_fov_deg) / 2.0)
                * width
                / height
            )
        )
        report[name] = {
            "prim_path": config.prim_path,
            "expected_parent_path": expected_parent,
            "actual_parent_path": actual_parent,
            "parent_matches": actual_parent == expected_parent,
            "focal_length": focal,
            "horizontal_aperture": horizontal_aperture,
            "vertical_aperture": vertical_aperture,
            "horizontal_fov_deg": horizontal_fov,
            "expected_horizontal_fov_deg": expected_horizontal_fov,
            "vertical_fov_deg": vertical_fov,
            "expected_vertical_fov_deg": config.vertical_fov_deg,
            "optics_match": bool(
                np.isclose(vertical_fov, config.vertical_fov_deg, atol=1e-4)
                and np.isclose(horizontal_fov, expected_horizontal_fov, atol=1e-4)
            ),
            "optical_axis_convention": "local_negative_z",
        }
    return report


def _gripper_material_report(environment: IsaacEnvironment) -> dict:
    from pxr import Usd, UsdGeom, UsdShade

    stage = environment.scene.world.stage
    root_path = (
        f"{environment.mapping.articulation_prim_path}/"
        f"{environment.mapping.gripper_visual_frame}"
    )
    expected_material_path = "/World/Looks/xarm_gripper_black"
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Gripper visual root is missing: {root_path}")

    root_material, _ = UsdShade.MaterialBindingAPI(
        root_prim
    ).ComputeBoundMaterial()
    root_material_path = (
        str(root_material.GetPath()) if root_material else None
    )
    bound_materials = {}
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        bound_materials[str(prim.GetPath())] = (
            str(material.GetPath()) if material else None
        )
    return {
        "root_prim_path": root_path,
        "expected_material_path": expected_material_path,
        "configured_color_rgb": list(environment.mapping.gripper_color_rgb),
        "root_material_path": root_material_path,
        "root_binding_black": root_material_path == expected_material_path,
        "gprim_count": len(bound_materials),
        "bound_materials": bound_materials,
        "all_direct_gprims_black": (
            None
            if not bound_materials
            else all(
                path == expected_material_path
                for path in bound_materials.values()
            )
        ),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path(
        os.environ.get("ISAAC_OUTPUT_DIR", "sim_isaac/output")
    ) / "camera_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "camera_report.json"
    report_path.write_text(
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
            environment.require_safe("camera inspection")
            observation = environment.observe()
            raw = environment.recording_frames()
            images = {
                "base_raw": raw["base"],
                "wrist_raw": raw["wrist"],
                "base_policy": observation.base_image,
                "wrist_policy": observation.wrist_image,
            }
            for name, image in images.items():
                _save(output_dir / f"{name}.png", image)
            stage_report = _camera_stage_report(environment)
            gripper_material = _gripper_material_report(environment)
            structural_ok = bool(
                all(item["parent_matches"] for item in stage_report.values())
                and all(item["optics_match"] for item in stage_report.values())
                and gripper_material["root_binding_black"]
            )
            report = {
                "status": "passed" if structural_ok else "failed",
                "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
                "schema_version": POLICY_SCHEMA_VERSION,
                "camera_backend": environment.cameras.backend,
                "frame_ids": observation.frame_ids,
                "images": {
                    name: image_diagnostics(image) for name, image in images.items()
                },
                "config": {
                    name: {
                        "prim_path": config.prim_path,
                        "parent_frame": config.parent_frame,
                        "resolution": list(config.resolution),
                        "vertical_fov_deg": config.vertical_fov_deg,
                    }
                    for name, config in environment.camera_configs.items()
                },
                "stage": stage_report,
                "all_parent_frames_match": all(
                    item["parent_matches"] for item in stage_report.values()
                ),
                "all_optics_match": all(
                    item["optics_match"] for item in stage_report.values()
                ),
                "gripper_material": gripper_material,
                "object_position_m": environment.scene.objects.position().tolist(),
                "safety": environment.safety_diagnostics(),
                "calibration_status": "initial_estimate_requires_visual_comparison",
            }
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(report, indent=2))
            if not structural_ok:
                print(
                    "ERROR: camera parent, optics, or gripper appearance validation failed",
                    file=sys.stderr,
                )
                return 3
    except (IsaacDependencyError, FileNotFoundError, RuntimeError, ValueError) as exc:
        report_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
