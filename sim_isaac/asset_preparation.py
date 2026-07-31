from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policy_runtime.config import load_yaml, repository_path


@dataclass(frozen=True)
class AssetPaths:
    source_xacro: Path
    package_root: Path
    xarm_description: Path
    generated_urdf: Path
    generated_usd: Path
    validation_report: Path
    xacro_arguments: dict[str, str]
    isaac_import_options: dict[str, Any]


def load_asset_paths(config_path: Path, project_root: Path) -> AssetPaths:
    config = load_yaml(config_path)
    return AssetPaths(
        source_xacro=repository_path(project_root, config["source"]["xacro"]),
        package_root=repository_path(project_root, config["source"]["package_root"]),
        xarm_description=repository_path(project_root, config["source"]["xarm_description"]),
        generated_urdf=repository_path(project_root, config["generated"]["urdf"]),
        generated_usd=repository_path(project_root, config["generated"]["usd"]),
        validation_report=repository_path(
            project_root,
            config["generated"]["validation_report"],
        ),
        xacro_arguments={
            str(key): str(value) for key, value in config["source"]["arguments"].items()
        },
        isaac_import_options=dict(config["isaac_import"]),
    )


def validate_source_assets(paths: AssetPaths) -> dict[str, Any]:
    required = {
        "source_xacro": paths.source_xacro,
        "xarm6_xacro": paths.xarm_description / "urdf" / "xarm6" / "xarm6.urdf.xacro",
        "gripper_xacro": paths.xarm_description
        / "urdf"
        / "gripper"
        / "xarm_gripper.urdf.xacro",
        "kinematics": paths.xarm_description
        / "config"
        / "kinematics"
        / "default"
        / "xarm6_default_kinematics.yaml",
        "arm_meshes": paths.xarm_description / "meshes" / "xarm6",
        "gripper_meshes": paths.xarm_description / "meshes" / "gripper" / "xarm",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    mesh_files = []
    for directory_name in ("arm_meshes", "gripper_meshes"):
        directory = required[directory_name]
        if directory.is_dir():
            mesh_files.extend(
                path
                for path in directory.rglob("*")
                if path.suffix.lower() in (".stl", ".dae", ".obj")
            )
    source_text = paths.source_xacro.read_text(encoding="utf-8") if paths.source_xacro.is_file() else ""
    result = {
        "source_files": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in required.items()
        },
        "mesh_file_count": len(mesh_files),
        "xacro_declares_device": "xacro:xarm_device" in source_text,
        "missing": missing,
    }
    result["valid"] = not missing and bool(mesh_files) and result["xacro_declares_device"]
    return result


def xacro_command() -> list[str] | None:
    executable = shutil.which("xacro")
    if executable:
        return [executable]
    if importlib.util.find_spec("xacro") is not None:
        return [sys.executable, "-m", "xacro"]
    return None


def expand_xacro(paths: AssetPaths) -> dict[str, Any]:
    command = xacro_command()
    if command is None:
        return {
            "attempted": False,
            "generated": False,
            "error": "xacro executable/module is not installed",
        }
    paths.generated_urdf.parent.mkdir(parents=True, exist_ok=True)
    args = [
        *command,
        str(paths.source_xacro),
        "-o",
        str(paths.generated_urdf),
        *[f"{key}:={value}" for key, value in paths.xacro_arguments.items()],
    ]
    environment = os.environ.copy()
    current_ros_package_path = environment.get("ROS_PACKAGE_PATH", "")
    environment["ROS_PACKAGE_PATH"] = os.pathsep.join(
        part
        for part in (str(paths.package_root), current_ros_package_path)
        if part
    )
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=environment,
    )
    return {
        "attempted": True,
        "generated": result.returncode == 0 and paths.generated_urdf.is_file(),
        "command": args,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def validate_urdf(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "valid": False, "error": f"URDF does not exist: {path}"}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return {"exists": True, "valid": False, "error": str(exc)}
    joints = [element.get("name", "") for element in root.findall("joint")]
    links = [element.get("name", "") for element in root.findall("link")]
    required_joints = [*(f"joint{i}" for i in range(1, 7)), "drive_joint"]
    missing = [name for name in required_joints if name not in joints]
    duplicate_joints = sorted({name for name in joints if joints.count(name) > 1})
    mesh_uris = [
        element.get("filename", "")
        for element in root.findall(".//mesh")
        if element.get("filename")
    ]
    return {
        "exists": True,
        "valid": root.tag == "robot" and not missing and not duplicate_joints,
        "robot_name": root.get("name"),
        "joint_count": len(joints),
        "link_count": len(links),
        "required_joints": required_joints,
        "missing_joints": missing,
        "duplicate_joints": duplicate_joints,
        "mesh_uri_count": len(mesh_uris),
        "mesh_uri_examples": mesh_uris[:10],
    }


def prepare_assets(
    paths: AssetPaths,
    *,
    expand: bool,
    import_usd: bool,
    headless: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_validation": validate_source_assets(paths),
        "xacro": {"attempted": False, "generated": paths.generated_urdf.is_file()},
        "urdf_validation": validate_urdf(paths.generated_urdf),
        "usd": {
            "attempted": False,
            "generated": paths.generated_usd.is_file(),
            "path": str(paths.generated_usd),
        },
    }
    if expand:
        report["xacro"] = expand_xacro(paths)
        report["urdf_validation"] = validate_urdf(paths.generated_urdf)
    if import_usd:
        if not report["urdf_validation"].get("valid"):
            raise RuntimeError("Cannot import USD because the generated URDF is not valid")
        from sim_isaac.version_compat import create_simulation_app, import_urdf_to_usd

        app = create_simulation_app(headless=headless)
        try:
            import_options = dict(paths.isaac_import_options)
            import_options.setdefault(
                "ros_package_paths",
                [
                    {
                        "name": "xarm_description",
                        "path": str(paths.xarm_description),
                    }
                ],
            )
            generated = import_urdf_to_usd(
                paths.generated_urdf,
                paths.generated_usd,
                import_options,
            )
            report["usd"] = {
                "attempted": True,
                "generated": generated.is_file(),
                "path": str(generated),
            }
        finally:
            app.close()
    report["ready_for_isaac"] = bool(
        report["source_validation"]["valid"]
        and report["urdf_validation"].get("valid")
        and report["usd"]["generated"]
    )
    return report


def write_asset_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
