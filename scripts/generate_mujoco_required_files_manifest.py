"""Generate hashed manifests for the MuJoCo migration file set."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "mujoco_migration"

CATEGORIES: dict[str, list[str]] = {
    "simulation_assets": [
        "simulation/assets/xarm6/xarm6_arm.xml",
        "simulation/assets/xarm6/xarm6_pick_scene.xml",
        "simulation/assets/xarm6/README.md",
        "sim_mujoco/scenes/minimal.xml",
        "simulation/config/camera_calibration.yaml",
        "simulation/config/gripper_mapping.yaml",
        "simulation/calibration/baseline_camera_calibration.yaml",
        "simulation/config/task_scenes.yaml",
        "third_party/xarm_ros2/xarm_description/config/kinematics/default/xarm6_default_kinematics.yaml",
        "third_party/xarm_ros2/xarm_description/config/link_inertial/xarm6_type6_HT_BR2.yaml",
        *[
            "third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/"
            + ("link_base.stl" if index == 0 else f"link{index}.stl")
            for index in range(7)
        ],
    ],
    "simulation_code": [
        "simulation/__init__.py",
        "simulation/configuration.py",
        "simulation/environment.py",
        "simulation/resources.py",
        "simulation/runtime.py",
        "simulation/observation/cameras.py",
        "simulation/observation/policy.py",
        "simulation/observation/state.py",
        "simulation/physics/collision.py",
        "simulation/robot/control.py",
        "simulation/robot/gripper.py",
        "simulation/robot/gripper_mapping.py",
        "simulation/robot/joint_mapping.py",
        "simulation/robot/legacy_gripper.py",
        "simulation/robot/model.py",
        "simulation/scene/objects.py",
        "simulation/scene/reset.py",
        "simulation/scene/runtime.py",
        "simulation/scene/tasks.py",
        "simulation/recording.py",
        "simulation/tools/generate_xarm6_mjcf.py",
        "simulation/tools/build_xarm6_pick_scene.py",
        "sim_mujoco/paths.py",
        "sim_mujoco/remote_policy_evaluation.py",
        "sim_mujoco/scripts/camera_calibration_lib.py",
        "sim_mujoco/scripts/render_task_scenes.py",
        "sim_mujoco/scripts/smoke_test_headless_render.py",
        "sim_mujoco/scripts/audit_kinematic_mapping.py",
    ],
    "policy_integration": [
        *[f"policy_runtime/{name}.py" for name in (
            "__init__",
            "action_decoder",
            "config",
            "environment_protocol",
            "episode_logging",
            "evaluation",
            "image_preprocessing",
            "observation_builder",
            "recording",
            "remote_policy_client",
            "runners",
            "safety",
            "schemas",
        )],
        "tools/evaluation_sim/run_remote_policy_dry_loop.py",
        "evaluation/sim/legacy/run_remote_policy_closed_loop.py",
        "evaluation/sim/legacy/evaluate_remote_policy_interactive.py",
        "sim_mujoco/scripts/test_remote_policy_mujoco.py",
        "sim_mujoco/scripts/test_remote_policy_once.py",
    ],
    "oracle_collection": [
        *[f"sim_mujoco/data_collection/{name}.py" for name in (
            "__init__",
            "conversions",
            "episode_recorder",
            "ik_solver",
            "lerobot_adapter",
            "oracle_controller",
            "real_raw_recorder",
            "task_success",
        )],
        "sim_mujoco/scripts/collect_oracle_data.py",
        "sim_mujoco/scripts/collect_real_raw_sim_data.py",
        "sim_mujoco/scripts/test_scripted_oracle.py",
    ],
    "dataset_tooling": [
        "data/real/conversion/convert_xarm_raw_to_lerobot.py",
        "fine_tune/xarm_lerobot_writer.py",
        "fine_tune/smoke_test_openpi_xarm_dataset.py",
        "sim_mujoco/scripts/convert_mujoco_to_lerobot.py",
        "sim_mujoco/scripts/validate_mujoco_lerobot_dataset.py",
        "sim_mujoco/scripts/validate_real_raw_sim_data.py",
        "sim_mujoco/scripts/compare_real_sim_datasets.py",
        "sim_mujoco/scripts/prepare_mujoco_hf_ready.py",
        "sim_mujoco/scripts/upload_mujoco_dataset_to_hf.py",
    ],
    "configuration_documentation_tests": [
        ".gitignore",
        ".gitmodules",
        "pytest.ini",
        "environment/mujoco_deltaai_requirements.txt",
        "environment/mujoco_deltaai_environment.md",
        "scripts/check_deltaai_mujoco_environment.py",
        "scripts/generate_mujoco_required_files_manifest.py",
        "sim_mujoco/README.md",
        "sim_mujoco/DATA_COLLECTION.md",
        "docs/mujoco_openpi_remote_inference_runbook.md",
        "docs/mujoco_task_scenes.md",
        *[
            f"tests/{name}.py"
            for name in (
                "test_mujoco_chunk_execution",
                "test_mujoco_data_conversions",
                "test_mujoco_episode_recorder",
                "test_mujoco_hf_safety",
                "test_mujoco_lerobot_pipeline",
                "test_mujoco_scripted_oracle",
                "test_policy_runtime_actions",
                "test_policy_runtime_config",
                "test_policy_runtime_evaluation",
                "test_policy_runtime_logging",
                "test_policy_runtime_observation",
                "test_policy_runtime_recording",
                "test_policy_runtime_safety",
                "test_remote_policy_evaluation",
                "test_remote_policy_pipeline",
            )
        ],
        *[
            f"tests/simulation/{name}.py"
            for name in (
                "test_collisions",
                "test_gripper_integration",
                "test_gripper_motion",
                "test_joint_mapping",
                "test_model_contract",
                "test_resources",
                "test_task_scenes",
            )
        ],
    ],
}

GENERATED = {
    "simulation/assets/xarm6/xarm6_arm.xml",
    "simulation/assets/xarm6/xarm6_pick_scene.xml",
}

PURPOSES = {
    "simulation_assets": "Runtime scene, calibration, task, robot-source, or mesh asset",
    "simulation_code": "MuJoCo scene construction, environment, rendering, mapping, or diagnostics",
    "policy_integration": "Observation/action contract, policy client, safety, runner, or evaluation",
    "oracle_collection": "Scripted oracle, IK, success logic, recorder, or collection entry point",
    "dataset_tooling": "LeRobot conversion, validation, comparison, or Hugging Face preparation",
    "configuration_documentation_tests": "Migration configuration, runbook, environment check, or regression test",
}

WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked(relative: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def dependencies(path: Path) -> list[str]:
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return sorted(names)


def portable(path: Path) -> bool:
    if path.suffix.lower() not in {".py", ".md", ".txt", ".yaml", ".yml", ".xml"}:
        return True
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return True
    return not bool(WINDOWS_PATH.search(text))


def build() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing: list[str] = []
    for category, values in CATEGORIES.items():
        for relative in values:
            if relative in seen:
                continue
            seen.add(relative)
            path = ROOT / relative
            if not path.is_file():
                missing.append(relative)
                continue
            rows.append(
                {
                    "relative_path": relative,
                    "category": category,
                    "purpose": PURPOSES[category],
                    "tracked_by_git": tracked(relative),
                    "generated": relative in GENERATED,
                    "portable": portable(path),
                    "required_on_deltaai": True,
                    "dependencies": dependencies(path),
                    "size_bytes": path.stat().st_size,
                    "approximate_size_kib": round(path.stat().st_size / 1024, 2),
                    "sha256": sha256(path),
                }
            )
    return {
        "manifest_version": 1,
        "generated_date": "2026-07-29",
        "repository_root_at_generation": str(ROOT),
        "file_count": len(rows),
        "missing_required_files": missing,
        "all_required_files_present": not missing,
        "all_required_files_tracked": all(row["tracked_by_git"] for row in rows),
        "files": rows,
    }


def write_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Required MuJoCo Migration Files",
        "",
        f"Files: {payload['file_count']}",
        f"All present: {payload['all_required_files_present']}",
        f"All tracked: {payload['all_required_files_tracked']}",
        "",
    ]
    if payload["missing_required_files"]:
        lines.extend(
            [
                "## Missing required files",
                "",
                *[f"- `{value}`" for value in payload["missing_required_files"]],
                "",
            ]
        )
    for category in CATEGORIES:
        lines.extend(
            [
                f"## {category.replace('_', ' ').title()}",
                "",
                "| Path | Tracked | Generated | Portable | Size (KiB) | SHA256 |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for row in payload["files"]:
            if row["category"] != category:
                continue
            lines.append(
                f"| `{row['relative_path']}` | {row['tracked_by_git']} | "
                f"{row['generated']} | {row['portable']} | "
                f"{row['approximate_size_kib']:.2f} | `{row['sha256']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "Generated XML files must be regenerated from their source scripts",
            "after source/config changes. Generated datasets, videos, calibration",
            "imagery, caches, and checkpoints are intentionally absent.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    payload = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "required_files_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "required_files_manifest.md").write_text(
        write_markdown(payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "file_count": payload["file_count"],
                "missing": payload["missing_required_files"],
                "all_tracked": payload["all_required_files_tracked"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
