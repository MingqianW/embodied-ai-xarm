"""Collect local Isaac Sim demonstrations with a deterministic scripted oracle.

This collector never connects to OpenPI.  It resets the configured task,
generates bounded joint-space waypoints from the target pose, executes them in
Isaac Sim, and stores observation/action pairs plus a recording for every
episode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_runtime.episode_logging import EpisodeLogger, write_json
from policy_runtime.recording import VideoRecorder
from sim_isaac.environment import DEFAULT_CONFIG_DIR, IsaacEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sim_isaac" / "output" / "oracle_dataset")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_CONFIG_DIR / "tasks.yaml")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--action-hz", type=float, default=10.0)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep already-successful episode directories and fill only missing/rejected slots.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Archive failed attempts under _rejected and retry until every requested episode succeeds.",
    )
    parser.add_argument(
        "--max-attempts-per-episode",
        type=int,
        default=20,
        help="Maximum attempts for one accepted episode when --require-success is enabled.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Isaac Sim headless mode. GUI mode is the safer default for RTX camera startup.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "legacy"),
        default="auto",
        help="Camera backend for local collection; auto uses Isaac Sim 6 RTX cameras.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Validate and write the 200-episode plan without launching Isaac Sim.")
    return parser.parse_args()


def _clipped(value: np.ndarray, limits: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    if result.shape != (7,) or limits.shape != (6, 2):
        raise ValueError("Oracle target must be shape (7,) with arm limits (6, 2)")
    result[:6] = np.clip(result[:6], limits[:, 0], limits[:, 1])
    return result


def oracle_waypoints(environment: IsaacEnvironment) -> list[tuple[str, np.ndarray]]:
    """Return DLS-IK-calibrated xArm grasp or ring-placement waypoints."""

    limits = environment.joint_limits
    # Calibrated from the repository's successful MuJoCo DLS-IK oracle. Both
    # simulators use the same xArm joint order and world-frame task geometry.
    target = np.asarray(environment.scene.objects.position(), dtype=np.float32)
    reference_xy = np.asarray([0.46895694, -0.22341706], dtype=np.float32)
    azimuth_delta = float(
        np.arctan2(target[1], target[0])
        - np.arctan2(reference_xy[1], reference_xy[0])
    )
    pregrasp = np.asarray(
        [-0.458020, 0.229616, -1.255107, 0.035675, 1.049681, -0.474920, 845.0],
        dtype=np.float32,
    )
    descend = np.asarray(
        [-0.456260, 0.376043, -1.154365, 0.043083, 0.802647, -0.485338, 845.0],
        dtype=np.float32,
    )
    lift = np.asarray(
        [-0.458324, 0.209310, -1.287446, 0.034668, 1.102302, -0.473117, 211.0],
        dtype=np.float32,
    )
    for waypoint in (pregrasp, descend, lift):
        waypoint[0] += azimuth_delta
    close = descend.copy()
    close[6] = 211.0
    waypoints = [
        ("pregrasp", _clipped(pregrasp, limits)),
        ("descend", _clipped(descend, limits)),
        ("close", _clipped(close, limits)),
        ("lift", _clipped(lift, limits)),
    ]
    if environment.task.place_in_ring:
        preplace = np.asarray(
            [0.101444, 0.067635, -1.102747, 0.185296, 1.029007, -0.280881, 211.0],
            dtype=np.float32,
        )
        place = np.asarray(
            [0.109282, 0.173250, -1.004916, 0.215755, 0.828348, -0.323691, 211.0],
            dtype=np.float32,
        )
        release = place.copy()
        release[6] = environment.scene.robot.mapping.gripper_policy_open
        waypoints.extend(
            [
                ("preplace", _clipped(preplace, limits)),
                ("place", _clipped(place, limits)),
                ("release", _clipped(release, limits)),
            ]
        )
    return waypoints


def _interpolate(
    start: np.ndarray, target: np.ndarray, steps: int | None = None
) -> list[np.ndarray]:
    if steps is None:
        arm_steps = int(
            np.ceil(float(np.max(np.abs(target[:6] - start[:6]))) / 0.025)
        )
        gripper_steps = int(
            np.ceil(abs(float(target[6] - start[6])) / 25.0)
        )
        steps = max(1, arm_steps, gripper_steps)
    return [
        (start + (target - start) * alpha).astype(np.float32)
        for alpha in np.linspace(0.0, 1.0, max(1, steps) + 1)[1:]
    ]


def collect_episode(
    environment: IsaacEnvironment,
    *,
    task: str,
    prompt: str,
    seed: int,
    episode_dir: Path,
    max_steps: int,
    action_hz: float,
    record: bool,
) -> dict[str, Any]:
    episode_dir.mkdir(parents=True, exist_ok=False)
    logger = EpisodeLogger(episode_dir, simulator="isaac")
    recorder = VideoRecorder(episode_dir / "recording", fps=round(action_hz)) if record else None
    observation = environment.reset(seed=seed)
    initial_state = observation.state.copy()
    initial_object_position = environment.scene.objects.position().copy()
    logger.save_array("initial/state.npy", observation.state)
    logger.save_image("initial/base_image.png", observation.base_image)
    logger.save_image("initial/wrist_image.png", observation.wrist_image)
    if recorder is not None:
        recorder.write(environment.recording_frames())
    transitions: list[dict[str, Any]] = []
    current = observation.state.copy()
    step_index = 0
    for stage, waypoint in oracle_waypoints(environment):
        for action in _interpolate(current, waypoint):
            if step_index >= max_steps:
                break
            logger.save_array(f"step_{step_index:03d}/state.npy", observation.state)
            logger.save_image(f"step_{step_index:03d}/base_image.png", observation.base_image)
            logger.save_image(f"step_{step_index:03d}/wrist_image.png", observation.wrist_image)
            logger.save_array(f"step_{step_index:03d}/action.npy", action)
            logger.log("oracle_step", step=step_index, stage=stage, state=observation.state, action=action)
            environment.apply_action(action)
            environment.step_physics(1.0 / action_hz)
            observation = environment.observe()
            if recorder is not None:
                recorder.write(environment.recording_frames())
            transitions.append({"step": step_index, "stage": stage, "action": action.tolist()})
            current = action
            step_index += 1
        if step_index >= max_steps:
            break
    final_observation = environment.observe()
    final_object_position = environment.scene.objects.position().copy()
    logger.save_array("final/state.npy", final_observation.state)
    logger.save_image("final/base_image.png", final_observation.base_image)
    logger.save_image("final/wrist_image.png", final_observation.wrist_image)
    recording = recorder.metadata() if recorder is not None else {}
    if recorder is not None:
        recorder.close()
    lift_height_m = float(final_object_position[2] - initial_object_position[2])
    if environment.task.place_in_ring:
        ring_center = np.asarray([0.48, 0.08], dtype=np.float32)
        placement_error_m = float(np.linalg.norm(final_object_position[:2] - ring_center))
        task_success = bool(placement_error_m <= 0.06)
        task_metric = {"placement_error_m": placement_error_m}
    else:
        task_success = bool(lift_height_m >= environment.task.success_lift_height_m)
        task_metric = {
            "lift_height_m": lift_height_m,
            "required_lift_height_m": environment.task.success_lift_height_m,
        }
    result = {
        "status": "collected",
        "task": task,
        "prompt": prompt,
        "seed": seed,
        "oracle": "joint_waypoint_v1",
        "steps": step_index,
        "initial_state": initial_state.tolist(),
        "final_state": final_observation.state.tolist(),
        "initial_object_position_m": initial_object_position.tolist(),
        "final_object_position_m": final_object_position.tolist(),
        "task_success": task_success,
        "task_metric": task_metric,
        "transitions": transitions,
        "recording": recording,
        "safety": environment.safety_diagnostics(),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    logger.write_metadata(result)
    return result


def _existing_success(episode_dir: Path, task: str) -> dict[str, Any] | None:
    metadata_path = episode_dir / "episode.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("task") != task or metadata.get("task_success") is not True:
        return None
    return metadata


def _archive_rejected(
    episode_dir: Path, output_dir: Path, task: str, *, seed: int, attempt: int
) -> Path:
    rejected_dir = output_dir / "_rejected" / task
    rejected_dir.mkdir(parents=True, exist_ok=True)
    candidate = rejected_dir / (
        f"{episode_dir.name}_seed_{seed:08d}_attempt_{attempt:02d}"
    )
    suffix = 1
    while candidate.exists():
        candidate = rejected_dir / (
            f"{episode_dir.name}_seed_{seed:08d}_attempt_{attempt:02d}_{suffix:02d}"
        )
        suffix += 1
    episode_dir.rename(candidate)
    return candidate


def main() -> int:
    args = parse_args()
    if args.camera_backend == "legacy":
        os.environ["ISAAC_CAMERA_BACKEND"] = "legacy"
    from policy_runtime.config import load_yaml

    raw = load_yaml(args.task_config)
    names = [args.task] if args.task != "all" else [
        "pick_up_red_pepper", "pick_up_light_blue_block", "pick_up_red_block",
        "pick_up_smallest_block", "pick_up_largest_block",
        "place_red_pepper_in_ring",
    ]
    for name in names:
        if name not in raw["tasks"]:
            raise ValueError(f"Unknown task {name!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"status": "starting", "oracle": "joint_waypoint_v1", "tasks": []}
    planned_total = 0
    for name in names:
        count = int(args.episodes) if args.episodes is not None else int(raw["tasks"][name].get("collection", {}).get("target_episodes", 0))
        planned_total += max(0, count)
    if args.plan_only:
        manifest["status"] = "planned"
        manifest["total_episodes"] = planned_total
        manifest["prompts"] = [str(raw["tasks"][name]["prompt"]) for name in names]
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return 0

    # Closing SimulationApp terminates Isaac's bundled Python.  For an all-task
    # run, keep this parent free of SimulationApp and launch one child process
    # per task.  Each child writes its task manifest before closing Isaac.
    if args.task == "all":
        for task_index, name in enumerate(names):
            count = (
                int(args.episodes)
                if args.episodes is not None
                else int(raw["tasks"][name]["collection"]["target_episodes"])
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--task", name,
                "--episodes", str(count),
                "--seed", str(args.seed + task_index * 10_000),
                "--output-dir", str(args.output_dir),
                "--task-config", str(args.task_config),
                "--max-steps", str(args.max_steps),
                "--action-hz", str(args.action_hz),
                "--camera-backend", args.camera_backend,
                "--headless" if args.headless else "--no-headless",
                "--record" if args.record else "--no-record",
            ]
            if args.resume:
                command.append("--resume")
            if args.require_success:
                command.extend(
                    [
                        "--require-success",
                        "--max-attempts-per-episode",
                        str(args.max_attempts_per_episode),
                    ]
                )
            child_env = os.environ.copy()
            child_env["ISAAC_ORACLE_CHILD"] = "1"
            completed = subprocess.run(command, cwd=ROOT, env=child_env)
            task_manifest_path = args.output_dir / f"task_manifest_{name}.json"
            if completed.returncode != 0 or not task_manifest_path.is_file():
                manifest["status"] = "failed"
                manifest["failed_task"] = name
                write_json(args.output_dir / "manifest.json", manifest)
                return completed.returncode or 2
            task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
            manifest["tasks"].extend(task_manifest["episodes"])
        manifest["status"] = "completed"
        manifest["total_episodes"] = len(manifest["tasks"])
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return 0

    name = names[0]
    count = int(args.episodes) if args.episodes is not None else int(raw["tasks"][name].get("collection", {}).get("target_episodes", 0))
    prompt = str(raw["tasks"][name]["prompt"])
    task_rows: list[dict[str, Any]] = []
    with IsaacEnvironment(
        task_config_path=args.task_config,
        task_name=name,
        prompt=prompt,
        headless=args.headless,
    ) as environment:
        for episode_index in range(count):
            episode_dir = args.output_dir / name / f"episode_{episode_index:04d}"
            existing = _existing_success(episode_dir, name) if args.resume else None
            if existing is not None:
                result = existing
                print(
                    f"ORACLE_RESUME task={name} episode={episode_index} "
                    f"seed={result.get('seed')} status=kept"
                )
            else:
                if episode_dir.exists():
                    if not args.resume:
                        raise FileExistsError(
                            f"{episode_dir} already exists; use --resume to reuse it"
                        )
                    old_seed = args.seed + episode_index
                    try:
                        old_metadata = json.loads(
                            (episode_dir / "episode.json").read_text(encoding="utf-8")
                        )
                        old_seed = int(old_metadata.get("seed", old_seed))
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                    archived = _archive_rejected(
                        episode_dir,
                        args.output_dir,
                        name,
                        seed=old_seed,
                        attempt=0,
                    )
                    print(
                        f"ORACLE_REJECT task={name} episode={episode_index} "
                        f"path={archived}"
                    )

                attempts = args.max_attempts_per_episode if args.require_success else 1
                result = {}
                for attempt in range(1, attempts + 1):
                    attempt_seed = (
                        args.seed + episode_index + (attempt - 1) * 1_000_000
                    )
                    result = collect_episode(
                        environment,
                        task=name,
                        prompt=prompt,
                        seed=attempt_seed,
                        episode_dir=episode_dir,
                        max_steps=args.max_steps,
                        action_hz=args.action_hz,
                        record=args.record,
                    )
                    if result["task_success"] or not args.require_success:
                        break
                    archived = _archive_rejected(
                        episode_dir,
                        args.output_dir,
                        name,
                        seed=attempt_seed,
                        attempt=attempt,
                    )
                    print(
                        f"ORACLE_REJECT task={name} episode={episode_index} "
                        f"seed={attempt_seed} path={archived}"
                    )
                if args.require_success and result.get("task_success") is not True:
                    raise RuntimeError(
                        f"Could not collect successful {name} episode "
                        f"{episode_index} after {attempts} attempts"
                    )
            task_rows.append({"task": name, "episode": episode_index, "path": str(episode_dir), "steps": result["steps"], "task_success": result["task_success"]})
        task_manifest = {"status": "completed", "task": name, "total_episodes": len(task_rows), "episodes": task_rows}
        write_json(args.output_dir / f"task_manifest_{name}.json", task_manifest)
        if os.environ.get("ISAAC_ORACLE_CHILD") != "1":
            write_json(args.output_dir / "manifest.json", {"status": "completed", "oracle": "joint_waypoint_v1", "total_episodes": len(task_rows), "tasks": task_rows})
        print(json.dumps(task_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
