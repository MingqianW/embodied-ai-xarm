"""Collect the requested 200 MuJoCo episodes in the real xArm raw format."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.data_collection.oracle_controller import (
    OracleConfig,
    PlaceOracleConfig,
    PlaceRedPepperOracleController,
    ScriptedOracleController,
)
from sim_mujoco.data_collection.real_raw_recorder import (
    RealRawEpisodeRecorder,
)
from sim_mujoco.data_collection.task_success import (
    accepted_oracle_episode,
    simulation_is_finite,
    update_task_success,
)
from sim_mujoco.environment import MuJoCoEnvironment
from sim_mujoco.paths import mujoco_dataset_root


@dataclass(frozen=True)
class CollectionTask:
    task: str
    raw_task: str
    episodes: int
    clean_episodes: int
    distractor_episodes: int
    seed_start: int


COLLECTION_PLAN = (
    CollectionTask(
        "red_pepper",
        "pick_up_the_red_pepper",
        50,
        50,
        0,
        0,
    ),
    CollectionTask(
        "blue_block",
        "pick_up_the_blue_block",
        25,
        15,
        10,
        10_000,
    ),
    CollectionTask(
        "red_block",
        "pick_up_the_red_block",
        25,
        15,
        10,
        20_000,
    ),
    CollectionTask(
        "smallest_block",
        "pick_up_the_smallest_block",
        25,
        15,
        10,
        30_000,
    ),
    CollectionTask(
        "largest_block",
        "pick up the largest block",
        25,
        15,
        10,
        40_000,
    ),
    CollectionTask(
        "place_red_pepper_in_ring",
        "place_the_red_pepper_in_the_ring",
        50,
        30,
        20,
        50_000,
    ),
)
DEFAULT_OUTPUT = mujoco_dataset_root() / "xarm_mujoco_200"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rename_with_retry(source: Path, destination: Path) -> None:
    """Tolerate brief Windows file-indexer/antivirus directory locks."""

    last_error: PermissionError | None = None
    for _ in range(20):
        try:
            source.rename(destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.25)
    assert last_error is not None
    raise last_error


def _selected_plan(args: argparse.Namespace) -> tuple[CollectionTask, ...]:
    selected = set(args.task or ())
    plan = tuple(
        item for item in COLLECTION_PLAN if not selected or item.task in selected
    )
    if not plan:
        raise ValueError("No collection tasks selected")
    if args.limit_per_task is None:
        return plan
    if args.limit_per_task < 1:
        raise ValueError("--limit-per-task must be positive")
    limited = []
    for item in plan:
        count = min(item.episodes, int(args.limit_per_task))
        if args.scene_variant == "distractors":
            clean = 0
            distractors = count
        else:
            clean = count
            distractors = 0
        limited.append(
            CollectionTask(
                task=item.task,
                raw_task=item.raw_task,
                episodes=count,
                clean_episodes=clean,
                distractor_episodes=distractors,
                seed_start=item.seed_start,
            )
        )
    return tuple(limited)


def _run_config(
    args: argparse.Namespace,
    plan: tuple[CollectionTask, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "xarm_real_raw_compatible_collection_v1",
        "action_hz": 10,
        "object_xy_range_m": float(args.object_xy_range),
        "object_yaw_range_deg": float(args.object_yaw_range_deg),
        "joint_noise_rad": float(args.joint_noise),
        "max_attempts_per_episode": int(args.max_attempts_per_episode),
        "scene_variant_override": args.scene_variant,
        "plan": [asdict(item) for item in plan],
        "total_target_episodes": sum(item.episodes for item in plan),
    }


def _load_or_initialize(
    output_dir: Path,
    run_config: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    config_path = output_dir / "run_config.json"
    manifest_path = output_dir / "collection_manifest.json"
    if resume:
        if not config_path.is_file() or not manifest_path.is_file():
            raise ValueError(
                "--resume requires run_config.json and collection_manifest.json"
            )
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise ValueError("Resume arguments do not match the saved run config")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output is non-empty; use --resume or another directory: {output_dir}"
        )
    for path in (
        output_dir / "raw",
        output_dir / ".staging",
        output_dir / "failed_attempts",
    ):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, run_config)
    manifest = {
        "schema_version": run_config["schema_version"],
        "completed": [],
        "failed_attempts": [],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _scene_variant(
    task: CollectionTask,
    episode_index: int,
    override: str | None,
) -> str:
    if override is not None:
        return override
    return "clean" if episode_index < task.clean_episodes else "distractors"


def _controller(environment: MuJoCoEnvironment, task: str):
    runtime = environment.task_runtime
    assert runtime is not None
    if runtime.spec["success"]["type"] == "place_in_ring":
        return PlaceRedPepperOracleController(
            environment,
            PlaceOracleConfig(task=task, action_dt_s=0.1),
        )
    return ScriptedOracleController(
        environment,
        OracleConfig(task=task, action_dt_s=0.1),
    )


def _record_attempt(
    environment: MuJoCoEnvironment,
    *,
    task: CollectionTask,
    episode_index: int,
    scene_variant: str,
    seed: int,
    staging_dir: Path,
) -> tuple[bool, dict[str, Any]]:
    environment.scene_variant = scene_variant
    environment.reset(seed=seed)
    controller = _controller(environment, task.task)
    recorder = RealRawEpisodeRecorder(
        staging_dir,
        task=task.raw_task,
        episode_index=episode_index,
        seed=seed,
        scene_variant=scene_variant,
        environment=environment,
    )
    runtime = environment.task_runtime
    assert runtime is not None
    metrics = runtime.metrics()
    last_action = None
    while not controller.terminal:
        action = controller.next_action()
        if action is None:
            break
        recorder.record_observation(gripper_target=float(action[6]))
        environment.apply_action(action)
        environment.step_physics(0.1)
        metrics = update_task_success(environment)
        controller.notify_post_step(
            task_metrics=metrics,
            collision=environment.safety_diagnostics()["collision"],
            simulation_finite=simulation_is_finite(environment),
        )
        last_action = action
    recorder.record_observation(
        gripper_target=(
            float(last_action[6]) if last_action is not None else None
        )
    )
    success = accepted_oracle_episode(
        terminal_stage=controller.stage.value,
        task_metrics=metrics,
        failure_reason=controller.failure_reason,
    )
    meta = recorder.finalize(
        success=success,
        failure_reason=controller.failure_reason,
        initial_conditions=environment.initial_conditions,
        task_metrics=metrics,
        oracle_transitions=controller.transition_log(),
        oracle_plan=controller.plan.to_json(),
    )
    return success, meta


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_attempts_per_episode < 1:
        raise ValueError("--max-attempts-per-episode must be positive")
    plan = _selected_plan(args)
    output_dir = args.output_dir.resolve()
    run_config = _run_config(args, plan)
    manifest = _load_or_initialize(
        output_dir,
        run_config,
        resume=args.resume,
    )
    completed = list(manifest.get("completed") or [])
    failed_attempts = list(manifest.get("failed_attempts") or [])
    completed_keys = {
        (str(item["task"]), int(item["episode_index"])) for item in completed
    }

    for task in plan:
        raw_task_dir = output_dir / "raw" / task.raw_task
        raw_task_dir.mkdir(parents=True, exist_ok=True)
        with MuJoCoEnvironment(
            task=task.task,
            prompt=task.raw_task,
            object_xy_range=args.object_xy_range,
            object_yaw_range_deg=args.object_yaw_range_deg,
            joint_noise=args.joint_noise,
        ) as environment:
            for episode_index in range(task.episodes):
                key = (task.task, episode_index)
                if key in completed_keys:
                    continue
                variant = _scene_variant(
                    task,
                    episode_index,
                    args.scene_variant,
                )
                episode_succeeded = False
                for retry in range(args.max_attempts_per_episode):
                    seed = task.seed_start + episode_index * 100 + retry
                    staging_dir = (
                        output_dir
                        / ".staging"
                        / task.task
                        / f"episode_{episode_index:03d}_attempt_{retry:02d}"
                    )
                    if staging_dir.exists():
                        raise FileExistsError(
                            f"Stale staging directory blocks resume: {staging_dir}"
                        )
                    try:
                        success, meta = _record_attempt(
                            environment,
                            task=task,
                            episode_index=episode_index,
                            scene_variant=variant,
                            seed=seed,
                            staging_dir=staging_dir,
                        )
                        failure_reason = meta["simulation"]["failure_reason"]
                        rows = int(meta["simulation"]["robot_log_rows"])
                    except Exception as exc:
                        staging_dir.mkdir(parents=True, exist_ok=True)
                        success = False
                        failure_reason = (
                            f"exception:{type(exc).__name__}:{exc}"
                        )
                        rows = 0
                        _write_json(
                            staging_dir / "failure.json",
                            {
                                "task": task.task,
                                "episode_index": episode_index,
                                "retry": retry,
                                "seed": seed,
                                "failure_reason": failure_reason,
                            },
                        )
                    record = {
                        "task": task.task,
                        "raw_task": task.raw_task,
                        "episode_index": episode_index,
                        "scene_variant": variant,
                        "retry": retry,
                        "seed": seed,
                        "success": bool(success),
                        "robot_log_rows": rows,
                        "failure_reason": failure_reason,
                    }
                    if success:
                        destination = (
                            raw_task_dir / f"episode_{episode_index:03d}"
                        )
                        if destination.exists():
                            raise FileExistsError(
                                f"Refusing to overwrite episode: {destination}"
                            )
                        _rename_with_retry(staging_dir, destination)
                        record["path"] = destination.relative_to(
                            output_dir
                        ).as_posix()
                        completed.append(record)
                        completed_keys.add(key)
                        episode_succeeded = True
                    else:
                        destination = (
                            output_dir
                            / "failed_attempts"
                            / task.task
                            / f"episode_{episode_index:03d}_attempt_{retry:02d}"
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            raise FileExistsError(
                                f"Refusing to overwrite failure: {destination}"
                            )
                        _rename_with_retry(staging_dir, destination)
                        record["path"] = destination.relative_to(
                            output_dir
                        ).as_posix()
                        failed_attempts.append(record)
                    manifest.update(
                        {
                            "completed": completed,
                            "failed_attempts": failed_attempts,
                            "updated_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
                    _write_json(
                        output_dir / "collection_manifest.json",
                        manifest,
                    )
                    print(
                        f"task={task.task} episode={episode_index:03d} "
                        f"variant={variant} retry={retry} seed={seed} "
                        f"success={success} rows={rows} "
                        f"failure={failure_reason}"
                    )
                    if success:
                        break
                if not episode_succeeded:
                    raise RuntimeError(
                        f"Failed to collect {task.task} episode "
                        f"{episode_index} after "
                        f"{args.max_attempts_per_episode} attempts"
                    )

    counts = {
        item.task: sum(1 for record in completed if record["task"] == item.task)
        for item in plan
    }
    result = {
        "output_dir": str(output_dir),
        "raw_root": str(output_dir / "raw"),
        "target_episodes": sum(item.episodes for item in plan),
        "completed_episodes": sum(counts.values()),
        "failed_attempts": len(failed_attempts),
        "counts": counts,
        "converted": False,
        "uploaded": False,
    }
    _write_json(output_dir / "collection_summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--task",
        action="append",
        choices=tuple(item.task for item in COLLECTION_PLAN),
        help="Collect only this task. Repeat for multiple tasks.",
    )
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument(
        "--scene-variant",
        choices=("clean", "distractors"),
        help="Override the planned clean/distractor split (useful for smoke tests).",
    )
    parser.add_argument("--object-xy-range", type=float, default=0.02)
    parser.add_argument("--object-yaw-range-deg", type=float, default=10.0)
    parser.add_argument("--joint-noise", type=float, default=0.005)
    parser.add_argument("--max-attempts-per-episode", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action="store_true")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
