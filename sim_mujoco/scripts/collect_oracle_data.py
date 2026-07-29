from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.data_collection.episode_recorder import (
    EpisodeRecorder,
    EpisodeRecorderConfig,
    RAW_SCHEMA_VERSION,
    REAL_TRAINING_PROMPT,
)
from sim_mujoco.paths import mujoco_dataset_root
from sim_mujoco.data_collection.oracle_controller import (
    OracleConfig,
    ScriptedOracleController,
)
from sim_mujoco.data_collection.task_success import (
    accepted_oracle_episode,
    simulation_is_finite,
    update_task_success,
)
from sim_mujoco.environment import MuJoCoEnvironment


DEFAULT_OUTPUT = mujoco_dataset_root() / "xarm_mujoco_red_block_raw"
RUN_CONFIG_NAME = "run_config.json"
MANIFEST_NAME = "manifest.json"
RUN_SCHEMA_VERSION = "xarm_mujoco_collection_run_v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "raw_episode_schema_version": RAW_SCHEMA_VERSION,
        "task": args.task,
        "prompt": REAL_TRAINING_PROMPT,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "action_hz": args.action_hz,
        "action_dt_s": 1.0 / float(args.action_hz),
        "object_xy_range_m": args.object_xy_range,
        "object_yaw_range_deg": args.object_yaw_range_deg,
        "joint_noise_rad": args.joint_noise,
        "save_only_success": bool(args.save_only_success),
        "record_video": bool(args.record_video),
        "video_every": args.video_every,
        "max_attempts": args.max_attempts,
        "allow_partial": bool(getattr(args, "allow_partial", False)),
        "headless": bool(args.headless),
    }


def _initial_manifest(run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_config_sha256": _config_hash(run_config),
        "completed_episodes": [],
        "failed_attempts": [],
        "next_attempt_index": 0,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def should_record_success_video(
    *,
    record_video: bool,
    prospective_episode_index: int,
    video_every: int,
) -> bool:
    """Use human-facing one-based cadence: 5 means episodes 5, 10, 15, ..."""

    if video_every < 1:
        raise ValueError("video_every must be positive")
    return bool(
        record_video
        and (prospective_episode_index + 1) % video_every == 0
    )


def load_or_initialize_run(
    output_dir: Path,
    run_config: dict[str, Any],
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = output_dir.resolve()
    config_path = output_dir / RUN_CONFIG_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if resume:
        if not config_path.exists() or not manifest_path.exists():
            raise ValueError(
                "--resume requires existing run_config.json and manifest.json"
            )
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            changed = sorted(
                key
                for key in set(existing_config) | set(run_config)
                if existing_config.get(key) != run_config.get(key)
            )
            raise ValueError(
                "Resume arguments do not match the original run configuration. "
                f"Changed fields: {changed}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_config_sha256") != _config_hash(existing_config):
            raise ValueError("Run manifest/config hash mismatch")
        return existing_config, manifest

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; use --resume or choose another "
            f"directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("episodes", "failed_attempts", ".staging"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    manifest = _initial_manifest(run_config)
    _write_json(config_path, run_config)
    _write_json(manifest_path, manifest)
    return run_config, manifest


def _record_attempt(
    environment: MuJoCoEnvironment,
    *,
    seed: int,
    attempt_dir: Path,
    args: argparse.Namespace,
    record_video: bool,
) -> tuple[bool, dict[str, Any]]:
    environment.reset(seed=seed)
    initial_conditions = dict(environment.initial_conditions)
    initial_conditions["scene_prompt"] = initial_conditions.get("prompt")
    initial_conditions["prompt"] = REAL_TRAINING_PROMPT
    runtime = environment.task_runtime
    assert runtime is not None
    controller = ScriptedOracleController(
        environment,
        OracleConfig(action_dt_s=1.0 / float(args.action_hz)),
    )
    recorder = EpisodeRecorder(
        EpisodeRecorderConfig(
            output_dir=attempt_dir,
            task=args.task,
            prompt=REAL_TRAINING_PROMPT,
            seed=seed,
            fps=args.action_hz,
            record_video=record_video,
        ),
        environment,
    )
    task_metrics = runtime.metrics()
    while not controller.terminal:
        action = controller.next_action()
        if action is None:
            break
        recorder.record_pre_action(
            action=action,
            oracle_stage=controller.stage.value,
        )
        environment.apply_action(action)
        environment.step_physics(controller.config.action_dt_s)
        task_metrics = update_task_success(environment)
        collision = environment.safety_diagnostics()["collision"]
        controller.notify_post_step(
            task_metrics=task_metrics,
            collision=collision,
            simulation_finite=simulation_is_finite(environment),
        )

    success = accepted_oracle_episode(
        terminal_stage=controller.stage.value,
        task_metrics=task_metrics,
        failure_reason=controller.failure_reason,
    )
    metadata = recorder.finalize(
        success=success,
        failure_reason=controller.failure_reason,
        task_metrics=task_metrics,
        initial_conditions=initial_conditions,
        randomization={
            "object_xy_range_m": float(args.object_xy_range),
            "object_yaw_range_deg": float(args.object_yaw_range_deg),
            "joint_noise_rad": float(args.joint_noise),
        },
        transitions=controller.transition_log(),
        oracle_plan=controller.plan.to_json(),
    )
    return success, metadata


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.task != "red_block":
        raise ValueError("The initial collection pipeline supports only red_block")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.action_hz != 10:
        raise ValueError(
            "The audited real dataset uses 10 Hz; initial red_block collection "
            "requires --action-hz 10"
        )
    if args.video_every < 1:
        raise ValueError("--video-every must be positive")
    if args.max_attempts is None:
        args.max_attempts = max(args.episodes, args.episodes * 3)
    if args.max_attempts < args.episodes:
        raise ValueError("--max-attempts cannot be smaller than --episodes")

    output_dir = args.output_dir.resolve()
    run_config = _run_config_from_args(args)
    _, manifest = load_or_initialize_run(
        output_dir,
        run_config,
        resume=args.resume,
    )
    completed = list(manifest.get("completed_episodes") or [])
    failed = list(manifest.get("failed_attempts") or [])
    attempt_index = int(manifest.get("next_attempt_index", 0))

    with MuJoCoEnvironment(
        task=args.task,
        prompt=REAL_TRAINING_PROMPT,
        object_xy_range=args.object_xy_range,
        object_yaw_range_deg=args.object_yaw_range_deg,
        joint_noise=args.joint_noise,
    ) as environment:
        physics_steps = round(
            (1.0 / args.action_hz)
            / float(environment.context.model.opt.timestep)
        )
        represented_dt = (
            physics_steps * float(environment.context.model.opt.timestep)
        )
        if abs(represented_dt - 1.0 / args.action_hz) > 1e-12:
            raise ValueError(
                "action_dt is not an integer multiple of the physics timestep"
            )

        while (
            len(completed) < args.episodes
            and attempt_index < args.max_attempts
        ):
            seed = args.seed_start + attempt_index
            prospective_episode_index = len(completed)
            record_video = should_record_success_video(
                record_video=bool(args.record_video),
                prospective_episode_index=prospective_episode_index,
                video_every=args.video_every,
            )
            staging_dir = (
                output_dir
                / ".staging"
                / f"attempt_{attempt_index:06d}_seed_{seed}"
            )
            if staging_dir.exists():
                raise FileExistsError(
                    f"Stale staging directory blocks safe resume: {staging_dir}"
                )
            try:
                success, metadata = _record_attempt(
                    environment,
                    seed=seed,
                    attempt_dir=staging_dir,
                    args=args,
                    record_video=record_video,
                )
                record = {
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "success": success,
                    "number_of_samples": metadata["number_of_samples"],
                    "failure_reason": metadata["failure_reason"],
                    "terminal_task_metrics": metadata["task_metrics"],
                }
            except Exception as exc:
                staging_dir.mkdir(parents=True, exist_ok=True)
                record = {
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "success": False,
                    "number_of_samples": 0,
                    "failure_reason": f"exception:{type(exc).__name__}:{exc}",
                    "terminal_task_metrics": None,
                }
                _write_json(staging_dir / "failure.json", record)
                success = False

            if success:
                episode_index = len(completed)
                destination = (
                    output_dir
                    / "episodes"
                    / f"episode_{episode_index:06d}"
                )
                if destination.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite completed episode: {destination}"
                    )
                staging_dir.rename(destination)
                record["episode_index"] = episode_index
                record["path"] = destination.relative_to(output_dir).as_posix()
                completed.append(record)
            else:
                destination = (
                    output_dir
                    / "failed_attempts"
                    / f"attempt_{attempt_index:06d}_seed_{seed}"
                )
                if destination.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite failed attempt: {destination}"
                    )
                staging_dir.rename(destination)
                record["path"] = destination.relative_to(output_dir).as_posix()
                failed.append(record)

            attempt_index += 1
            manifest.update(
                {
                    "completed_episodes": completed,
                    "failed_attempts": failed,
                    "next_attempt_index": attempt_index,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json(output_dir / MANIFEST_NAME, manifest)
            print(
                f"attempt={record['attempt_index']} seed={seed} "
                f"success={success} completed={len(completed)}/"
                f"{args.episodes} samples={record['number_of_samples']} "
                f"failure={record['failure_reason']}"
            )

    result = {
        "output_dir": str(output_dir),
        "requested_successful_episodes": args.episodes,
        "completed_successful_episodes": len(completed),
        "failed_attempts": len(failed),
        "attempts": attempt_index,
        "complete": len(completed) >= args.episodes,
    }
    print(json.dumps(result, indent=2))
    if not result["complete"] and not getattr(args, "allow_partial", False):
        raise RuntimeError(
            f"Collection stopped at max attempts with "
            f"{len(completed)}/{args.episodes} successes"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--task", default="red_block")
    parser.add_argument("--action-hz", type=int, default=10)
    parser.add_argument("--object-xy-range", type=float, default=0.0)
    parser.add_argument("--object-yaw-range-deg", type=float, default=0.0)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument("--save-only-success", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Return successfully after --max-attempts even if fewer than "
            "--episodes successes were accepted. Intended for fixed-attempt "
            "validation stages; successful episodes remain the only training data."
        ),
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
