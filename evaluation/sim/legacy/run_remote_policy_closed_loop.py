from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
from PIL import Image


from policy_runtime.remote_policy_client import (  # noqa: E402
    PolicyTimeoutError,
    RemotePolicyClient,
    RemotePolicyConfig,
)
from policy_runtime.action_decoder import DEFAULT_ACTION_HORIZON  # noqa: E402
from policy_runtime.safety import SafetyConfig, validate_action_chunk  # noqa: E402
from simulation.robot.control import (  # noqa: E402
    apply_safe_control_target,
    compute_safe_control_target,
    validate_policy_actions,
)
from simulation.physics.collision import collision_diagnostics  # noqa: E402
from evaluation.sim.video import VideoRecorder  # noqa: E402
from evaluation.sim.legacy.remote_policy_evaluation import write_json  # noqa: E402
from simulation.resources import DEFAULT_CAMERA_CONFIG_PATH  # noqa: E402
from simulation.resources import DEFAULT_MODEL_PATH  # noqa: E402
from simulation.robot.model import arm_actuator_ctrl_limits  # noqa: E402
from simulation.robot.model import arm_joint_limits  # noqa: E402
from simulation.observation.policy import build_policy_observation  # noqa: E402
from simulation.observation.state import get_robot_state  # noqa: E402
from simulation.robot.gripper import actuator_ctrl_from_raw_hardware  # noqa: E402
from policy_runtime.image_preprocessing import image_diagnostics  # noqa: E402
from simulation.runtime import initialize_scene  # noqa: E402
from simulation.runtime import load_simulation  # noqa: E402
from simulation.scene import (  # noqa: E402
    configure_task_scene,
    resolve_task,
    task_names,
)
from simulation.resources import output_root  # noqa: E402


DEFAULT_OUTPUT_DIR = output_root() / "remote_policy_closed_loop"
MAX_EXECUTE_CHUNK_STEPS = DEFAULT_ACTION_HORIZON


@dataclass
class EpisodeConfig:
    host: str = "127.0.0.1"
    port: int = 18000
    task: str = "red_block"
    prompt: str | None = None
    max_policy_steps: int = 20
    execute_chunk_steps: int = 1
    max_joint_step: float = 0.05
    control_duration: float = 0.02
    headless: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int | None = None
    object_xy_range: float = 0.0
    object_yaw_range_deg: float = 0.0
    joint_noise: float = 0.0
    record_video: bool = False
    video_fps: int = 30
    run_preflight: bool = True
    timeout: float = 120.0


def create_policy_client(
    host: str, port: int, timeout: float = 10.0
) -> RemotePolicyClient:
    client = RemotePolicyClient(
        RemotePolicyConfig(
            host=host,
            port=port,
            connect_timeout_s=timeout,
            inference_timeout_s=timeout,
        )
    )
    client.connect()
    return client


def validate_execute_chunk_steps(value: int) -> int:
    steps = int(value)
    if steps < 1 or steps > MAX_EXECUTE_CHUNK_STEPS:
        raise ValueError(
            "execute_chunk_steps must be between 1 and "
            f"{MAX_EXECUTE_CHUNK_STEPS} (the policy action horizon)"
        )
    return steps


def preflight(
    *,
    host: str,
    port: int,
    task: str,
    prompt: str,
    output_dir: Path,
    timeout: float,
) -> bool:
    rows: list[tuple[str, bool, str]] = []

    def check(label: str, fn):
        try:
            detail = fn()
            rows.append((label, True, str(detail)))
            return detail
        except Exception as exc:
            rows.append((label, False, str(exc)))
            return None

    check(
        "XML scene exists",
        lambda: DEFAULT_MODEL_PATH
        if DEFAULT_MODEL_PATH.is_file()
        else (_ for _ in ()).throw(FileNotFoundError(DEFAULT_MODEL_PATH)),
    )
    check(
        "camera calibration config exists",
        lambda: DEFAULT_CAMERA_CONFIG_PATH
        if DEFAULT_CAMERA_CONFIG_PATH.is_file()
        else (_ for _ in ()).throw(FileNotFoundError(DEFAULT_CAMERA_CONFIG_PATH)),
    )
    check("task scene config resolves", lambda: resolve_task(task)[0])
    policy = check(
        "remote WebSocket policy connection succeeds",
        lambda: create_policy_client(host, port, timeout),
    )
    if policy is not None:
        rows[-1] = (rows[-1][0], rows[-1][1], "connected")
    context = check("metadata/model can be read", lambda: load_simulation())
    if context is not None:
        rows[-1] = (rows[-1][0], rows[-1][1], f"timestep={context.model.opt.timestep}")
    observation = None
    actions = None
    if context is not None:
        initialize_scene(context.model, context.data, settle_steps=0)
        task_setup = check(
            "task scene can be initialized",
            lambda: configure_task_scene(
                context.model,
                context.data,
                task=task,
                seed=0,
                object_xy_range=0.0,
                object_yaw_range_deg=0.0,
                joint_noise=0.0,
                settle_steps=5,
            ),
        )
        if task_setup is not None:
            runtime, initial = task_setup
            rows[-1] = (
                rows[-1][0],
                rows[-1][1],
                f"{runtime.task_name}: active={initial['active_bodies']}, "
                f"target={runtime.target_body}",
            )
            check(
                "task starts without forbidden robot collision",
                lambda: (
                    "ok"
                    if not collision_diagnostics(context.model, context.data)[
                        "forbidden"
                    ]
                    else (_ for _ in ()).throw(
                        RuntimeError(
                            collision_diagnostics(context.model, context.data)[
                                "forbidden_contacts"
                            ]
                        )
                    )
                ),
            )
        observation = check(
            "one observation can be constructed",
            lambda: build_policy_observation(
                context.model, context.data, context.renderer, context.config, prompt
            ),
        )
        if observation is not None and task_setup is not None:
            task_setup[0].adjust_observation(observation)
        if observation is not None:
            rows[-1] = (rows[-1][0], rows[-1][1], "ok")
    if context is not None and observation is not None:
        check(
            "images have correct shape and dtype",
            lambda: "ok"
            if observation["observation/image"].shape == (224, 224, 3)
            and observation["observation/image"].dtype == np.uint8
            and observation["observation/wrist_image"].shape == (224, 224, 3)
            and observation["observation/wrist_image"].dtype == np.uint8
            else (_ for _ in ()).throw(ValueError("bad image shape/dtype")),
        )
        check(
            "state has correct shape and dtype",
            lambda: "ok"
            if observation["observation/state"].shape == (7,)
            and observation["observation/state"].dtype == np.float32
            else (_ for _ in ()).throw(ValueError("bad state shape/dtype")),
        )
        check(
            "current state is finite",
            lambda: "ok"
            if np.isfinite(observation["observation/state"]).all()
            else (_ for _ in ()).throw(ValueError("state has NaN/Inf")),
        )
        limits = arm_joint_limits(context.model)
        check(
            "current state lies inside valid limits",
            lambda: "ok"
            if np.all(observation["observation/state"][:6] >= limits[:, 0])
            and np.all(observation["observation/state"][:6] <= limits[:, 1])
            else (_ for _ in ()).throw(ValueError("state outside limits")),
        )
    if policy is not None and observation is not None:
        result = check("one inference succeeds", lambda: policy.infer(observation))
        if result is not None:
            rows[-1] = (rows[-1][0], rows[-1][1], "ok")
            actions = check(
                "action shape is (10, 7)",
                lambda: validate_policy_actions(
                    np.asarray(result["actions"], dtype=np.float32)
                ),
            )
            if actions is not None:
                rows[-1] = (rows[-1][0], rows[-1][1], "ok")
            check(
                "action values are finite",
                lambda: "ok"
                if actions is not None and np.isfinite(actions).all()
                else (_ for _ in ()).throw(ValueError("actions have NaN/Inf")),
            )
    check(
        "output directory is writable",
        lambda: (
            output_dir.mkdir(parents=True, exist_ok=True),
            write_json(output_dir / "preflight_write_test.json", {"ok": True}),
            "ok",
        )[2],
    )

    print("Preflight")
    print("---------")
    all_passed = True
    for label, passed, detail in rows:
        all_passed = all_passed and passed
        print(f"{'PASS' if passed else 'FAIL'} {label}: {detail}")
    if context is not None:
        context.close()
    if policy is not None and hasattr(policy, "close"):
        policy.close()
    return all_passed


def save_observation_images(
    output_dir: Path, prefix: str, observation: dict[str, Any]
) -> None:
    Image.fromarray(observation["observation/image"]).save(
        output_dir / f"{prefix}_base.png"
    )
    Image.fromarray(observation["observation/wrist_image"]).save(
        output_dir / f"{prefix}_wrist.png"
    )


def run_episode(
    config: EpisodeConfig,
    *,
    policy: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    validate_execute_chunk_steps(config.execute_chunk_steps)
    if config.max_policy_steps < 1:
        raise ValueError("max_policy_steps must be at least 1")
    task_name, task_spec = resolve_task(config.task)
    prompt = str(config.prompt or task_spec["prompt"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.run_preflight:
        passed = preflight(
            host=config.host,
            port=config.port,
            task=task_name,
            prompt=prompt,
            output_dir=config.output_dir,
            timeout=config.timeout,
        )
        if not passed:
            raise RuntimeError("Preflight failed; refusing to begin control")

    owns_policy = policy is None
    if policy is None:
        policy = create_policy_client(config.host, config.port, config.timeout)

    context = load_simulation()
    viewer = None
    recorder: VideoRecorder | None = None
    start_wall = time.perf_counter()
    step_records: list[dict[str, Any]] = []
    termination_reason = "max_policy_steps"
    policy_error: str | None = None
    initial_conditions: dict[str, Any] = {}
    task_runtime = None
    final_task_metrics: dict[str, Any] | None = None
    try:
        initialize_scene(context.model, context.data, settle_steps=0)
        task_runtime, initial_conditions = configure_task_scene(
            context.model,
            context.data,
            task=task_name,
            seed=config.seed,
            object_xy_range=config.object_xy_range,
            object_yaw_range_deg=config.object_yaw_range_deg,
            joint_noise=config.joint_noise,
        )

        initial_observation = build_policy_observation(
            context.model,
            context.data,
            context.renderer,
            context.config,
            prompt,
        )
        task_runtime.adjust_observation(initial_observation)
        save_observation_images(config.output_dir, "initial", initial_observation)

        if config.record_video:
            recorder = VideoRecorder(config.output_dir, fps=config.video_fps)
            recorder.maybe_record(context)

        if not config.headless:
            import mujoco.viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(context.model, context.data)

        for step in range(config.max_policy_steps):
            if viewer is not None and not viewer.is_running():
                termination_reason = "viewer_closed"
                break

            step_dir = config.output_dir / f"policy_step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            observation = build_policy_observation(
                context.model,
                context.data,
                context.renderer,
                context.config,
                prompt,
            )
            task_runtime.adjust_observation(observation)
            Image.fromarray(observation["observation/image"]).save(
                step_dir / "base.png"
            )
            Image.fromarray(observation["observation/wrist_image"]).save(
                step_dir / "wrist.png"
            )

            infer_start = time.perf_counter()
            try:
                result = policy.infer(observation)
            except PolicyTimeoutError as exc:
                current_state = get_robot_state(
                    context.model, context.data, context.config
                )
                context.data.ctrl[:6] = current_state[:6]
                context.data.ctrl[6] = actuator_ctrl_from_raw_hardware(
                    float(current_state[6]),
                    context.config,
                )
                termination_reason = "policy_timeout"
                policy_error = str(exc)
                write_json(
                    step_dir / "diagnostics.json",
                    {
                        "policy_step": step,
                        "observation_state": observation["observation/state"],
                        "rejected": True,
                        "rejection_reason": "policy_timeout",
                        "error": str(exc),
                    },
                )
                break
            latency = time.perf_counter() - infer_start
            actions = validate_policy_actions(
                np.asarray(result["actions"], dtype=np.float32)
            )
            model_limits = arm_joint_limits(context.model)
            actuator_limits = arm_actuator_ctrl_limits(context.model)
            canonical_limits = np.column_stack(
                (
                    np.maximum(model_limits[:, 0], actuator_limits[:, 0]),
                    np.minimum(model_limits[:, 1], actuator_limits[:, 1]),
                )
            )
            chunk_safety = validate_action_chunk(
                actions,
                observation["observation/state"],
                canonical_limits,
                SafetyConfig(
                    max_joint_delta_rad=config.max_joint_step,
                    reject_if_clip_exceeds_rad=0.25,
                ),
            )
            if not chunk_safety.accepted:
                current_state = get_robot_state(
                    context.model, context.data, context.config
                )
                context.data.ctrl[:6] = current_state[:6]
                context.data.ctrl[6] = actuator_ctrl_from_raw_hardware(
                    float(current_state[6]),
                    context.config,
                )
                termination_reason = "unsafe_action"
                write_json(
                    step_dir / "diagnostics.json",
                    {
                        "policy_step": step,
                        "observation_state": observation["observation/state"],
                        "raw_actions": actions,
                        "chunk_safety": chunk_safety.to_json(),
                        "rejected": True,
                        "rejection_reason": chunk_safety.reason,
                    },
                )
                np.save(step_dir / "actions.npy", actions)
                break

            executed_targets = []
            step_collision = collision_diagnostics(context.model, context.data)
            for action_index in range(config.execute_chunk_steps):
                safe_target = compute_safe_control_target(
                    context.model,
                    context.data,
                    context.config,
                    chunk_safety.actions[action_index],
                    max_joint_step=config.max_joint_step,
                    control_dt_s=config.control_duration,
                )
                executed_targets.append(safe_target)
                task_runtime.release_if_requested(safe_target.gripper_raw_clamped)
                physical_gripper_raw = task_runtime.physical_gripper_raw_target(
                    safe_target.gripper_raw_clamped
                )
                physical_gripper_ctrl = actuator_ctrl_from_raw_hardware(
                    physical_gripper_raw,
                    context.config,
                )
                safe_target.gripper_ctrl_target = physical_gripper_ctrl
                safe_target.ctrl_target[6] = physical_gripper_ctrl
                apply_safe_control_target(context.data, safe_target)
                sim_steps = max(
                    1,
                    int(float(config.control_duration) / context.model.opt.timestep),
                )
                for _ in range(sim_steps):
                    mujoco.mj_step(context.model, context.data)
                    if recorder is not None:
                        recorder.maybe_record(context)
                    step_collision = collision_diagnostics(context.model, context.data)
                    if step_collision["forbidden"]:
                        break
                if step_collision["forbidden"]:
                    break
            if viewer is not None:
                viewer.sync()

            resulting_qpos = np.asarray(context.data.qpos, dtype=np.float32).copy()
            task_metrics = task_runtime.update_success()
            final_task_metrics = task_metrics
            diagnostics = {
                "policy_step": step,
                "observation_state": observation["observation/state"],
                "base_image": image_diagnostics(observation["observation/image"]),
                "wrist_image": image_diagnostics(
                    observation["observation/wrist_image"]
                ),
                "raw_actions": actions,
                "chunk_safety": chunk_safety.to_json(),
                "safe_target": executed_targets[0].to_json(),
                "executed_targets": [target.to_json() for target in executed_targets],
                "executed_action_count": len(executed_targets),
                "resulting_qpos": resulting_qpos,
                "policy_timing": result.get("policy_timing"),
                "server_timing": result.get("server_timing"),
                "total_client_latency_seconds": latency,
                "control_duration_seconds": config.control_duration,
                "task_metrics": task_metrics,
                "collision": step_collision,
            }
            write_json(step_dir / "diagnostics.json", diagnostics)
            np.save(step_dir / "actions.npy", actions)
            step_records.append(diagnostics)

            update = {
                "step": step,
                "max_policy_steps": config.max_policy_steps,
                "ctrl_target": executed_targets[-1].ctrl_target,
                "latency": latency,
                "clip_messages": [
                    message
                    for target in executed_targets
                    for message in target.clip_messages
                ],
            }
            if progress_callback is not None:
                progress_callback(update)
            else:
                print(
                    f"[closed-loop {step + 1}/{config.max_policy_steps}] "
                    f"ctrl={executed_targets[-1].ctrl_target} latency={latency:.3f}s"
                )
                for message in update["clip_messages"]:
                    print(" ", message)
            if step_collision["forbidden"]:
                termination_reason = str(step_collision["termination_reason"])
                break
            if task_metrics["task_success"]:
                termination_reason = "task_success"
                break
    except KeyboardInterrupt:
        termination_reason = "interrupted"
        raise
    except Exception as exc:
        termination_reason = "error"
        policy_error = repr(exc)
        raise
    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()

        try:
            final_observation = build_policy_observation(
                context.model,
                context.data,
                context.renderer,
                context.config,
                prompt,
            )
            if task_runtime is not None:
                task_runtime.adjust_observation(final_observation)
            save_observation_images(config.output_dir, "final", final_observation)
            final_state = final_observation["observation/state"]
        except Exception:
            final_state = None

        video_metadata = {}
        if recorder is not None:
            recorder.validate_outputs()
            video_metadata = recorder.metadata()

        trajectory_path = config.output_dir / "trajectory.npz"
        if step_records:
            np.savez_compressed(
                trajectory_path,
                observation_states=np.asarray(
                    [record["observation_state"] for record in step_records],
                    dtype=np.float32,
                ),
                resulting_qpos=np.asarray(
                    [record["resulting_qpos"] for record in step_records],
                    dtype=np.float32,
                ),
                raw_actions=np.asarray(
                    [record["raw_actions"] for record in step_records], dtype=np.float32
                ),
            )
        else:
            np.savez_compressed(
                trajectory_path, observation_states=np.empty((0, 7), dtype=np.float32)
            )

        wall_time = time.perf_counter() - start_wall
        final_collision = collision_diagnostics(context.model, context.data)
        result_payload = {
            "termination_reason": termination_reason,
            "policy_error": policy_error,
            "task": task_name,
            "prompt": prompt,
            "task_success": bool(
                final_task_metrics and final_task_metrics.get("task_success")
            ),
            "task_metrics": final_task_metrics,
            "collision": final_collision,
            "policy_steps": len(step_records),
            "sim_time": float(context.data.time),
            "wall_time": wall_time,
            "initial_conditions": initial_conditions,
            "final_state": None
            if final_state is None
            else np.asarray(final_state, dtype=np.float32),
            "final_qpos": np.asarray(context.data.qpos, dtype=np.float32).copy(),
            "clipping_count": int(
                sum(bool(record["safe_target"]["clipped"]) for record in step_records)
            ),
            "clip_messages": [
                message
                for record in step_records
                for message in record["safe_target"].get("clip_messages", [])
            ],
            "trajectory_path": trajectory_path,
            **video_metadata,
        }
        write_json(config.output_dir / "result.json", result_payload)
        context.close()
        if owns_policy and hasattr(policy, "close"):
            policy.close()
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--task", choices=task_names(), default="red_block")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-policy-steps", type=int, default=20)
    parser.add_argument(
        "--execute-chunk-steps",
        type=int,
        default=1,
        help=f"Actions executed before re-observation and inference (1-{MAX_EXECUTE_CHUNK_STEPS}).",
    )
    parser.add_argument("--max-joint-step", type=float, default=0.05)
    parser.add_argument("--control-duration", type=float, default=0.02)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    result = run_episode(
        EpisodeConfig(
            host=args.host,
            port=args.port,
            task=args.task,
            prompt=args.prompt,
            max_policy_steps=args.max_policy_steps,
            execute_chunk_steps=args.execute_chunk_steps,
            max_joint_step=args.max_joint_step,
            control_duration=args.control_duration,
            headless=args.headless,
            output_dir=args.output_dir,
            timeout=args.timeout,
            run_preflight=True,
        )
    )
    print("termination_reason:", result["termination_reason"])
    print("policy_steps:", result["policy_steps"])
    print("sim_time:", result["sim_time"])
    print("wall_time:", result["wall_time"])


if __name__ == "__main__":
    main()
