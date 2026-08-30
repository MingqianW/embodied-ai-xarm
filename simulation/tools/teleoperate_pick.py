"""Interactive local keyboard control for all canonical xArm6 MuJoCo tasks.

This tool controls only the simulated robot. It never imports or connects to
the real xArm runtime.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import mujoco
import mujoco.viewer
import numpy as np
from mujoco.glfw import glfw

from data.common.task_identity import TASKS, resolve_task_id
from simulation.environment import MuJoCoEnvironment
from simulation.observation.state import get_robot_state
from simulation.robot.gripper_mapping import GripperMapping
from simulation.robot.ik import solve_site_pose


TCP_SITE = "tool_center_point"
DEFAULT_TASK = "red_block"
TASK_IDS = tuple(task.task_id for task in TASKS)
TASK_KEY_CODES = (
    glfw.KEY_1,
    glfw.KEY_2,
    glfw.KEY_3,
    glfw.KEY_4,
    glfw.KEY_5,
    glfw.KEY_6,
)
TASK_BY_KEY = dict(zip(TASK_KEY_CODES, TASK_IDS, strict=True))
TCP_KEY_DELTAS = {
    glfw.KEY_UP: np.array([1.0, 0.0, 0.0]),
    glfw.KEY_DOWN: np.array([-1.0, 0.0, 0.0]),
    glfw.KEY_LEFT: np.array([0.0, 1.0, 0.0]),
    glfw.KEY_RIGHT: np.array([0.0, -1.0, 0.0]),
    glfw.KEY_PAGE_UP: np.array([0.0, 0.0, 1.0]),
    glfw.KEY_PAGE_DOWN: np.array([0.0, 0.0, -1.0]),
}


@dataclass
class TeleoperationState:
    """Mutable user-command state shared by the viewer callback and main loop."""

    action: np.ndarray
    gripper_raw_limits: np.ndarray
    gripper_step_raw: float
    cartesian_step_m: float
    seed: int | None
    motion_requests: list[np.ndarray] = field(default_factory=list)
    hold_requested: bool = False
    reset_requested: bool = False
    status_requested: bool = False
    next_task: str | None = None
    quit_requested: bool = False

    def handle_key(self, key: int) -> bool:
        direction = TCP_KEY_DELTAS.get(key)
        if direction is not None:
            self.motion_requests.append(direction)
        elif key == glfw.KEY_O:
            self.action[6] += self.gripper_step_raw
        elif key == glfw.KEY_C:
            self.action[6] -= self.gripper_step_raw
        elif key == glfw.KEY_H:
            self.hold_requested = True
        elif key == glfw.KEY_X:
            self.reset_requested = True
        elif key == glfw.KEY_N:
            self.seed = 0 if self.seed is None else self.seed + 1
            self.reset_requested = True
        elif key == glfw.KEY_M:
            self.status_requested = True
        elif key in TASK_BY_KEY:
            self.next_task = TASK_BY_KEY[key]
        elif key == glfw.KEY_ESCAPE:
            self.quit_requested = True
        else:
            return False
        self.action[6] = np.clip(
            self.action[6],
            self.gripper_raw_limits[0],
            self.gripper_raw_limits[1],
        )
        return True


def _task_lines() -> list[str]:
    return [
        f"  {index}. {task.task_id:<31} {task.prompt}"
        for index, task in enumerate(TASKS, start=1)
    ]


def print_tasks() -> None:
    print("Available MuJoCo tasks:")
    print("\n".join(_task_lines()))


def select_task(
    requested: str | None,
    *,
    input_fn: Callable[[str], str] = input,
) -> str:
    """Resolve a CLI task or prompt for one interactively when omitted."""

    if requested:
        return resolve_task_id(requested)
    print_tasks()
    while True:
        try:
            value = input_fn(
                f"Select task [1-{len(TASKS)}] ({DEFAULT_TASK}): "
            ).strip()
        except EOFError as exc:
            raise RuntimeError(
                "Interactive task selection requires a terminal"
            ) from exc
        if not value:
            return DEFAULT_TASK
        if value.isdigit() and 1 <= int(value) <= len(TASKS):
            return TASK_IDS[int(value) - 1]
        try:
            return resolve_task_id(value)
        except ValueError:
            print(f"Unknown task selection: {value!r}")


def _print_controls() -> None:
    print("Keyboard controls (click the MuJoCo viewer first):")
    print("  Arrow Up/Down: TCP +X/-X")
    print("  Arrow Left/Right: TCP +Y/-Y")
    print("  Page Up/Page Down: TCP +Z/-Z")
    print("  O/C: open/close gripper; H: hold current physical position")
    print("  X: reset same seed; N: reset with next seed; M: print status")
    print("  1-6: switch task; Esc: quit")


def _current_action(environment: MuJoCoEnvironment) -> np.ndarray:
    return np.asarray(
        get_robot_state(
            environment.context.model,
            environment.context.data,
            environment.context.config,
        ),
        dtype=np.float64,
    )


def _gripper_raw_limits(environment: MuJoCoEnvironment) -> np.ndarray:
    mapping = GripperMapping.from_config(environment.context.config)
    return np.asarray(
        [mapping.raw_closed, mapping.raw_open],
        dtype=np.float64,
    )


def _move_tcp(
    environment: MuJoCoEnvironment,
    state: TeleoperationState,
    direction: np.ndarray,
) -> None:
    model, data = environment.context.model, environment.context.data
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        TCP_SITE,
    )
    if site_id < 0:
        raise RuntimeError(f"TCP site not found: {TCP_SITE}")
    target_position = (
        np.asarray(data.site_xpos[site_id], dtype=np.float64)
        + direction * state.cartesian_step_m
    )
    target_rotation = np.asarray(
        data.site_xmat[site_id],
        dtype=np.float64,
    ).reshape(3, 3)
    solution = solve_site_pose(
        model,
        data,
        site_name=TCP_SITE,
        target_position=target_position,
        target_rotation=target_rotation,
        seed_joint_qpos=state.action[:6],
        max_iterations=100,
        position_tolerance_m=5e-4,
    )
    if not solution.success:
        print(
            "IK rejected movement: "
            f"{solution.reason}; error={solution.position_error_m * 1000:.1f} mm"
        )
        return
    state.action[:6] = solution.joint_position
    print("TCP target (m):", np.array2string(target_position, precision=3))


def _status(
    environment: MuJoCoEnvironment,
    seed: int | None,
) -> dict[str, object]:
    runtime = environment.task_runtime
    if runtime is None:
        raise RuntimeError("Task runtime is unavailable before reset")
    diagnostics = environment.safety_diagnostics()
    collision = diagnostics["collision"]
    return {
        "task": environment.task,
        "prompt": environment.prompt,
        "seed": seed,
        "scene_variant": environment.scene_variant,
        "target_body": runtime.target_body,
        "simulation_time_s": diagnostics["simulation_time_s"],
        "task_metrics": runtime.metrics(),
        "collision": {
            "contact_kind_counts": collision["contact_kind_counts"],
            "forbidden": collision["forbidden"],
            "termination_reason": collision["termination_reason"],
        },
    }


def _reset(
    environment: MuJoCoEnvironment,
    state: TeleoperationState,
) -> None:
    environment.reset(seed=state.seed)
    state.action[:] = _current_action(environment)
    state.motion_requests.clear()
    state.reset_requested = False
    print(f"Task reset: task={environment.task} seed={state.seed}")


def _run_task(task: str, args: argparse.Namespace) -> str | None:
    with MuJoCoEnvironment(
        task=task,
        settle_steps=args.settle_steps,
        object_xy_range=args.object_xy_range_m,
        object_yaw_range_deg=args.object_yaw_range_deg,
        joint_noise=args.joint_noise_rad,
        scene_variant=args.scene_variant,
    ) as environment:
        environment.reset(seed=args.seed)
        model, data = environment.context.model, environment.context.data
        state = TeleoperationState(
            action=_current_action(environment),
            gripper_raw_limits=_gripper_raw_limits(environment),
            gripper_step_raw=float(args.gripper_step_raw),
            cartesian_step_m=float(args.cartesian_step_mm) / 1000.0,
            seed=args.seed,
        )
        runtime = environment.task_runtime
        if runtime is None:
            raise RuntimeError("Task runtime was not initialized")
        print(
            f"Task: {environment.task}; prompt={environment.prompt!r}; "
            f"target={runtime.target_body}; seed={state.seed}; "
            f"scene={args.scene_variant}"
        )
        _print_controls()
        print_tasks()

        def viewer_key_callback(key: int) -> None:
            state.handle_key(key)

        period_s = 1.0 / float(args.control_hz)
        success_reported = False
        collision_reported = False
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=viewer_key_callback,
        ) as viewer:
            overview_camera_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "overview_camera",
            )
            if overview_camera_id < 0:
                raise RuntimeError(
                    "overview_camera was not found in the MuJoCo scene"
                )
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = overview_camera_id
            while (
                viewer.is_running()
                and not state.quit_requested
                and state.next_task is None
            ):
                started = time.perf_counter()
                if state.reset_requested:
                    _reset(environment, state)
                    success_reported = False
                    collision_reported = False
                if state.hold_requested:
                    state.action[:] = _current_action(environment)
                    state.hold_requested = False
                    print("Holding current physical position.")
                while state.motion_requests:
                    _move_tcp(
                        environment,
                        state,
                        state.motion_requests.pop(0),
                    )
                if state.status_requested:
                    print(
                        json.dumps(
                            _status(environment, state.seed),
                            indent=2,
                        )
                    )
                    state.status_requested = False

                environment.apply_action(state.action)
                environment.step_physics(period_s)
                runtime = environment.task_runtime
                if runtime is None:
                    raise RuntimeError("Task runtime disappeared after reset")
                metrics = runtime.update_success()
                if metrics["task_success"] and not success_reported:
                    print(
                        "TASK SUCCESS. Press X to retry or 1-6 to change task."
                    )
                    success_reported = True
                elif not metrics["instant_success"]:
                    success_reported = False
                collision = environment.safety_diagnostics()["collision"]
                if collision["forbidden"] and not collision_reported:
                    print(
                        "WARNING: forbidden collision: "
                        f"{collision['termination_reason']}. Press X to reset."
                    )
                    collision_reported = True
                elif not collision["forbidden"]:
                    collision_reported = False
                viewer.sync()
                remaining = period_s - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        args.seed = state.seed
        return state.next_task


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        help="Task ID, prompt, or alias. Omit for an interactive menu.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the task menu and exit without opening MuJoCo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic initial scene seed (default: 0).",
    )
    parser.add_argument(
        "--scene-variant",
        choices=("clean", "distractors"),
        default="clean",
    )
    parser.add_argument("--object-xy-range-m", type=float, default=0.02)
    parser.add_argument("--object-yaw-range-deg", type=float, default=10.0)
    parser.add_argument("--joint-noise-rad", type=float, default=0.005)
    parser.add_argument(
        "--cartesian-step-mm",
        type=float,
        default=10.0,
        help="TCP distance per direction-key press.",
    )
    parser.add_argument(
        "--gripper-step-raw",
        type=float,
        default=25.0,
        help="Canonical xArm raw gripper units per O/C key press.",
    )
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--settle-steps", type=int, default=500)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--cartesian-step-mm": args.cartesian_step_mm,
        "--gripper-step-raw": args.gripper_step_raw,
        "--control-hz": args.control_hz,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    nonnegative = {
        "--object-xy-range-m": args.object_xy_range_m,
        "--object-yaw-range-deg": args.object_yaw_range_deg,
        "--joint-noise-rad": args.joint_noise_rad,
        "--settle-steps": args.settle_steps,
    }
    for name, value in nonnegative.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_tasks:
        print_tasks()
        return
    _validate_args(args)
    task = select_task(args.task)
    while task is not None:
        task = _run_task(task, args)


if __name__ == "__main__":
    main()
