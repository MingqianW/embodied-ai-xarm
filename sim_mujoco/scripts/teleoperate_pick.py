"""Cartesian keyboard teleoperation for the xArm6 MuJoCo pick scene."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from mujoco.glfw import glfw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.data_collection.ik_solver import solve_site_pose
from simulation.environment import MuJoCoEnvironment
from simulation.robot.model import ARM_JOINT_NAMES
from simulation.robot.model import LEFT_GRIPPER_DRIVER_JOINT_NAME


TCP_SITE = "tool_center_point"
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
    target: np.ndarray
    gripper_limits: np.ndarray
    gripper_step_m: float
    cartesian_step_m: float
    reset_requested: bool = False
    stop_requested: bool = False

    def apply_non_motion_key(self, key: int) -> bool:
        if key == glfw.KEY_O:
            # Menagerie xArm four-bar: smaller driver angle means open.
            self.target[6] -= self.gripper_step_m
        elif key == glfw.KEY_C:
            self.target[6] += self.gripper_step_m
        elif key == glfw.KEY_X:
            self.reset_requested = True
            return True
        elif key == glfw.KEY_ESCAPE:
            self.stop_requested = True
            return True
        else:
            return False
        self.target[6] = np.clip(self.target[6], self.gripper_limits[0], self.gripper_limits[1])
        return True


def _print_controls() -> None:
    print("Keyboard controls (click the viewer first):")
    print("  Up/Down: TCP +X/-X (away from/toward robot base)")
    print("  Left/Right: TCP +Y/-Y")
    print("  Page Up/Page Down: TCP +Z/-Z")
    print("  O/C: open/close gripper; H: hold; X: reset; Esc: quit")


def _current_targets(environment: MuJoCoEnvironment) -> np.ndarray:
    return np.asarray(environment.context.data.ctrl[:7], dtype=np.float64).copy()


def _current_position_targets(environment: MuJoCoEnvironment) -> np.ndarray:
    model, data = environment.context.model, environment.context.data
    arm = []
    for joint_name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        arm.append(data.qpos[model.jnt_qposadr[joint_id]])
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, LEFT_GRIPPER_DRIVER_JOINT_NAME)
    return np.asarray([*arm, data.qpos[model.jnt_qposadr[gripper_id]]], dtype=np.float64)


def _move_tcp(environment: MuJoCoEnvironment, state: TeleoperationState, direction: np.ndarray) -> None:
    model, data = environment.context.model, environment.context.data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    if site_id < 0:
        raise RuntimeError(f"TCP site not found: {TCP_SITE}")
    target_position = np.asarray(data.site_xpos[site_id], dtype=np.float64) + direction * state.cartesian_step_m
    target_rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    solution = solve_site_pose(
        model, data, site_name=TCP_SITE, target_position=target_position,
        target_rotation=target_rotation, seed_joint_qpos=state.target[:6],
        max_iterations=100, position_tolerance_m=5e-4,
    )
    if not solution.success:
        print(f"IK rejected movement: {solution.reason}; error={solution.position_error_m * 1000:.1f} mm")
        return
    state.target[:6] = solution.joint_position
    print("TCP target (m):", np.array2string(target_position, precision=3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="red_block", help="Task key, e.g. red_block or blue_block.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic scene seed.")
    parser.add_argument("--cartesian-step-mm", type=float, default=10.0, help="TCP distance per direction key press.")
    parser.add_argument("--gripper-step-mm", type=float, default=10.0, help="Raw xArm gripper increment per key press.")
    parser.add_argument("--control-hz", type=float, default=50.0, help="Physics/control update rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cartesian_step_mm <= 0 or args.gripper_step_mm <= 0 or args.control_hz <= 0:
        raise ValueError("motion steps and --control-hz must be positive")

    with MuJoCoEnvironment(task=args.task, settle_steps=500) as environment:
        environment.reset(seed=args.seed)
        model, data = environment.context.model, environment.context.data
        state = TeleoperationState(
            target=_current_targets(environment),
            gripper_limits=np.asarray(model.actuator_ctrlrange[6], dtype=np.float64),
            # The xArm driver uses 1000 raw-position units per radian.
            gripper_step_m=float(args.gripper_step_mm) / 1000.0,
            cartesian_step_m=float(args.cartesian_step_mm) / 1000.0,
        )
        _print_controls()
        print(f"Task: {environment.task}. Target: {environment.task_runtime.target_body}")

        def viewer_key_callback(key: int) -> None:
            direction = TCP_KEY_DELTAS.get(key)
            if direction is not None:
                _move_tcp(environment, state, direction)
            elif key == glfw.KEY_H:
                state.target[:] = _current_position_targets(environment)
                print("Holding current physical position.")
            elif state.apply_non_motion_key(key):
                print("target:", np.array2string(state.target, precision=3))

        period_s = 1.0 / float(args.control_hz)
        with mujoco.viewer.launch_passive(model, data, key_callback=viewer_key_callback) as viewer:
            overview_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_camera")
            if overview_camera_id < 0:
                raise RuntimeError("overview_camera was not found in the MuJoCo scene")
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = overview_camera_id
            while viewer.is_running() and not state.stop_requested:
                started = time.perf_counter()
                if state.reset_requested:
                    environment.reset(seed=args.seed)
                    state.target[:] = _current_targets(environment)
                    state.reset_requested = False
                    print("Task reset.")
                data.ctrl[:7] = state.target
                environment.step_physics(period_s)
                metrics = environment.task_runtime.update_success()
                if metrics["task_success"]:
                    print("Task success. Continue moving or press X to try again.")
                    environment.task_runtime.success_streak = 0
                viewer.sync()
                remaining = period_s - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)


if __name__ == "__main__":
    main()
