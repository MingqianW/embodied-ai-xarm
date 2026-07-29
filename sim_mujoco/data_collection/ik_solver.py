"""Named-joint damped-least-squares IK for the MuJoCo xArm TCP."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim_mujoco.remote_policy_observation import ARM_JOINT_NAMES


@dataclass(frozen=True)
class IKSolution:
    success: bool
    joint_qpos: np.ndarray
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    reason: str


def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-frame small-angle error from current rotation to target."""

    return 0.5 * sum(
        (np.cross(current[:, axis], target[:, axis]) for axis in range(3)),
        np.zeros(3, dtype=np.float64),
    )


def _arm_addresses(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    limits: list[np.ndarray] = []
    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if joint_id < 0:
            raise RuntimeError(f"Arm joint not found: {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        limits.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
    return (
        np.asarray(qpos_addresses, dtype=np.int64),
        np.asarray(dof_addresses, dtype=np.int64),
        np.asarray(limits, dtype=np.float64),
    )


def solve_site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_name: str,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    seed_joint_qpos: np.ndarray | None = None,
    max_iterations: int = 400,
    damping: float = 1e-3,
    max_iteration_joint_step_rad: float = 0.05,
    position_tolerance_m: float = 1e-5,
    orientation_tolerance_rad: float = 1e-4,
) -> IKSolution:
    """Solve a six-dimensional site pose without mutating the caller's data."""

    target_position = np.asarray(target_position, dtype=np.float64)
    target_rotation = np.asarray(target_rotation, dtype=np.float64)
    if target_position.shape != (3,):
        raise ValueError(
            f"target_position must have shape (3,), got {target_position.shape}"
        )
    if target_rotation.shape != (3, 3):
        raise ValueError(
            f"target_rotation must have shape (3, 3), got {target_rotation.shape}"
        )
    if not np.isfinite(target_position).all() or not np.isfinite(
        target_rotation
    ).all():
        raise ValueError("IK target contains NaN or Inf")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if damping <= 0.0 or max_iteration_joint_step_rad <= 0.0:
        raise ValueError("IK damping and step limit must be positive")

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )
    if site_id < 0:
        raise RuntimeError(f"IK site not found: {site_name}")

    qpos_addresses, dof_addresses, limits = _arm_addresses(model)
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    work.qvel[:] = 0.0
    work.ctrl[:] = data.ctrl

    if seed_joint_qpos is not None:
        seed = np.asarray(seed_joint_qpos, dtype=np.float64)
        if seed.shape != (6,) or not np.isfinite(seed).all():
            raise ValueError("seed_joint_qpos must be finite with shape (6,)")
        work.qpos[qpos_addresses] = np.clip(
            seed,
            limits[:, 0],
            limits[:, 1],
        )
    mujoco.mj_forward(model, work)

    position_error_norm = float("inf")
    orientation_error_norm = float("inf")
    for iteration in range(max_iterations):
        current_position = np.asarray(
            work.site_xpos[site_id],
            dtype=np.float64,
        )
        current_rotation = np.asarray(
            work.site_xmat[site_id],
            dtype=np.float64,
        ).reshape(3, 3)
        position_error = target_position - current_position
        orientation_error = _orientation_error(
            current_rotation,
            target_rotation,
        )
        position_error_norm = float(np.linalg.norm(position_error))
        orientation_error_norm = float(np.linalg.norm(orientation_error))
        if (
            position_error_norm <= position_tolerance_m
            and orientation_error_norm <= orientation_tolerance_rad
        ):
            return IKSolution(
                success=True,
                joint_qpos=np.asarray(
                    work.qpos[qpos_addresses],
                    dtype=np.float64,
                ).copy(),
                iterations=iteration,
                position_error_m=position_error_norm,
                orientation_error_rad=orientation_error_norm,
                reason="converged",
            )

        jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            model,
            work,
            jacobian_position,
            jacobian_rotation,
            site_id,
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, dof_addresses],
                jacobian_rotation[:, dof_addresses],
            )
        )
        error = np.concatenate((position_error, orientation_error))
        normal = jacobian @ jacobian.T
        try:
            delta = jacobian.T @ np.linalg.solve(
                normal + damping * np.eye(6, dtype=np.float64),
                error,
            )
        except np.linalg.LinAlgError:
            return IKSolution(
                success=False,
                joint_qpos=np.asarray(
                    work.qpos[qpos_addresses],
                    dtype=np.float64,
                ).copy(),
                iterations=iteration,
                position_error_m=position_error_norm,
                orientation_error_rad=orientation_error_norm,
                reason="singular_linear_system",
            )
        if not np.isfinite(delta).all():
            return IKSolution(
                success=False,
                joint_qpos=np.asarray(
                    work.qpos[qpos_addresses],
                    dtype=np.float64,
                ).copy(),
                iterations=iteration,
                position_error_m=position_error_norm,
                orientation_error_rad=orientation_error_norm,
                reason="non_finite_update",
            )

        largest_delta = float(np.max(np.abs(delta)))
        if largest_delta > max_iteration_joint_step_rad:
            delta *= max_iteration_joint_step_rad / largest_delta
        work.qpos[qpos_addresses] = np.clip(
            work.qpos[qpos_addresses] + delta,
            limits[:, 0] + 1e-6,
            limits[:, 1] - 1e-6,
        )
        mujoco.mj_forward(model, work)

    return IKSolution(
        success=False,
        joint_qpos=np.asarray(
            work.qpos[qpos_addresses],
            dtype=np.float64,
        ).copy(),
        iterations=max_iterations,
        position_error_m=position_error_norm,
        orientation_error_rad=orientation_error_norm,
        reason="max_iterations",
    )
