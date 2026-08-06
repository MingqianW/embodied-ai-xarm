"""Deterministic finite-state scripted oracles for xArm collection tasks."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from sim_mujoco.data_collection.conversions import (
    mujoco_gripper_target_from_raw,
    policy_action_from_mujoco_target,
    policy_state_from_mujoco,
)
from sim_mujoco.data_collection.ik_solver import IKSolution, solve_site_pose
from sim_mujoco.environment import MuJoCoEnvironment


class OracleStage(str, enum.Enum):
    RESET = "RESET"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    MOVE_TO_PREGRASP = "MOVE_TO_PREGRASP"
    DESCEND = "DESCEND"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    HOLD = "HOLD"
    LIFT = "LIFT"
    VERIFY = "VERIFY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class OracleConfig:
    task: str = "red_block"
    tcp_site: str = "tool_center_point"
    action_dt_s: float = 0.1
    open_gripper_raw: float = 845.0
    closed_gripper_raw: float = 211.0
    max_joint_step_rad: float = 0.025
    lift_max_joint_step_rad: float = 0.015
    max_gripper_step_raw: float = 25.0
    pregrasp_clearance_from_object_m: float = 0.087
    grasp_tcp_offset_from_object_m: float = -0.011
    lift_clearance_from_object_m: float = 0.107
    hold_steps: int = 5
    verify_steps: int = 10
    max_action_steps: int = 180

    def validate(self) -> None:
        if not self.task:
            raise ValueError("Oracle task must be non-empty")
        if self.action_dt_s <= 0.0:
            raise ValueError("action_dt_s must be positive")
        if self.max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive")
        if self.lift_max_joint_step_rad <= 0.0:
            raise ValueError("lift_max_joint_step_rad must be positive")
        if self.max_gripper_step_raw <= 0.0:
            raise ValueError("max_gripper_step_raw must be positive")
        if self.hold_steps < 1 or self.verify_steps < 1:
            raise ValueError("hold_steps and verify_steps must be positive")
        if self.max_action_steps < 1:
            raise ValueError("max_action_steps must be positive")


RED_PEPPER_CLOSED_GRIPPER_RAW = 250.0
RED_PEPPER_GRASP_TCP_OFFSET_M = -0.020


def oracle_config_for_task(
    task: str,
    *,
    action_dt_s: float,
    closed_gripper_raw: float | None = None,
    grasp_tcp_offset_from_object_m: float | None = None,
) -> OracleConfig:
    """Return an oracle configuration with explicit experimental overrides."""

    values: dict[str, Any] = {
        "task": task,
        "action_dt_s": action_dt_s,
    }
    if task == "red_pepper":
        values.update(
            {
                "closed_gripper_raw": RED_PEPPER_CLOSED_GRIPPER_RAW,
                "grasp_tcp_offset_from_object_m": (
                    RED_PEPPER_GRASP_TCP_OFFSET_M
                ),
            }
        )
    if closed_gripper_raw is not None:
        values["closed_gripper_raw"] = float(closed_gripper_raw)
    if grasp_tcp_offset_from_object_m is not None:
        values["grasp_tcp_offset_from_object_m"] = float(
            grasp_tcp_offset_from_object_m
        )
    return OracleConfig(**values)


@dataclass(frozen=True)
class StageTransition:
    action_step: int
    simulation_time_s: float
    from_stage: str | None
    to_stage: str
    reason: str


@dataclass(frozen=True)
class OraclePlan:
    object_position: np.ndarray
    tcp_rotation: np.ndarray
    initial_arm_qpos: np.ndarray
    pregrasp: IKSolution
    grasp: IKSolution
    lift: IKSolution

    def to_json(self) -> dict[str, Any]:
        return {
            "object_position": self.object_position.tolist(),
            "tcp_rotation": self.tcp_rotation.tolist(),
            "initial_arm_qpos": self.initial_arm_qpos.tolist(),
            "pregrasp": {
                **asdict(self.pregrasp),
                "joint_qpos": self.pregrasp.joint_qpos.tolist(),
            },
            "grasp": {
                **asdict(self.grasp),
                "joint_qpos": self.grasp.joint_qpos.tolist(),
            },
            "lift": {
                **asdict(self.lift),
                "joint_qpos": self.lift.joint_qpos.tolist(),
            },
        }


def _interpolate_arm(
    start: np.ndarray,
    target: np.ndarray,
    *,
    max_step_rad: float,
) -> list[np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    steps = max(
        1,
        int(np.ceil(float(np.max(np.abs(target - start))) / max_step_rad)),
    )
    return [
        start + (target - start) * alpha
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]
    ]


def _interpolate_gripper(
    start: float,
    target: float,
    *,
    max_step_raw: float,
) -> list[float]:
    steps = max(1, int(np.ceil(abs(target - start) / max_step_raw)))
    return [
        float(value)
        for value in np.linspace(start, target, steps + 1)[1:]
    ]


def _policy_action(arm_qpos: np.ndarray, gripper_raw: float) -> np.ndarray:
    slide_target = mujoco_gripper_target_from_raw(gripper_raw)
    internal_target = np.concatenate(
        (
            np.asarray(arm_qpos, dtype=np.float64),
            [slide_target],
        )
    )
    return policy_action_from_mujoco_target(internal_target)


class ScriptedOracleController:
    """Pre-planned pose IK with an explicit action-timestep state machine."""

    _SEQUENCE = (
        OracleStage.RESET,
        OracleStage.OPEN_GRIPPER,
        OracleStage.MOVE_TO_PREGRASP,
        OracleStage.DESCEND,
        OracleStage.CLOSE_GRIPPER,
        OracleStage.HOLD,
        OracleStage.LIFT,
        OracleStage.VERIFY,
    )

    def __init__(
        self,
        environment: MuJoCoEnvironment,
        config: OracleConfig | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or OracleConfig()
        self.config.validate()
        if environment.task != self.config.task:
            raise ValueError(
                f"Oracle task {self.config.task!r} does not match "
                f"environment task {environment.task!r}"
            )
        if environment.task_runtime is None:
            raise RuntimeError("Reset the environment before constructing the oracle")

        self.stage = OracleStage.RESET
        self.failure_reason: str | None = None
        self.action_steps = 0
        self.transitions: list[StageTransition] = [
            StageTransition(
                action_step=0,
                simulation_time_s=float(environment.context.data.time),
                from_stage=None,
                to_stage=OracleStage.RESET.value,
                reason="environment_reset",
            )
        ]
        self.plan = self._build_plan()
        self._stage_actions = self._build_stage_actions()
        self._stage_action_index = 0

    def _build_plan(self) -> OraclePlan:
        model = self.environment.context.model
        data = self.environment.context.data
        runtime = self.environment.task_runtime
        assert runtime is not None
        if runtime.spec["success"]["type"] != "lift":
            raise ValueError(
                f"Lift oracle requires a lift task, got "
                f"{runtime.spec['success']['type']!r}"
            )

        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.config.tcp_site,
        )
        if site_id < 0:
            raise RuntimeError(f"Oracle TCP site not found: {self.config.tcp_site}")
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            runtime.target_body,
        )
        if body_id < 0:
            raise RuntimeError(f"Oracle target body not found: {runtime.target_body}")

        object_position = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
        tcp_rotation = np.asarray(
            data.site_xmat[site_id],
            dtype=np.float64,
        ).reshape(3, 3).copy()
        initial_arm = policy_state_from_mujoco(model, data)[:6].astype(
            np.float64
        )
        target_xy = object_position[:2]
        pregrasp_position = np.asarray(
            [
                target_xy[0],
                target_xy[1],
                object_position[2]
                + self.config.pregrasp_clearance_from_object_m,
            ],
            dtype=np.float64,
        )
        grasp_position = np.asarray(
            [
                target_xy[0],
                target_xy[1],
                object_position[2]
                + self.config.grasp_tcp_offset_from_object_m,
            ],
            dtype=np.float64,
        )
        lift_position = np.asarray(
            [
                target_xy[0],
                target_xy[1],
                object_position[2]
                + self.config.lift_clearance_from_object_m,
            ],
            dtype=np.float64,
        )

        pregrasp = solve_site_pose(
            model,
            data,
            site_name=self.config.tcp_site,
            target_position=pregrasp_position,
            target_rotation=tcp_rotation,
            seed_joint_qpos=initial_arm,
        )
        grasp = solve_site_pose(
            model,
            data,
            site_name=self.config.tcp_site,
            target_position=grasp_position,
            target_rotation=tcp_rotation,
            seed_joint_qpos=pregrasp.joint_qpos,
        )
        lift = solve_site_pose(
            model,
            data,
            site_name=self.config.tcp_site,
            target_position=lift_position,
            target_rotation=tcp_rotation,
            seed_joint_qpos=grasp.joint_qpos,
        )
        plan = OraclePlan(
            object_position=object_position,
            tcp_rotation=tcp_rotation,
            initial_arm_qpos=initial_arm,
            pregrasp=pregrasp,
            grasp=grasp,
            lift=lift,
        )
        failed = [
            name
            for name, solution in (
                ("pregrasp", pregrasp),
                ("grasp", grasp),
                ("lift", lift),
            )
            if not solution.success
        ]
        if failed:
            self.failure_reason = "ik_failure:" + ",".join(failed)
            self.stage = OracleStage.FAILED
            self.transitions.append(
                StageTransition(
                    action_step=0,
                    simulation_time_s=float(data.time),
                    from_stage=OracleStage.RESET.value,
                    to_stage=OracleStage.FAILED.value,
                    reason=self.failure_reason,
                )
            )
        return plan

    def _build_stage_actions(self) -> dict[OracleStage, list[np.ndarray]]:
        if self.failure_reason is not None:
            return {}
        cfg = self.config
        initial_gripper = float(
            policy_state_from_mujoco(
                self.environment.context.model,
                self.environment.context.data,
            )[6]
        )
        open_values = _interpolate_gripper(
            initial_gripper,
            cfg.open_gripper_raw,
            max_step_raw=cfg.max_gripper_step_raw,
        )
        pregrasp_arms = _interpolate_arm(
            self.plan.initial_arm_qpos,
            self.plan.pregrasp.joint_qpos,
            max_step_rad=cfg.max_joint_step_rad,
        )
        descend_arms = _interpolate_arm(
            self.plan.pregrasp.joint_qpos,
            self.plan.grasp.joint_qpos,
            max_step_rad=cfg.max_joint_step_rad,
        )
        close_values = _interpolate_gripper(
            cfg.open_gripper_raw,
            cfg.closed_gripper_raw,
            max_step_raw=cfg.max_gripper_step_raw,
        )
        lift_arms = _interpolate_arm(
            self.plan.grasp.joint_qpos,
            self.plan.lift.joint_qpos,
            max_step_rad=cfg.lift_max_joint_step_rad,
        )
        return {
            OracleStage.RESET: [],
            OracleStage.OPEN_GRIPPER: [
                _policy_action(self.plan.initial_arm_qpos, value)
                for value in open_values
            ],
            OracleStage.MOVE_TO_PREGRASP: [
                _policy_action(arm, cfg.open_gripper_raw)
                for arm in pregrasp_arms
            ],
            OracleStage.DESCEND: [
                _policy_action(arm, cfg.open_gripper_raw)
                for arm in descend_arms
            ],
            OracleStage.CLOSE_GRIPPER: [
                _policy_action(self.plan.grasp.joint_qpos, value)
                for value in close_values
            ],
            OracleStage.HOLD: [
                _policy_action(
                    self.plan.grasp.joint_qpos,
                    cfg.closed_gripper_raw,
                )
                for _ in range(cfg.hold_steps)
            ],
            OracleStage.LIFT: [
                _policy_action(arm, cfg.closed_gripper_raw)
                for arm in lift_arms
            ],
            OracleStage.VERIFY: [
                _policy_action(
                    self.plan.lift.joint_qpos,
                    cfg.closed_gripper_raw,
                )
                for _ in range(cfg.verify_steps)
            ],
        }

    @property
    def terminal(self) -> bool:
        return self.stage in {OracleStage.COMPLETE, OracleStage.FAILED}

    def _transition(self, to_stage: OracleStage, reason: str) -> None:
        previous = self.stage
        self.stage = to_stage
        self._stage_action_index = 0
        self.transitions.append(
            StageTransition(
                action_step=self.action_steps,
                simulation_time_s=float(
                    self.environment.context.data.time
                ),
                from_stage=previous.value,
                to_stage=to_stage.value,
                reason=reason,
            )
        )

    def _fail(self, reason: str) -> None:
        if self.terminal:
            return
        self.failure_reason = reason
        self._transition(OracleStage.FAILED, reason)

    def next_action(self) -> np.ndarray | None:
        if self.terminal:
            return None
        if self.action_steps >= self.config.max_action_steps:
            self._fail("oracle_action_timeout")
            return None

        while not self.terminal:
            actions = self._stage_actions[self.stage]
            if self._stage_action_index < len(actions):
                action = actions[self._stage_action_index].copy()
                self._stage_action_index += 1
                return action

            if self.stage == OracleStage.VERIFY:
                self._fail("verification_timeout")
                return None
            sequence_index = self._SEQUENCE.index(self.stage)
            next_stage = self._SEQUENCE[sequence_index + 1]
            self._transition(next_stage, f"{self.stage.value.lower()}_complete")
        return None

    def notify_post_step(
        self,
        *,
        task_metrics: dict[str, Any],
        collision: dict[str, Any],
        simulation_finite: bool,
    ) -> None:
        if self.terminal:
            return
        self.action_steps += 1
        if not simulation_finite:
            self._fail("simulation_non_finite")
            return
        if collision.get("forbidden"):
            reason = str(
                collision.get("termination_reason")
                or "unexpected_collision"
            )
            self._fail(reason)
            return
        if self.stage == OracleStage.VERIFY and task_metrics.get(
            "task_success"
        ):
            self._transition(OracleStage.COMPLETE, "sustained_task_success")

    def transition_log(self) -> list[dict[str, Any]]:
        return [asdict(transition) for transition in self.transitions]


class PlaceOracleStage(str, enum.Enum):
    RESET = "RESET"
    MOVE_TO_PREPLACE = "MOVE_TO_PREPLACE"
    LOWER_TO_TARGET = "LOWER_TO_TARGET"
    RELEASE = "RELEASE"
    VERIFY = "VERIFY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class PlaceOracleConfig:
    task: str = "place_red_pepper_in_ring"
    tcp_site: str = "tool_center_point"
    action_dt_s: float = 0.1
    open_gripper_raw: float = 845.0
    max_joint_step_rad: float = 0.025
    max_gripper_step_raw: float = 25.0
    preplace_pepper_height_m: float = 0.20
    release_pepper_height_m: float = 0.125
    verify_steps: int = 20
    max_action_steps: int = 180

    def validate(self) -> None:
        if not self.task:
            raise ValueError("Oracle task must be non-empty")
        if self.action_dt_s <= 0.0:
            raise ValueError("action_dt_s must be positive")
        if self.max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive")
        if self.max_gripper_step_raw <= 0.0:
            raise ValueError("max_gripper_step_raw must be positive")
        if self.verify_steps < 1 or self.max_action_steps < 1:
            raise ValueError("verify_steps and max_action_steps must be positive")


@dataclass(frozen=True)
class PlaceOraclePlan:
    ring_position: np.ndarray
    held_pepper_offset_from_tcp: np.ndarray
    tcp_rotation: np.ndarray
    initial_arm_qpos: np.ndarray
    preplace: IKSolution
    release: IKSolution

    def to_json(self) -> dict[str, Any]:
        return {
            "ring_position": self.ring_position.tolist(),
            "held_pepper_offset_from_tcp": (
                self.held_pepper_offset_from_tcp.tolist()
            ),
            "tcp_rotation": self.tcp_rotation.tolist(),
            "initial_arm_qpos": self.initial_arm_qpos.tolist(),
            "preplace": {
                **asdict(self.preplace),
                "joint_qpos": self.preplace.joint_qpos.tolist(),
            },
            "release": {
                **asdict(self.release),
                "joint_qpos": self.release.joint_qpos.tolist(),
            },
        }


class PlaceRedPepperOracleController:
    """Move the already-held pepper above the ring and release it."""

    _SEQUENCE = (
        PlaceOracleStage.RESET,
        PlaceOracleStage.MOVE_TO_PREPLACE,
        PlaceOracleStage.LOWER_TO_TARGET,
        PlaceOracleStage.RELEASE,
        PlaceOracleStage.VERIFY,
    )

    def __init__(
        self,
        environment: MuJoCoEnvironment,
        config: PlaceOracleConfig | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or PlaceOracleConfig()
        self.config.validate()
        if environment.task != self.config.task:
            raise ValueError(
                f"Oracle task {self.config.task!r} does not match "
                f"environment task {environment.task!r}"
            )
        runtime = environment.task_runtime
        if runtime is None:
            raise RuntimeError("Reset the environment before constructing the oracle")
        if runtime.spec["success"]["type"] != "place_in_ring":
            raise ValueError("Place oracle requires a place_in_ring task")

        self.stage = PlaceOracleStage.RESET
        self.failure_reason: str | None = None
        self.action_steps = 0
        self.transitions: list[StageTransition] = [
            StageTransition(
                action_step=0,
                simulation_time_s=float(environment.context.data.time),
                from_stage=None,
                to_stage=PlaceOracleStage.RESET.value,
                reason="environment_reset",
            )
        ]
        self.plan = self._build_plan()
        self._stage_actions = self._build_stage_actions()
        self._stage_action_index = 0

    def _named_position(self, name: str, object_type: mujoco.mjtObj) -> np.ndarray:
        model = self.environment.context.model
        data = self.environment.context.data
        object_id = mujoco.mj_name2id(model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"MuJoCo object not found: {name}")
        if object_type == mujoco.mjtObj.mjOBJ_SITE:
            return np.asarray(data.site_xpos[object_id], dtype=np.float64).copy()
        return np.asarray(data.xpos[object_id], dtype=np.float64).copy()

    def _build_plan(self) -> PlaceOraclePlan:
        model = self.environment.context.model
        data = self.environment.context.data
        runtime = self.environment.task_runtime
        assert runtime is not None
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.config.tcp_site,
        )
        if site_id < 0:
            raise RuntimeError(f"Oracle TCP site not found: {self.config.tcp_site}")
        tcp_position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
        tcp_rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(
            3, 3
        ).copy()
        ring_position = self._named_position(
            str(runtime.spec["success"]["ring_body"]),
            mujoco.mjtObj.mjOBJ_BODY,
        )
        held_position = self._named_position(
            "held_red_pepper",
            mujoco.mjtObj.mjOBJ_BODY,
        )
        held_offset = held_position - tcp_position
        initial_arm = policy_state_from_mujoco(model, data)[:6].astype(np.float64)

        preplace_pepper = np.asarray(
            [
                ring_position[0],
                ring_position[1],
                self.config.preplace_pepper_height_m,
            ],
            dtype=np.float64,
        )
        release_pepper = np.asarray(
            [
                ring_position[0],
                ring_position[1],
                self.config.release_pepper_height_m,
            ],
            dtype=np.float64,
        )
        preplace = solve_site_pose(
            model,
            data,
            site_name=self.config.tcp_site,
            target_position=preplace_pepper - held_offset,
            target_rotation=tcp_rotation,
            seed_joint_qpos=initial_arm,
        )
        release = solve_site_pose(
            model,
            data,
            site_name=self.config.tcp_site,
            target_position=release_pepper - held_offset,
            target_rotation=tcp_rotation,
            seed_joint_qpos=preplace.joint_qpos,
        )
        plan = PlaceOraclePlan(
            ring_position=ring_position,
            held_pepper_offset_from_tcp=held_offset,
            tcp_rotation=tcp_rotation,
            initial_arm_qpos=initial_arm,
            preplace=preplace,
            release=release,
        )
        failed = [
            name
            for name, solution in (("preplace", preplace), ("release", release))
            if not solution.success
        ]
        if failed:
            self.failure_reason = "ik_failure:" + ",".join(failed)
            self.stage = PlaceOracleStage.FAILED
            self.transitions.append(
                StageTransition(
                    action_step=0,
                    simulation_time_s=float(data.time),
                    from_stage=PlaceOracleStage.RESET.value,
                    to_stage=PlaceOracleStage.FAILED.value,
                    reason=self.failure_reason,
                )
            )
        return plan

    def _build_stage_actions(
        self,
    ) -> dict[PlaceOracleStage, list[np.ndarray]]:
        if self.failure_reason is not None:
            return {}
        current_gripper = float(
            policy_state_from_mujoco(
                self.environment.context.model,
                self.environment.context.data,
            )[6]
        )
        preplace_arms = _interpolate_arm(
            self.plan.initial_arm_qpos,
            self.plan.preplace.joint_qpos,
            max_step_rad=self.config.max_joint_step_rad,
        )
        release_arms = _interpolate_arm(
            self.plan.preplace.joint_qpos,
            self.plan.release.joint_qpos,
            max_step_rad=self.config.max_joint_step_rad,
        )
        open_values = _interpolate_gripper(
            current_gripper,
            self.config.open_gripper_raw,
            max_step_raw=self.config.max_gripper_step_raw,
        )
        return {
            PlaceOracleStage.RESET: [],
            PlaceOracleStage.MOVE_TO_PREPLACE: [
                _policy_action(arm, current_gripper) for arm in preplace_arms
            ],
            PlaceOracleStage.LOWER_TO_TARGET: [
                _policy_action(arm, current_gripper) for arm in release_arms
            ],
            PlaceOracleStage.RELEASE: [
                _policy_action(self.plan.release.joint_qpos, value)
                for value in open_values
            ],
            PlaceOracleStage.VERIFY: [
                _policy_action(
                    self.plan.release.joint_qpos,
                    self.config.open_gripper_raw,
                )
                for _ in range(self.config.verify_steps)
            ],
        }

    @property
    def terminal(self) -> bool:
        return self.stage in {
            PlaceOracleStage.COMPLETE,
            PlaceOracleStage.FAILED,
        }

    def _transition(self, to_stage: PlaceOracleStage, reason: str) -> None:
        previous = self.stage
        self.stage = to_stage
        self._stage_action_index = 0
        self.transitions.append(
            StageTransition(
                action_step=self.action_steps,
                simulation_time_s=float(self.environment.context.data.time),
                from_stage=previous.value,
                to_stage=to_stage.value,
                reason=reason,
            )
        )

    def _fail(self, reason: str) -> None:
        if self.terminal:
            return
        self.failure_reason = reason
        self._transition(PlaceOracleStage.FAILED, reason)

    def next_action(self) -> np.ndarray | None:
        if self.terminal:
            return None
        if self.action_steps >= self.config.max_action_steps:
            self._fail("oracle_action_timeout")
            return None
        while not self.terminal:
            actions = self._stage_actions[self.stage]
            if self._stage_action_index < len(actions):
                action = actions[self._stage_action_index].copy()
                self._stage_action_index += 1
                return action
            if self.stage == PlaceOracleStage.VERIFY:
                self._fail("verification_timeout")
                return None
            sequence_index = self._SEQUENCE.index(self.stage)
            self._transition(
                self._SEQUENCE[sequence_index + 1],
                f"{self.stage.value.lower()}_complete",
            )
        return None

    def notify_post_step(
        self,
        *,
        task_metrics: dict[str, Any],
        collision: dict[str, Any],
        simulation_finite: bool,
    ) -> None:
        if self.terminal:
            return
        self.action_steps += 1
        if not simulation_finite:
            self._fail("simulation_non_finite")
            return
        if collision.get("forbidden"):
            self._fail(
                str(
                    collision.get("termination_reason")
                    or "unexpected_collision"
                )
            )
            return
        if (
            self.stage == PlaceOracleStage.VERIFY
            and task_metrics.get("task_success")
        ):
            self._transition(
                PlaceOracleStage.COMPLETE,
                "sustained_task_success",
            )

    def transition_log(self) -> list[dict[str, Any]]:
        return [asdict(transition) for transition in self.transitions]
