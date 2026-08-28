"""Immutable protocol configuration for formal xArm policy evaluation."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from sim_mujoco.data_generation.registry import TASKS
from sim_mujoco.paths import active_model_path
from sim_mujoco.paths import camera_config_path
from sim_mujoco.paths import task_config_path

FORMAL_PROTOCOL_VERSION = "xarm-pi05-formal-evaluation-v1"
FORMAL_STABLE_HOLD_PROTOCOL_VERSION = "xarm-pi05-formal-evaluation-v2"
SMOKE_PROTOCOL_VERSION = "xarm-pi05-evaluation-smoke-v1"
SMOKE_STABLE_HOLD_PROTOCOL_VERSION = "xarm-pi05-evaluation-smoke-v2"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str


@dataclass(frozen=True)
class FormalProtocol:
    protocol_version: str
    tasks: tuple[TaskSpec, ...]
    seed_start: int
    seed_count: int
    execute_chunk_steps: int
    policy_action_horizon: int
    max_policy_steps: int
    control_duration_s: float
    expected_physics_timestep_s: float
    object_xy_range_m: float
    object_yaw_range_deg: float
    joint_noise_rad: float
    camera_config_path: Path
    task_scene_config_path: Path
    robot_xml_path: Path
    output_root: Path
    video_policy: str
    representatives_per_category: int
    periodic_video_every: int
    fail_on_invalid: bool
    rng_salt: str
    pick_lift_height_m: float
    pick_meaningful_lift_diagnostic_m: float
    pick_success_checks: int
    pick_post_success_hold_checks: int
    pick_max_post_success_drop_m: float
    placement_initial_validation_checks: int
    placement_initial_validation_dt_s: float
    placement_initial_max_relative_drift_m: float
    placement_initial_min_height_above_table_m: float
    placement_initial_min_gripper_contacts: int
    placement_ring_inner_radius_m: float
    placement_pepper_effective_radius_m: float
    placement_containment_tolerance_m: float
    placement_min_height_above_table_m: float
    placement_max_height_above_table_m: float
    placement_max_linear_speed_mps: float
    placement_max_angular_speed_radps: float
    placement_min_gripper_distance_m: float
    placement_release_gripper_raw: float
    placement_success_checks: int

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))

    @property
    def placement_max_center_distance_m(self) -> float:
        return (
            self.placement_ring_inner_radius_m
            - self.placement_pepper_effective_radius_m
            - self.placement_containment_tolerance_m
        )

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in value.items():
            if isinstance(item, Path):
                value[key] = str(item)
        return value


def default_protocol_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "formal_xarm_pi05_eval_v2.json"


def _absolute_path(value: str, *, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _canonical_input_path(
    value: str,
    *,
    config_path: Path,
    fallback: Path,
) -> Path:
    """Use a deployed absolute input when present, otherwise the repo input.

    Checked-in formal protocols contain cluster paths by design.  Falling back
    only when those read-only inputs are unavailable lets the same protocol run
    against this checkout without redirecting its configured output root.
    """

    configured = _absolute_path(value, config_path=config_path)
    return configured if configured.is_file() else fallback.resolve()


def _task_specs(rows: list[dict[str, Any]]) -> tuple[TaskSpec, ...]:
    expected = tuple(TaskSpec(task.task_id, task.prompt) for task in TASKS)
    actual = tuple(TaskSpec(str(row["task_id"]), str(row["prompt"])) for row in rows)
    if actual != expected:
        raise ValueError(
            "Formal protocol tasks/prompts must exactly match the canonical six-task registry"
        )
    return actual


def load_protocol(path: Path | None = None) -> FormalProtocol:
    path = (path or default_protocol_path()).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("protocol_version") not in {
        FORMAL_PROTOCOL_VERSION,
        FORMAL_STABLE_HOLD_PROTOCOL_VERSION,
        SMOKE_PROTOCOL_VERSION,
        SMOKE_STABLE_HOLD_PROTOCOL_VERSION,
    }:
        raise ValueError(f"Unsupported formal evaluation protocol: {raw.get('protocol_version')!r}")
    environment = raw["environment"]
    control = raw["control"]
    video = raw["video"]
    placement = raw["placement_success"]
    reset = raw["placement_reset_validation"]
    protocol = FormalProtocol(
        protocol_version=str(raw["protocol_version"]),
        tasks=_task_specs(list(raw["tasks"])),
        seed_start=int(raw["seeds"]["start"]),
        seed_count=int(raw["seeds"]["count"]),
        execute_chunk_steps=int(control["execute_chunk_steps"]),
        policy_action_horizon=int(control["policy_action_horizon"]),
        max_policy_steps=int(control["max_policy_steps"]),
        control_duration_s=float(control["control_duration_s"]),
        expected_physics_timestep_s=float(control["expected_physics_timestep_s"]),
        object_xy_range_m=float(environment["object_xy_range_m"]),
        object_yaw_range_deg=float(environment["object_yaw_range_deg"]),
        joint_noise_rad=float(environment["joint_noise_rad"]),
        camera_config_path=_canonical_input_path(
            raw["paths"]["camera_config"],
            config_path=path,
            fallback=camera_config_path(),
        ),
        task_scene_config_path=_canonical_input_path(
            raw["paths"]["task_scene_config"],
            config_path=path,
            fallback=task_config_path(),
        ),
        robot_xml_path=_canonical_input_path(
            raw["paths"]["robot_xml"],
            config_path=path,
            fallback=active_model_path(),
        ),
        output_root=_absolute_path(raw["outputs"]["formal_output_root"], config_path=path),
        video_policy=str(video.get("video_policy", "periodic")),
        representatives_per_category=int(video.get("representatives_per_category", 1)),
        periodic_video_every=int(video.get("periodic_every_n_episodes", video.get("every_n_episodes", 1))),
        fail_on_invalid=bool(raw["invalid_episode_behavior"]["fail_on_invalid"]),
        rng_salt=str(raw["policy_rng"]["salt"]),
        pick_lift_height_m=float(raw["pick_success"]["lift_height_m"]),
        pick_meaningful_lift_diagnostic_m=float(raw["pick_success"]["meaningful_lift_diagnostic_m"]),
        pick_success_checks=int(raw["pick_success"]["sustained_policy_checks"]),
        pick_post_success_hold_checks=int(raw["pick_success"].get("post_success_hold_policy_checks", 0)),
        pick_max_post_success_drop_m=float(raw["pick_success"].get("max_post_success_drop_m", 0.0)),
        placement_initial_validation_checks=int(reset["checks"]),
        placement_initial_validation_dt_s=float(reset["check_interval_s"]),
        placement_initial_max_relative_drift_m=float(reset["max_relative_drift_m"]),
        placement_initial_min_height_above_table_m=float(reset["min_height_above_table_m"]),
        placement_initial_min_gripper_contacts=int(reset["min_gripper_contacts"]),
        placement_ring_inner_radius_m=float(placement["ring_inner_radius_m"]),
        placement_pepper_effective_radius_m=float(placement["pepper_effective_radius_m"]),
        placement_containment_tolerance_m=float(placement["containment_tolerance_m"]),
        placement_min_height_above_table_m=float(placement["min_height_above_table_m"]),
        placement_max_height_above_table_m=float(placement["max_height_above_table_m"]),
        placement_max_linear_speed_mps=float(placement["max_linear_speed_mps"]),
        placement_max_angular_speed_radps=float(placement["max_angular_speed_radps"]),
        placement_min_gripper_distance_m=float(placement["min_gripper_distance_m"]),
        placement_release_gripper_raw=float(placement["release_gripper_raw"]),
        placement_success_checks=int(placement["sustained_policy_checks"]),
    )
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: FormalProtocol) -> None:
    if protocol.seed_start < 0:
        raise ValueError("Evaluation seeds must be non-negative")
    if protocol.protocol_version in {
        FORMAL_PROTOCOL_VERSION,
        FORMAL_STABLE_HOLD_PROTOCOL_VERSION,
    } and protocol.seed_count != 20:
        raise ValueError("Formal protocol requires exactly 20 fixed seeds")
    if protocol.protocol_version in {
        SMOKE_PROTOCOL_VERSION,
        SMOKE_STABLE_HOLD_PROTOCOL_VERSION,
    } and not 1 <= protocol.seed_count <= 3:
        raise ValueError("Smoke protocol requires one to three fixed seeds")
    if (protocol.execute_chunk_steps, protocol.policy_action_horizon, protocol.max_policy_steps) != (5, 10, 50):
        raise ValueError("Formal protocol requires c5, action horizon 10, and i50")
    if protocol.control_duration_s <= 0 or protocol.expected_physics_timestep_s <= 0:
        raise ValueError("Control durations must be positive")
    if protocol.video_policy not in {"category_representative", "all", "periodic"}:
        raise ValueError("Formal protocol has an unsupported video policy")
    if protocol.representatives_per_category != 1:
        raise ValueError("Formal protocol currently supports exactly one representative per category")
    if protocol.periodic_video_every < 1 or not protocol.fail_on_invalid:
        raise ValueError("Formal protocol requires positive periodic cadence and fail-on-invalid")
    if not 0 < protocol.pick_meaningful_lift_diagnostic_m < protocol.pick_lift_height_m:
        raise ValueError("Pick meaningful-lift diagnostic threshold must be between zero and success lift")
    stable_hold_protocol = protocol.protocol_version in {
        FORMAL_STABLE_HOLD_PROTOCOL_VERSION,
        SMOKE_STABLE_HOLD_PROTOCOL_VERSION,
    }
    if stable_hold_protocol and (
        protocol.pick_post_success_hold_checks < 1
        or not 0 < protocol.pick_max_post_success_drop_m < protocol.pick_lift_height_m
    ):
        raise ValueError("Stable-hold pick protocol requires positive hold checks and a bounded drop tolerance")
    if not stable_hold_protocol and (
        protocol.pick_post_success_hold_checks != 0 or protocol.pick_max_post_success_drop_m != 0.0
    ):
        raise ValueError("Only the stable-hold formal v2 protocol may enable post-success pick holding")
    if protocol.placement_max_center_distance_m <= 0:
        raise ValueError("Placement containment geometry leaves no usable ring area")
    if not all(
        path.is_file()
        for path in (
            protocol.camera_config_path,
            protocol.task_scene_config_path,
            protocol.robot_xml_path,
        )
    ):
        raise FileNotFoundError("Formal protocol camera, task, and XML files must exist")
    authorized_output_parent = Path("/work/nvme/bfmk/mw89").resolve(strict=False)
    if (
        protocol.output_root == authorized_output_parent
        or authorized_output_parent not in protocol.output_root.parents
    ):
        raise ValueError("Formal evaluation output root must be under /work/nvme/bfmk/mw89")
    # Catch accidental divergence from canonical path resolution before a job
    # spends GPU time loading a policy.
    expected_paths = (camera_config_path(), task_config_path(), active_model_path())
    actual_paths = (
        protocol.camera_config_path,
        protocol.task_scene_config_path,
        protocol.robot_xml_path,
    )
    if actual_paths != expected_paths:
        raise ValueError("Formal protocol must use the active calibrated camera, task, and XML paths")
