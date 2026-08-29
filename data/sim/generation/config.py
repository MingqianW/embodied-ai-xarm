"""Typed loading and strict validation for versioned collection plans."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

from data.common.task_identity import TASKS, TASK_BY_ID
from data.sim.generation.plans import expected_counts, expected_roots, work_root


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def repository_root() -> Path:
    """Return the source checkout root independently of cwd/config location."""

    return REPOSITORY_ROOT


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    prompt: str
    aliases: tuple[str, ...]
    episodes: int
    clean_episodes: int
    distractor_episodes: int
    base_seed: int
    required_active_objects: tuple[str, ...]
    closed_gripper_raw: float | None
    grasp_tcp_offset_from_object_m: float | None


@dataclass(frozen=True)
class PickVerificationConfig:
    entry_lift_height_m: float
    minimum_lift_height_m: float
    maximum_relative_downward_slip_m: float
    maximum_final_relative_downward_slip_m: float
    maximum_final_downward_speed_mps: float
    maximum_grasp_region_delta_m: float
    steps: int
    action_dt_s: float
    velocity_fit_samples: int


@dataclass(frozen=True)
class PlaceInitialGraspConfig:
    steps: int
    action_dt_s: float
    maximum_relative_drift_m: float
    maximum_grasp_region_delta_m: float
    minimum_height_above_table_m: float
    gripper_raw: float
    tcp_to_pepper_translation_m: tuple[float, float, float]
    tcp_to_pepper_quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class PlaceVerificationConfig:
    steps: int
    action_dt_s: float
    ring_radius_m: float
    maximum_height_above_table_m: float
    maximum_final_speed_mps: float
    velocity_fit_samples: int


@dataclass(frozen=True)
class OutputRoots:
    raw: Path
    converted: Path
    smoke: Path
    log: Path


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    schema_version: int
    dataset_version: str
    overwrite_existing_outputs: bool
    camera_config: Path
    task_scene_config: Path
    tasks: tuple[TaskPlan, ...]
    action_hz: int
    object_xy_range_m: float
    object_yaw_range_deg: float
    joint_noise_rad: float
    max_attempts_per_episode: int
    seed_retry_stride: int
    scene_variant: str
    distractor_count: int
    record_all_smoke_videos: bool
    representative_video_every: int
    output_schema_version: str
    pick: PickVerificationConfig
    place_initial: PlaceInitialGraspConfig
    place: PlaceVerificationConfig
    outputs: OutputRoots

    @property
    def total_episodes(self) -> int:
        return sum(task.episodes for task in self.tasks)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _absolute_path(value: Any, name: str, *, relative_to: Path) -> Path:
    expanded = str(value).replace("${XARM_WORK_ROOT}", str(work_root()))
    expanded = os.path.expandvars(expanded)
    if "$" in expanded or "%" in expanded:
        raise ValueError(f"{name} contains an unresolved environment variable: {value}")
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    resolved = path.resolve(strict=False)
    if not resolved.is_absolute():
        raise ValueError(f"{name} did not resolve to an absolute path")
    return resolved


def load_pipeline_config(path: Path) -> PipelineConfig:
    path = Path(path).expanduser().resolve()
    data = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    collection = _mapping(data.get("collection"), "collection")
    verification = _mapping(data.get("verification"), "verification")
    pick = _mapping(verification.get("pick_stable_grasp"), "pick_stable_grasp")
    place_initial = _mapping(
        verification.get("place_initial_grasp"), "place_initial_grasp"
    )
    place = _mapping(verification.get("place_stable_release"), "place_stable_release")
    recording = _mapping(data.get("recording"), "recording")
    outputs = _mapping(data.get("outputs"), "outputs")
    base = repository_root()

    task_rows = _mapping(data.get("tasks"), "tasks")
    plans: list[TaskPlan] = []
    for definition in TASKS:
        row = _mapping(task_rows.get(definition.task_id), definition.task_id)
        oracle = _mapping(row.get("oracle") or {}, f"{definition.task_id}.oracle")
        plan = TaskPlan(
            task_id=definition.task_id,
            prompt=str(row.get("prompt")),
            aliases=tuple(str(value) for value in row.get("aliases") or ()),
            episodes=int(row.get("episodes", -1)),
            clean_episodes=int(row.get("clean_episodes", -1)),
            distractor_episodes=int(row.get("distractor_episodes", -1)),
            base_seed=int(row.get("base_seed", -1)),
            required_active_objects=tuple(
                str(value) for value in row.get("required_active_objects") or ()
            ),
            closed_gripper_raw=(
                float(oracle["closed_gripper_raw"])
                if "closed_gripper_raw" in oracle
                else None
            ),
            grasp_tcp_offset_from_object_m=(
                float(oracle["grasp_tcp_offset_from_object_m"])
                if "grasp_tcp_offset_from_object_m" in oracle
                else None
            ),
        )
        if plan.prompt != definition.prompt:
            raise ValueError(
                f"{plan.task_id} prompt must be exactly {definition.prompt!r}"
            )
        if plan.aliases != definition.aliases:
            raise ValueError(f"{plan.task_id} aliases differ from the task registry")
        if plan.required_active_objects != definition.required_active_objects:
            raise ValueError(
                f"{plan.task_id} required objects differ from the task registry"
            )
        if plan.episodes <= 0 or plan.clean_episodes != plan.episodes:
            raise ValueError(f"{plan.task_id} must contain only clean episodes")
        if plan.distractor_episodes != 0:
            raise ValueError(f"{plan.task_id} distractor episodes must be zero")
        if definition.kind == "pick":
            if (
                plan.closed_gripper_raw is None
                or plan.grasp_tcp_offset_from_object_m is None
            ):
                raise ValueError(
                    f"{plan.task_id} requires explicit Pick oracle grasp parameters"
                )
            if not 50.0 <= plan.closed_gripper_raw <= 845.0:
                raise ValueError(f"{plan.task_id} closed_gripper_raw is out of range")
            if not -0.05 <= plan.grasp_tcp_offset_from_object_m <= 0.05:
                raise ValueError(
                    f"{plan.task_id} grasp_tcp_offset_from_object_m is out of range"
                )
        elif oracle:
            raise ValueError(f"{plan.task_id} does not use Pick oracle parameters")
        plans.append(plan)
    if set(task_rows) != set(TASK_BY_ID):
        raise ValueError("Config must contain exactly the six canonical task IDs")

    config = PipelineConfig(
        path=path,
        schema_version=int(data.get("schema_version", -1)),
        dataset_version=str(data.get("dataset_version", "")),
        overwrite_existing_outputs=bool(data.get("overwrite_existing_outputs", False)),
        camera_config=_absolute_path(
            data.get("camera_config"), "camera_config", relative_to=base
        ),
        task_scene_config=_absolute_path(
            data.get("task_scene_config"), "task_scene_config", relative_to=base
        ),
        tasks=tuple(plans),
        action_hz=int(collection.get("action_hz", -1)),
        object_xy_range_m=float(collection.get("object_xy_range_m", -1.0)),
        object_yaw_range_deg=float(collection.get("object_yaw_range_deg", -1.0)),
        joint_noise_rad=float(collection.get("joint_noise_rad", -1.0)),
        max_attempts_per_episode=int(collection.get("max_attempts_per_episode", -1)),
        seed_retry_stride=int(collection.get("seed_retry_stride", -1)),
        scene_variant=str(collection.get("scene_variant", "")),
        distractor_count=int(collection.get("distractor_count", -1)),
        record_all_smoke_videos=bool(recording.get("record_all_smoke_videos", True)),
        representative_video_every=int(recording.get("representative_video_every", 0)),
        output_schema_version=str(data.get("output_schema_version", "")),
        pick=PickVerificationConfig(
            entry_lift_height_m=float(pick["entry_lift_height_m"]),
            minimum_lift_height_m=float(pick["minimum_lift_height_m"]),
            maximum_relative_downward_slip_m=float(
                pick["maximum_relative_downward_slip_m"]
            ),
            maximum_final_relative_downward_slip_m=float(
                pick["maximum_final_relative_downward_slip_m"]
            ),
            maximum_final_downward_speed_mps=float(
                pick["maximum_final_downward_speed_mps"]
            ),
            maximum_grasp_region_delta_m=float(pick["maximum_grasp_region_delta_m"]),
            steps=int(pick["steps"]),
            action_dt_s=float(pick["action_dt_s"]),
            velocity_fit_samples=int(pick["velocity_fit_samples"]),
        ),
        place_initial=PlaceInitialGraspConfig(
            steps=int(place_initial["steps"]),
            action_dt_s=float(place_initial["action_dt_s"]),
            maximum_relative_drift_m=float(place_initial["maximum_relative_drift_m"]),
            maximum_grasp_region_delta_m=float(
                place_initial["maximum_grasp_region_delta_m"]
            ),
            minimum_height_above_table_m=float(
                place_initial["minimum_height_above_table_m"]
            ),
            gripper_raw=float(place_initial["gripper_raw"]),
            tcp_to_pepper_translation_m=tuple(
                float(value) for value in place_initial["tcp_to_pepper_translation_m"]
            ),
            tcp_to_pepper_quaternion_wxyz=tuple(
                float(value) for value in place_initial["tcp_to_pepper_quaternion_wxyz"]
            ),
        ),
        place=PlaceVerificationConfig(
            steps=int(place["steps"]),
            action_dt_s=float(place["action_dt_s"]),
            ring_radius_m=float(place["ring_radius_m"]),
            maximum_height_above_table_m=float(place["maximum_height_above_table_m"]),
            maximum_final_speed_mps=float(place["maximum_final_speed_mps"]),
            velocity_fit_samples=int(place["velocity_fit_samples"]),
        ),
        outputs=OutputRoots(
            raw=_absolute_path(outputs.get("raw"), "outputs.raw", relative_to=base),
            converted=_absolute_path(
                outputs.get("converted"), "outputs.converted", relative_to=base
            ),
            smoke=_absolute_path(
                outputs.get("smoke"), "outputs.smoke", relative_to=base
            ),
            log=_absolute_path(outputs.get("log"), "outputs.log", relative_to=base),
        ),
    )
    validate_pipeline_config(config)
    return config


def validate_pipeline_config(config: PipelineConfig) -> None:
    plan_counts = expected_counts(config.dataset_version)
    if config.schema_version != 1 or not config.dataset_version:
        raise ValueError("Unsupported or missing pipeline schema/dataset version")
    if {task.task_id: task.episodes for task in config.tasks} != plan_counts:
        raise ValueError(f"Task episode counts do not match {config.dataset_version}")
    if config.total_episodes != sum(plan_counts.values()):
        raise ValueError(f"Collection total does not match {config.dataset_version}")
    if config.action_hz != 10 or config.scene_variant != "clean":
        raise ValueError("This version requires 10 Hz clean-scene collection")
    if config.distractor_count != 0:
        raise ValueError("This version requires distractor_count=0")
    if config.max_attempts_per_episode < 1 or config.seed_retry_stride < 1:
        raise ValueError("Retry values must be positive")
    if not config.record_all_smoke_videos or config.representative_video_every < 1:
        raise ValueError("Smoke and representative video sampling must be enabled")
    if len({task.base_seed for task in config.tasks}) != len(config.tasks):
        raise ValueError("Every task requires a distinct base seed range")
    if config.pick.steps != 20 or config.pick.action_dt_s != 0.1:
        raise ValueError("Pick verification must be exactly 20 steps at 0.1 s")
    if config.place_initial.steps != 10 or config.place_initial.action_dt_s != 0.1:
        raise ValueError("Place initial validation must be exactly 10 steps at 0.1 s")
    if config.place.steps != 20 or config.place.action_dt_s != 0.1:
        raise ValueError("Place verification must be exactly 20 steps at 0.1 s")
    if len(config.place_initial.tcp_to_pepper_translation_m) != 3:
        raise ValueError("TCP-to-pepper translation must contain three values")
    if len(config.place_initial.tcp_to_pepper_quaternion_wxyz) != 4:
        raise ValueError("TCP-to-pepper quaternion must contain four values")
    if not 50.0 <= config.place_initial.gripper_raw <= 845.0:
        raise ValueError("Place initial gripper raw command must be in [50, 845]")
    prompts = [task.prompt for task in config.tasks]
    if len(set(prompts)) != 6 or any("_" in prompt for prompt in prompts):
        raise ValueError("Canonical prompts must be unique natural-language strings")
    if any(
        not task.prompt.startswith("pick up ")
        for task in config.tasks
        if TASK_BY_ID[task.task_id].kind == "pick"
    ):
        raise ValueError("Every Pick prompt must begin with 'pick up '")
    if not config.camera_config.is_file() or not config.task_scene_config.is_file():
        raise FileNotFoundError("Camera and task-scene configuration files must exist")
    if set(vars(config.outputs).values()) != expected_roots(config.dataset_version):
        raise ValueError(
            "Configured outputs must be the four exact roots for the dataset version"
        )
    scene_config = _mapping(
        yaml.safe_load(config.task_scene_config.read_text(encoding="utf-8")),
        "task scene config",
    )
    scene_tasks = _mapping(scene_config.get("tasks"), "task scene tasks")
    for task in config.tasks:
        scene = _mapping(scene_tasks.get(task.task_id), task.task_id)
        if str(scene.get("prompt")) != task.prompt:
            raise ValueError(f"Task-scene prompt mismatch for {task.task_id}")
        if tuple(scene.get("active_bodies") or ()) != task.required_active_objects:
            raise ValueError(f"Task-scene active objects mismatch for {task.task_id}")
    place_scene = _mapping(
        scene_tasks.get("place_red_pepper_in_ring"),
        "place_red_pepper_in_ring",
    )
    if float(place_scene.get("initial_gripper_raw", -1.0)) != (
        config.place_initial.gripper_raw
    ):
        raise ValueError("Place gripper raw command differs across configs")
    if tuple(place_scene.get("active_bodies") or ()) != (
        "held_red_pepper",
        "ring",
    ):
        raise ValueError("Place must use the LOCAL held-pepper reset convention")
