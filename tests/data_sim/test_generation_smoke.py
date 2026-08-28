from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data.common.validation import validate_training_record
from data.common.schema import XARM_STATE_COLUMNS
from data.sim.generation.collection import _record_attempt, resolve_seed
from data.sim.generation.config import load_pipeline_config
from data.sim.generation.conversion import training_records_from_raw_episode
from simulation.environment import MuJoCoEnvironment


CONFIG_PATH = Path("configs/data/sim/generation/clean_multitask_stable_v3.yaml")


def test_one_deterministic_generation_attempt_reaches_training_contract(
    tmp_path: Path,
) -> None:
    config = load_pipeline_config(CONFIG_PATH)
    task = next(task for task in config.tasks if task.task_id == "red_block")
    resolved_seed = resolve_seed(task, 0, 0, config.seed_retry_stride)
    episode_dir = tmp_path / "episode"

    with MuJoCoEnvironment(
        task=task.task_id,
        prompt=task.prompt,
        camera_config_path=config.camera_config,
        task_scene_config_path=config.task_scene_config,
        object_xy_range=config.object_xy_range_m,
        object_yaw_range_deg=config.object_yaw_range_deg,
        joint_noise=config.joint_noise_rad,
        scene_variant="clean",
    ) as environment:
        success, metadata = _record_attempt(
            environment,
            config=config,
            task=task,
            requested_episode_index=0,
            global_episode_index=0,
            retry_index=0,
            resolved_seed=resolved_seed,
            staging_dir=episode_dir,
        )

    assert success, metadata.get("simulation", {}).get("failure_reason")
    records = training_records_from_raw_episode(
        episode_dir, task_id=task.task_id, episode_index=0
    )
    with (episode_dir / "robot_log.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(records) == len(rows) - 1 > 0
    assert records[0]["source"] == "sim"
    assert records[0]["frame_index"] == records[0]["source_frame_index"] == 0
    assert records[0]["timestamp"] == float(rows[0]["ts"])
    assert records[0]["task"] == task.prompt
    np.testing.assert_array_equal(
        records[0]["state"],
        np.asarray(
            [float(rows[0][name]) for name in XARM_STATE_COLUMNS],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        records[0]["actions"],
        np.asarray(
            [float(rows[1][name]) for name in XARM_STATE_COLUMNS],
            dtype=np.float32,
        ),
    )
    validate_training_record(records[0])
