from __future__ import annotations

import csv
import unittest
from pathlib import Path

import mujoco
import numpy as np

from simulation.robot.joint_mapping import (
    mujoco_qpos_to_raw_arm_state,
    raw_arm_state_to_mujoco_qpos,
)
from simulation.robot.model import ARM_JOINT_NAMES
from simulation.runtime import initialize_scene
from simulation.runtime import load_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class JointMappingTests(unittest.TestCase):
    def test_identity_mapping_and_round_trip(self) -> None:
        raw = np.asarray([-0.35, 0.17, -1.01, 0.018, 0.84, -0.37])
        mujoco_qpos = raw_arm_state_to_mujoco_qpos(raw)
        np.testing.assert_array_equal(mujoco_qpos, raw)
        np.testing.assert_array_equal(
            mujoco_qpos_to_raw_arm_state(mujoco_qpos),
            raw,
        )
        self.assertIsNot(mujoco_qpos, raw)

    def test_mapping_rejects_invalid_arm_vectors(self) -> None:
        with self.assertRaises(ValueError):
            raw_arm_state_to_mujoco_qpos(np.zeros(5))
        with self.assertRaises(ValueError):
            mujoco_qpos_to_raw_arm_state(
                np.asarray([0.0, 0.0, 0.0, np.nan, 0.0, 0.0])
            )

    def test_raw_frame_matches_recorded_controller_flange_position(self) -> None:
        from data.real.config import get_raw_data_root

        log_path = (
            get_raw_data_root()
            / "pick_up_the_red_block"
            / "episode_000"
            / "robot_log.csv"
        )
        if not log_path.is_file():
            self.skipTest(f"optional real-data fixture is unavailable: {log_path}")
        with log_path.open("r", encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        raw = np.asarray([float(row[f"j{i}_rad"]) for i in range(1, 7)])
        reference_position = (
            np.asarray(
                [
                    float(row["tcp_x_m"]),
                    float(row["tcp_y_m"]),
                    float(row["tcp_z_m"]),
                ]
            )
            / 1000.0
        )

        context = load_simulation()
        try:
            initialize_scene(context.model, context.data, settle_steps=0)
            mapped = raw_arm_state_to_mujoco_qpos(raw)
            for joint_name, value in zip(ARM_JOINT_NAMES, mapped):
                joint_id = mujoco.mj_name2id(
                    context.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                context.data.qpos[
                    int(context.model.jnt_qposadr[joint_id])
                ] = value
            mujoco.mj_forward(context.model, context.data)
            flange_id = mujoco.mj_name2id(
                context.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "link6",
            )
            error = np.linalg.norm(
                np.asarray(context.data.xpos[flange_id]) - reference_position
            )
            self.assertLess(error, 0.006)
        finally:
            context.close()


if __name__ == "__main__":
    unittest.main()
