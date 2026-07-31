from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from sim_isaac.articulation import (
    load_robot_mapping,
    policy_state_to_isaac,
    policy_state_to_isaac_reset,
    validate_articulation_joint_names,
    validate_robot_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIG = PROJECT_ROOT / "sim_isaac" / "config" / "robot.yaml"


class JointMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = load_robot_mapping(ROBOT_CONFIG)
        self.names = [
            "extra",
            "joint3",
            "joint1",
            "drive_joint",
            "joint2",
            "joint6",
            "joint5",
            "joint4",
            "left_finger_joint",
            "left_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_finger_joint",
            "right_inner_knuckle_joint",
        ]

    def test_mapping_is_one_to_one(self) -> None:
        validate_robot_mapping(self.mapping)
        indices = validate_articulation_joint_names(self.names, self.mapping)
        self.assertEqual(indices["joint1"], 2)
        self.assertEqual(indices["drive_joint"], 3)

    def test_missing_joint_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required joint"):
            validate_articulation_joint_names(self.names[:-1], self.mapping)

    def test_policy_state_targets_correct_indices(self) -> None:
        state = np.asarray([1, 2, 3, 4, 5, 6, 845], dtype=np.float32)
        targets, indices = policy_state_to_isaac(state, self.names, self.mapping)
        np.testing.assert_allclose(targets[:6], state[:6])
        self.assertAlmostEqual(float(targets[6]), 0.0, places=6)
        self.assertEqual(indices.tolist(), [2, 4, 1, 7, 6, 5, 3])

    def test_reset_state_initializes_passive_gripper_mimics(self) -> None:
        state = np.asarray([1, 2, 3, 4, 5, 6, 845], dtype=np.float32)
        targets, indices = policy_state_to_isaac_reset(
            state,
            self.names,
            self.mapping,
        )
        self.assertEqual(len(targets), 12)
        self.assertEqual(len(indices), 12)
        np.testing.assert_allclose(targets[7:], 0.0, atol=1e-6)
        self.assertEqual(
            indices[7:].tolist(),
            [
                self.names.index(name)
                for name in self.mapping.gripper_mimic_joint_names
            ],
        )

    def test_duplicate_mapping_is_rejected(self) -> None:
        broken = replace(
            self.mapping,
            isaac_arm_joint_names=("joint1", "joint1", "joint3", "joint4", "joint5", "joint6"),
        )
        with self.assertRaisesRegex(ValueError, "not one-to-one"):
            validate_robot_mapping(broken)


if __name__ == "__main__":
    unittest.main()
