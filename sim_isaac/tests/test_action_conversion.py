from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from sim_isaac.articulation import (
    isaac_state_to_policy,
    load_robot_mapping,
    policy_action_to_isaac,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ActionConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = load_robot_mapping(
            PROJECT_ROOT / "sim_isaac" / "config" / "robot.yaml"
        )
        self.names = [
            *self.mapping.isaac_arm_joint_names,
            self.mapping.gripper_joint_name,
            *self.mapping.gripper_mimic_joint_names,
        ]

    def test_absolute_action_round_trip(self) -> None:
        action = np.asarray([0, -0.6, -1.2, 0, 1.8, 0, 400], dtype=np.float32)
        targets, indices = policy_action_to_isaac(action, self.names, self.mapping)
        full = np.zeros(len(self.names), dtype=np.float32)
        full[indices] = targets
        state = isaac_state_to_policy(full, self.names, self.mapping)
        np.testing.assert_allclose(state, action, atol=1e-5)

    def test_delta_mode_requires_current_state(self) -> None:
        mapping = replace(self.mapping, action_mode="joint_position_delta")
        with self.assertRaisesRegex(ValueError, "requires current_policy_state"):
            policy_action_to_isaac(np.zeros(7, dtype=np.float32), self.names, mapping)


if __name__ == "__main__":
    unittest.main()
