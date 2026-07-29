from __future__ import annotations

import unittest

import numpy as np

from policy_runtime.safety import SafetyConfig, validate_action_chunk


class SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = np.asarray([0, -0.6, -1.2, 0, 1.8, 0, 845], dtype=np.float32)
        self.limits = np.asarray([[-3.2, 3.2]] * 6, dtype=np.float32)

    def test_validates_and_clamps_entire_chunk_sequentially(self) -> None:
        actions = np.tile(self.state, (10, 1))
        actions[:, 0] = np.linspace(0.01, 0.1, 10)
        result = validate_action_chunk(
            actions,
            self.state,
            self.limits,
            SafetyConfig(max_joint_delta_rad=0.05),
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.clipped)

    def test_rejects_heavy_clip(self) -> None:
        actions = np.tile(self.state, (10, 1))
        actions[0, 0] = 2.0
        result = validate_action_chunk(
            actions,
            self.state,
            self.limits,
            SafetyConfig(
                max_joint_delta_rad=0.05,
                reject_if_clip_exceeds_rad=0.25,
            ),
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.clipped)
        self.assertIn("above rejection threshold", result.reason or "")

    def test_absolute_and_delta_modes_are_explicit(self) -> None:
        actions = np.zeros((2, 7), dtype=np.float32)
        actions[:, 6] = 845
        result = validate_action_chunk(
            actions,
            np.zeros(7, dtype=np.float32),
            self.limits,
            SafetyConfig(action_mode="joint_position_delta", max_joint_delta_rad=0.05),
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.clipped)

    def test_rejects_nan_without_clipping(self) -> None:
        actions = np.tile(self.state, (10, 1))
        actions[0, 0] = np.nan
        result = validate_action_chunk(actions, self.state, self.limits)
        self.assertFalse(result.accepted)
        self.assertFalse(result.clipped)


if __name__ == "__main__":
    unittest.main()
