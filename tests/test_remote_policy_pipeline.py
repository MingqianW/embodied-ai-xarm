from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.robot.control import (  # noqa: E402
    ACTION_SHAPE,
    clamp_gripper_raw,
    clamp_joint_target,
    extract_first_action,
    rate_limit_gripper_raw,
    validate_policy_actions,
)
from simulation.robot.model import ARM_JOINT_NAMES  # noqa: E402
from simulation.observation.policy import build_policy_observation  # noqa: E402
from simulation.robot.gripper import actuator_ctrl_from_raw_hardware  # noqa: E402
from simulation.runtime import initialize_scene  # noqa: E402
from simulation.runtime import load_simulation  # noqa: E402
from simulation.configuration import load_simulation_config  # noqa: E402
from simulation.robot.gripper_mapping import actuator_ctrl_rad_to_raw_hardware  # noqa: E402
from simulation.environment import MuJoCoEnvironment  # noqa: E402


class GripperMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_simulation_config()

    def test_raw_to_sim_conversion(self) -> None:
        self.assertAlmostEqual(
            actuator_ctrl_from_raw_hardware(50.0, self.config), 0.8, places=6
        )
        self.assertAlmostEqual(
            actuator_ctrl_from_raw_hardware(845.0, self.config), 0.005, places=6
        )

    def test_sim_to_raw_conversion(self) -> None:
        self.assertAlmostEqual(
            actuator_ctrl_rad_to_raw_hardware(0.8, self.config), 50.0, places=3
        )
        self.assertAlmostEqual(
            actuator_ctrl_rad_to_raw_hardware(0.005, self.config), 845.0, places=3
        )

    def test_round_trip_consistency(self) -> None:
        for value in (50.0, 150.0, 400.0, 845.0):
            ctrl = actuator_ctrl_from_raw_hardware(value, self.config)
            raw = actuator_ctrl_rad_to_raw_hardware(ctrl, self.config)
            self.assertAlmostEqual(raw, value, places=5)


class ActionSafetyTests(unittest.TestCase):
    def test_validate_policy_actions_accepts_expected_shape(self) -> None:
        actions = np.zeros(ACTION_SHAPE, dtype=np.float32)
        validated = validate_policy_actions(actions)
        self.assertEqual(validated.shape, ACTION_SHAPE)
        self.assertEqual(validated.dtype, np.float32)

    def test_validate_policy_actions_rejects_nan(self) -> None:
        actions = np.zeros(ACTION_SHAPE, dtype=np.float32)
        actions[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_policy_actions(actions)

    def test_validate_policy_actions_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            validate_policy_actions(np.zeros((1, 7), dtype=np.float32))

    def test_extract_first_action(self) -> None:
        actions = np.arange(70, dtype=np.float32).reshape(ACTION_SHAPE)
        first = extract_first_action(actions)
        np.testing.assert_array_equal(first, actions[0])

    def test_joint_limit_and_step_clamping(self) -> None:
        raw = np.asarray([1.0, -1.0, 0.2, 0.0, 2.0, -2.0], dtype=np.float32)
        current = np.zeros(6, dtype=np.float32)
        joint_limits = np.asarray([[-0.2, 0.2]] * 6, dtype=np.float32)
        actuator_limits = np.asarray([[-0.3, 0.3]] * 6, dtype=np.float32)
        clamped, messages = clamp_joint_target(
            raw,
            current,
            joint_limits,
            actuator_limits,
            max_joint_step=0.05,
        )
        np.testing.assert_allclose(
            clamped, np.asarray([0.05, -0.05, 0.05, 0.0, 0.05, -0.05], dtype=np.float32)
        )
        self.assertTrue(messages)

    def test_gripper_clamping(self) -> None:
        self.assertEqual(clamp_gripper_raw(10.0)[0], 50.0)
        self.assertEqual(clamp_gripper_raw(900.0)[0], 845.0)
        self.assertEqual(clamp_gripper_raw(400.0)[0], 400.0)

    def test_gripper_rate_limit_uses_direction_specific_real_il_rates(self) -> None:
        closing, closing_messages = rate_limit_gripper_raw(
            100.0, 400.0, control_dt_s=0.1
        )
        opening, opening_messages = rate_limit_gripper_raw(
            800.0, 400.0, control_dt_s=0.1
        )

        self.assertAlmostEqual(closing, 375.6)
        self.assertAlmostEqual(opening, 422.0)
        self.assertTrue(closing_messages)
        self.assertTrue(opening_messages)

    def test_json_serialization(self) -> None:
        payload = {"state": np.zeros(7, dtype=np.float32).tolist()}
        encoded = json.dumps(payload)
        self.assertIn("state", encoded)


class ObservationTests(unittest.TestCase):
    def test_observation_shapes_dtypes_and_state_ordering(self) -> None:
        context = load_simulation()
        try:
            initialize_scene(context.model, context.data)
            observation = build_policy_observation(
                context.model,
                context.data,
                context.renderer,
                context.config,
                "pick up the object",
            )
            self.assertEqual(observation["observation/image"].shape, (224, 224, 3))
            self.assertEqual(
                observation["observation/wrist_image"].shape, (224, 224, 3)
            )
            self.assertEqual(observation["observation/image"].dtype, np.uint8)
            self.assertEqual(observation["observation/wrist_image"].dtype, np.uint8)
            self.assertEqual(observation["observation/state"].shape, (7,))
            self.assertEqual(observation["observation/state"].dtype, np.float32)
            self.assertEqual(tuple(f"joint{i}" for i in range(1, 7)), ARM_JOINT_NAMES)
        finally:
            context.close()

    def test_environment_adapter_canonical_contract(self) -> None:
        with MuJoCoEnvironment(settle_steps=1) as environment:
            observation = environment.reset(seed=7)
            self.assertEqual(observation.state.shape, (7,))
            self.assertEqual(environment.joint_limits.shape, (6, 2))
            environment.hold_position()
            before = float(environment.context.data.time)
            environment.step_physics(0.004)
            self.assertGreater(float(environment.context.data.time), before)
            self.assertTrue(environment.is_safe())
            frames = environment.recording_frames()
            self.assertEqual(set(frames), {"viewer", "base", "wrist"})


if __name__ == "__main__":
    unittest.main()
