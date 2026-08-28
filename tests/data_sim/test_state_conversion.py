from __future__ import annotations

import unittest

import mujoco
import numpy as np

from data.sim.generation.state_conversion import (
    gripper_hardware_raw_from_mujoco,
    mujoco_gripper_actuator_ctrl_from_hardware_raw,
    mujoco_joint_target_from_policy_action,
    policy_action_from_mujoco_target,
    policy_state_from_mujoco,
)
from simulation.robot.gripper import set_raw_gripper_configuration
from simulation.resources import DEFAULT_MODEL_PATH
from simulation.runtime import initialize_scene
from simulation.configuration import load_simulation_config


class MuJoCoDataConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))

    def setUp(self) -> None:
        self.data = mujoco.MjData(self.model)
        initialize_scene(self.model, self.data, settle_steps=0)

    def test_policy_state_uses_named_arm_joints_and_raw_gripper(self) -> None:
        state = policy_state_from_mujoco(self.model, self.data)
        np.testing.assert_allclose(
            state[:6],
            np.asarray([0.0, -0.6, -1.2, 0.0, 1.8, 0.0], dtype=np.float32),
            atol=1e-7,
        )
        self.assertAlmostEqual(float(state[6]), 845.0, places=3)

    def test_gripper_forward_inverse_round_trip(self) -> None:
        for raw in np.linspace(50.0, 845.0, 33):
            ctrl = mujoco_gripper_actuator_ctrl_from_hardware_raw(float(raw))
            target = np.asarray([0, 0, 0, 0, 0, 0, ctrl], dtype=np.float64)
            recovered = policy_action_from_mujoco_target(target)
            self.assertAlmostEqual(float(recovered[6]), float(raw), places=3)

    def test_policy_action_internal_target_round_trip(self) -> None:
        action = np.asarray(
            [-0.35, 0.42, -1.05, 0.02, 0.65, -0.40, 410.0],
            dtype=np.float32,
        )
        target = mujoco_joint_target_from_policy_action(action)
        recovered = policy_action_from_mujoco_target(target)
        np.testing.assert_allclose(recovered, action, rtol=0.0, atol=2e-5)

    def test_gripper_state_reads_mean_driver_configuration(self) -> None:
        set_raw_gripper_configuration(
            self.model,
            self.data,
            300.0,
            load_simulation_config(),
        )
        mujoco.mj_forward(self.model, self.data)
        self.assertAlmostEqual(
            gripper_hardware_raw_from_mujoco(self.model, self.data),
            300.0,
            places=3,
        )

    def test_gripper_target_uses_local_driver_angle(self) -> None:
        self.assertAlmostEqual(
            mujoco_gripper_actuator_ctrl_from_hardware_raw(300.0), 0.55
        )

    def test_conversion_rejects_wrong_shape_and_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            mujoco_joint_target_from_policy_action(np.zeros(6))
        invalid = np.zeros(7)
        invalid[2] = np.nan
        with self.assertRaises(ValueError):
            mujoco_joint_target_from_policy_action(invalid)


if __name__ == "__main__":
    unittest.main()
