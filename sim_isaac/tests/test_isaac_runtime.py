from __future__ import annotations

import os
import unittest

import numpy as np

from policy_runtime.observation_builder import validate_policy_observation
from sim_isaac.environment import IsaacEnvironment

try:
    import pytest

    pytestmark = [pytest.mark.isaac, pytest.mark.integration]
except ModuleNotFoundError:
    pytestmark = []


@unittest.skipUnless(
    os.environ.get("RUN_ISAAC_TESTS") == "1",
    "set RUN_ISAAC_TESTS=1 and run with Isaac Sim's Python launcher",
)
class IsaacRuntimeIntegrationTests(unittest.TestCase):
    def test_asset_joint_camera_and_reset_contracts(self) -> None:
        with IsaacEnvironment(headless=True, seed=17) as environment:
            observation = environment.reset(seed=17)
            validate_policy_observation(observation)
            self.assertEqual(
                tuple(environment.scene.robot.joint_names),
                tuple(environment.scene.robot.prim.dof_names),
            )
            self.assertTrue(
                set(environment.mapping.isaac_arm_joint_names).issubset(
                    environment.scene.robot.joint_names
                )
            )
            first_object_position = environment.scene.objects.position()
            second = environment.reset(seed=17)
            np.testing.assert_allclose(
                first_object_position, environment.scene.objects.position()
            )
            np.testing.assert_allclose(observation.state, second.state, atol=1e-4)
            self.assertEqual(second.base_image.shape, (224, 224, 3))
            self.assertEqual(second.wrist_image.shape, (224, 224, 3))

    def test_each_joint_and_gripper_can_be_commanded(self) -> None:
        with IsaacEnvironment(headless=True, seed=17) as environment:
            state = environment.reset(seed=17).state.copy()
            for index in range(6):
                target = state.copy()
                target[index] = np.clip(
                    target[index] + 0.01,
                    environment.joint_limits[index, 0],
                    environment.joint_limits[index, 1],
                )
                environment.apply_action(target)
                environment.step_physics(0.05)
                self.assertTrue(
                    np.isfinite(environment.scene.robot.get_policy_state()).all()
                )
            for gripper in (
                environment.mapping.gripper_policy_closed,
                environment.mapping.gripper_policy_open,
            ):
                target = environment.scene.robot.get_policy_state()
                target[6] = gripper
                environment.apply_action(target)
                environment.step_physics(0.05)
                self.assertTrue(
                    np.isfinite(environment.scene.robot.get_policy_state()[6])
                )
            environment.apply_action(environment.scene.robot.get_policy_state())
            environment.step_physics(0.1)
            refreshed = environment.observe()
            validate_policy_observation(refreshed)
            self.assertTrue(environment.is_safe(), environment.safety_diagnostics())


if __name__ == "__main__":
    unittest.main()
