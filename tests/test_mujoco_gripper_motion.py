from __future__ import annotations

import unittest

import mujoco

from sim_mujoco.remote_policy_observation import (
    GRIPPER_LEFT_JOINT,
    initialize_scene,
    joint_qpos,
    load_simulation,
)


class GripperMotionTests(unittest.TestCase):
    def test_gripper_closes_and_reopens(self) -> None:
        context = load_simulation()
        try:
            initialize_scene(context.model, context.data, settle_steps=0)
            context.data.ctrl[6] = 0.0
            for _ in range(500):
                mujoco.mj_step(context.model, context.data)
            closed = joint_qpos(
                context.model,
                context.data,
                GRIPPER_LEFT_JOINT,
            )

            context.data.ctrl[6] = 0.04
            for _ in range(500):
                mujoco.mj_step(context.model, context.data)
            reopened = joint_qpos(
                context.model,
                context.data,
                GRIPPER_LEFT_JOINT,
            )

            self.assertLess(closed, 0.005)
            self.assertGreater(reopened, 0.035)
        finally:
            context.close()


if __name__ == "__main__":
    unittest.main()
