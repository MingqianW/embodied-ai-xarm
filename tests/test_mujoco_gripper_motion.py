from __future__ import annotations

import unittest

import mujoco

from sim_mujoco.gripper_mapping import measure_fingertip_aperture_m
from sim_mujoco.remote_policy_observation import (
    initialize_scene,
    load_simulation,
)


class GripperMotionTests(unittest.TestCase):
    def test_gripper_closes_and_reopens(self) -> None:
        context = load_simulation()
        try:
            initialize_scene(context.model, context.data, settle_steps=0)
            context.data.ctrl[6] = 0.80
            for _ in range(500):
                mujoco.mj_step(context.model, context.data)
            closed = measure_fingertip_aperture_m(context.model, context.data)

            context.data.ctrl[6] = 0.005
            for _ in range(500):
                mujoco.mj_step(context.model, context.data)
            reopened = measure_fingertip_aperture_m(context.model, context.data)

            self.assertLess(closed, 0.02)
            self.assertGreater(reopened, 0.08)
        finally:
            context.close()


if __name__ == "__main__":
    unittest.main()
