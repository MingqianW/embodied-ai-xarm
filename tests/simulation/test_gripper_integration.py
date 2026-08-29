from __future__ import annotations

import unittest

import mujoco
import numpy as np

from simulation.robot.model import XARM_FOUR_BAR_JOINT_NAMES
from simulation.robot.gripper import measure_fingertip_aperture_m
from simulation.robot.gripper import set_raw_gripper_configuration
from simulation.robot.gripper_mapping import actuator_ctrl_rad_to_raw_hardware
from simulation.robot.gripper_mapping import raw_hardware_to_actuator_ctrl_rad
from simulation.resources import DEFAULT_MODEL_PATH
from simulation.configuration import load_simulation_config


def named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise AssertionError(f"Missing {object_type}: {name}")
    return int(value)


class MenagerieGripperIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
        cls.config = load_simulation_config()

    def test_exact_compiled_actuator_contract(self) -> None:
        actuator = named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "gripper_actuator",
        )
        self.assertEqual(self.model.nu, 7)
        np.testing.assert_array_equal(
            self.model.actuator_ctrlrange[actuator], [0.005, 0.85]
        )
        np.testing.assert_array_equal(
            self.model.actuator_forcerange[actuator], [-8.0, 8.0]
        )
        np.testing.assert_allclose(
            self.model.actuator_gainprm[actuator, :3], [120.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            self.model.actuator_biasprm[actuator, :3], [0.0, -120.0, 0.0]
        )

    def test_linkage_tendon_constraints_and_pads_exist(self) -> None:
        for name in XARM_FOUR_BAR_JOINT_NAMES:
            named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self.assertEqual(self.model.ntendon, 1)
        named_id(self.model, mujoco.mjtObj.mjOBJ_TENDON, "gripper_split")
        self.assertEqual(self.model.neq, 3)
        equality_names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
            for index in range(self.model.neq)
        }
        self.assertEqual(
            equality_names,
            {"left_finger_linkage", "right_finger_linkage", "symmetric_gripper"},
        )
        for side in ("left", "right"):
            for index in (1, 2):
                geom = named_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{side}_finger_pad_{index}",
                )
                np.testing.assert_allclose(
                    self.model.geom_size[geom], [0.015, 0.002, 0.0095]
                )
                self.assertAlmostEqual(
                    float(self.model.geom_friction[geom, 0]), 2.0
                )
                self.assertEqual(int(self.model.geom_condim[geom]), 4)

    def test_no_legacy_slide_gripper_remains(self) -> None:
        for name in ("left_finger_slide", "right_finger_slide", "symmetric_gripper"):
            self.assertEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name),
                -1,
            )

    def test_raw_ctrl_round_trip_and_kinematic_aperture_are_monotonic(self) -> None:
        data = mujoco.MjData(self.model)
        keyframe = named_id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        openings = []
        for raw in (50, 100, 200, 300, 400, 500, 600, 700, 800, 845):
            mujoco.mj_resetDataKeyframe(self.model, data, keyframe)
            ctrl = raw_hardware_to_actuator_ctrl_rad(raw, self.config)
            self.assertAlmostEqual(
                actuator_ctrl_rad_to_raw_hardware(ctrl, self.config),
                raw,
                places=7,
            )
            set_raw_gripper_configuration(self.model, data, raw, self.config)
            data.ctrl[6] = ctrl
            for _ in range(500):
                mujoco.mj_step(self.model, data)
            openings.append(measure_fingertip_aperture_m(self.model, data))
            left = named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
            right = named_id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint"
            )
            self.assertAlmostEqual(
                float(data.qpos[self.model.jnt_qposadr[left]]),
                float(data.qpos[self.model.jnt_qposadr[right]]),
                places=7,
            )
        self.assertTrue(np.all(np.diff(openings) > 0.0))
        self.assertLess(openings[0], 0.01)
        self.assertGreater(openings[-1], 0.08)


if __name__ == "__main__":
    unittest.main()
