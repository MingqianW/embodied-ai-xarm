from __future__ import annotations

import hashlib
import unittest

import mujoco
import numpy as np

from simulation.observation.cameras import camera_axes
from simulation.resources import DEFAULT_MODEL_PATH
from simulation.robot.model import BASE_CAMERA_NAME
from simulation.robot.model import WRIST_CAMERA_NAME
from simulation.robot.model import camera_id
from simulation.runtime import load_simulation
from simulation.tools import build_xarm6_pick_scene


BASELINE_XML_SHA256 = (
    "0fdec8fa8ec0de4ce0722482f00d52a06cdc86d89601513ed664ad2cc9be7954"
)


class CanonicalModelContractTests(unittest.TestCase):
    def test_checked_in_mjcf_matches_validated_baseline(self) -> None:
        self.assertEqual(
            hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest(),
            BASELINE_XML_SHA256,
        )
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
        self.assertEqual(
            (model.nq, model.nv, model.nu, model.nbody, model.njnt, model.ngeom),
            (47, 42, 7, 22, 17, 62),
        )
        self.assertEqual((model.nsite, model.ncam), (2, 3))
        self.assertEqual(float(model.opt.timestep), 0.002)
        self.assertEqual(int(model.opt.integrator), 3)
        self.assertEqual(int(model.opt.solver), 2)
        self.assertEqual(int(model.opt.cone), 1)
        self.assertEqual(float(model.opt.impratio), 10.0)
        self.assertEqual(int(model.opt.iterations), 100)
        self.assertEqual(float(model.opt.tolerance), 1e-10)

    def test_builder_validates_without_rewriting_authoritative_mjcf(self) -> None:
        before = hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
        build_xarm6_pick_scene.main()
        after = hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(after, before)
        self.assertFalse(
            DEFAULT_MODEL_PATH.with_name(
                f".{DEFAULT_MODEL_PATH.name}.candidate"
            ).exists()
        )

    def test_runtime_applies_package_owned_camera_calibration(self) -> None:
        context = load_simulation()
        try:
            for name in (BASE_CAMERA_NAME, WRIST_CAMERA_NAME):
                identifier = camera_id(context.model, name)
                parameters = context.config[name]
                np.testing.assert_allclose(
                    context.model.cam_pos[identifier], parameters["position"], atol=0.0
                )
                self.assertEqual(
                    float(context.model.cam_fovy[identifier]),
                    float(parameters["fovy_deg"]),
                )
                expected_rotation = camera_axes(
                    parameters["position"],
                    parameters["target"],
                    parameters.get("roll_deg", 0.0),
                )
                actual_rotation = np.empty(9, dtype=np.float64)
                mujoco.mju_quat2Mat(
                    actual_rotation, context.model.cam_quat[identifier]
                )
                np.testing.assert_allclose(
                    actual_rotation.reshape(3, 3), expected_rotation, atol=1e-12
                )
        finally:
            context.close()


if __name__ == "__main__":
    unittest.main()
