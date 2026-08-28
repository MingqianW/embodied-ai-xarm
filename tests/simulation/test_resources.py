from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sim_mujoco.paths import mujoco_dataset_root
from sim_mujoco.paths import mujoco_output_root
from simulation.resources import camera_config_path
from simulation.resources import gripper_config_path
from simulation.resources import model_path
from simulation.resources import repository_root
from simulation.resources import task_config_path


class MuJoCoPathTests(unittest.TestCase):
    def test_repository_relative_defaults_exist(self) -> None:
        self.assertTrue(model_path().is_file())
        self.assertTrue(camera_config_path().is_file())
        self.assertTrue(gripper_config_path().is_file())
        self.assertTrue(task_config_path().is_file())

    def test_package_resources_do_not_depend_on_working_directory(self) -> None:
        expected = model_path()
        original = Path.cwd()
        with TemporaryDirectory() as temporary:
            try:
                import os

                os.chdir(temporary)
                self.assertEqual(model_path(), expected)
                self.assertTrue(camera_config_path().is_file())
                self.assertTrue(gripper_config_path().is_file())
                self.assertTrue(task_config_path().is_file())
            finally:
                os.chdir(original)

    def test_each_simulation_resource_supports_an_explicit_override(self) -> None:
        root = Path.cwd().resolve()
        environ = {
            "XARM_MUJOCO_MODEL_PATH": str(root / "custom.xml"),
            "XARM_CAMERA_CONFIG_PATH": str(root / "camera.yaml"),
            "XARM_GRIPPER_CONFIG_PATH": str(root / "gripper.yaml"),
            "XARM_TASK_CONFIG_PATH": str(root / "tasks.yaml"),
        }
        self.assertEqual(model_path(environ), root / "custom.xml")
        self.assertEqual(camera_config_path(environ), root / "camera.yaml")
        self.assertEqual(gripper_config_path(environ), root / "gripper.yaml")
        self.assertEqual(task_config_path(environ), root / "tasks.yaml")

    def test_environment_overrides_are_composable(self) -> None:
        root = Path.cwd() / "portable-root"
        output = Path.cwd() / "portable-output"
        dataset = Path.cwd() / "portable-dataset"
        environ = {
            "EMBODIED_AI_ROOT": str(root),
            "MUJOCO_OUTPUT_ROOT": str(output),
            "MUJOCO_DATASET_ROOT": str(dataset),
        }
        self.assertEqual(repository_root(environ), root.resolve())
        self.assertEqual(mujoco_output_root(environ), output.resolve())
        self.assertEqual(mujoco_dataset_root(environ), dataset.resolve())
        self.assertEqual(
            model_path(environ),
            Path(__file__).resolve().parents[2]
            / "simulation"
            / "assets"
            / "xarm6"
            / "xarm6_pick_scene.xml",
        )


if __name__ == "__main__":
    unittest.main()
