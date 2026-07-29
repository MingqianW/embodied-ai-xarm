from __future__ import annotations

import unittest
from pathlib import Path

from sim_mujoco.paths import (
    active_model_path,
    camera_config_path,
    mujoco_dataset_root,
    mujoco_output_root,
    repository_root,
    task_config_path,
)


class MuJoCoPathTests(unittest.TestCase):
    def test_repository_relative_defaults_exist(self) -> None:
        self.assertTrue(active_model_path().is_file())
        self.assertTrue(camera_config_path().is_file())
        self.assertTrue(task_config_path().is_file())

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
            active_model_path(environ),
            root.resolve() / "sim_mujoco" / "assets" / "xarm6" / "xarm6_pick_scene.xml",
        )


if __name__ == "__main__":
    unittest.main()
