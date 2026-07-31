from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sim_isaac.object_spawning import load_task_config, randomized_object_pose


ROOT = Path(__file__).resolve().parents[2]


class ObjectSpawningTests(unittest.TestCase):
    def test_default_pose_is_deterministic(self) -> None:
        config = load_task_config(ROOT / "sim_isaac/config/tasks.yaml")
        first = randomized_object_pose(config, 7)
        second = randomized_object_pose(config, 7)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        np.testing.assert_allclose(first[0], config.object_position_m)

    def test_table_top(self) -> None:
        config = load_task_config(ROOT / "sim_isaac/config/tasks.yaml")
        self.assertAlmostEqual(config.table_top_z_m, 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
