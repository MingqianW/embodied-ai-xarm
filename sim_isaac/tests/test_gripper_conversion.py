from __future__ import annotations

import unittest
from pathlib import Path

from sim_isaac.articulation import (
    gripper_mm_to_isaac,
    isaac_gripper_to_mm,
    load_robot_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GripperConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = load_robot_mapping(
            PROJECT_ROOT / "sim_isaac" / "config" / "robot.yaml"
        )

    def test_endpoints_and_clipping(self) -> None:
        self.assertAlmostEqual(gripper_mm_to_isaac(50, self.mapping), 0.85)
        self.assertAlmostEqual(gripper_mm_to_isaac(447.5, self.mapping), 0.425)
        self.assertAlmostEqual(gripper_mm_to_isaac(845, self.mapping), 0.0)
        self.assertAlmostEqual(gripper_mm_to_isaac(-100, self.mapping), 0.85)
        self.assertAlmostEqual(gripper_mm_to_isaac(2000, self.mapping), 0.0)
        self.assertAlmostEqual(isaac_gripper_to_mm(0.85, self.mapping), 50.0)
        self.assertAlmostEqual(isaac_gripper_to_mm(0.425, self.mapping), 447.5)
        self.assertAlmostEqual(isaac_gripper_to_mm(0.0, self.mapping), 845.0)
        self.assertAlmostEqual(isaac_gripper_to_mm(1.0, self.mapping), 50.0)
        self.assertAlmostEqual(isaac_gripper_to_mm(-1.0, self.mapping), 845.0)

    def test_round_trip(self) -> None:
        for value in (50.0, 100.0, 400.0, 845.0):
            converted = gripper_mm_to_isaac(value, self.mapping)
            self.assertAlmostEqual(
                isaac_gripper_to_mm(converted, self.mapping),
                value,
                places=5,
            )

    def test_config_marks_physical_mapping_validated(self) -> None:
        self.assertTrue(self.mapping.physical_aperture_validated)


if __name__ == "__main__":
    unittest.main()
