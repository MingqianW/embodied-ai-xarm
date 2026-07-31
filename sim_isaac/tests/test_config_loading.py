from __future__ import annotations

import unittest
from pathlib import Path

from policy_runtime.config import load_yaml
from sim_isaac.articulation import load_robot_mapping
from sim_isaac.asset_preparation import load_asset_paths, validate_source_assets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "sim_isaac" / "config"


class IsaacConfigTests(unittest.TestCase):
    def test_all_configs_have_schema_versions(self) -> None:
        for name in ("asset_import.yaml", "robot.yaml", "cameras.yaml", "control.yaml", "tasks.yaml"):
            value = load_yaml(CONFIG_ROOT / name)
            self.assertEqual(value["schema_version"], "1.0", name)

    def test_robot_action_semantics_are_explicit(self) -> None:
        mapping = load_robot_mapping(CONFIG_ROOT / "robot.yaml")
        self.assertEqual(mapping.action_mode, "absolute_joint_position")
        self.assertFalse(mapping.gravity_enabled)
        self.assertEqual(mapping.gripper_color_rgb, (0.015, 0.015, 0.02))

    def test_authoritative_source_assets_validate(self) -> None:
        paths = load_asset_paths(CONFIG_ROOT / "asset_import.yaml", PROJECT_ROOT)
        report = validate_source_assets(paths)
        self.assertTrue(report["valid"], report)
        self.assertGreater(report["mesh_file_count"], 0)


if __name__ == "__main__":
    unittest.main()
