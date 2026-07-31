from __future__ import annotations

import importlib
import sys
import unittest


class ImportBoundaryTests(unittest.TestCase):
    def test_package_import_does_not_import_isaac_modules(self) -> None:
        before = {name for name in sys.modules if name == "isaacsim" or name.startswith("omni")}
        module = importlib.import_module("sim_isaac")
        after = {name for name in sys.modules if name == "isaacsim" or name.startswith("omni")}
        self.assertEqual(before, after)
        self.assertTrue(module.ISAAC_ADAPTER_VERSION)

    def test_dependency_status_is_non_importing(self) -> None:
        dependencies = importlib.import_module("sim_isaac.dependencies")
        status = dependencies.isaac_module_status()
        self.assertIn("isaacsim", status.available)
        self.assertIn("omni", status.available)
        self.assertIn("pxr", status.available)

    def test_adapter_modules_are_importable_without_isaac(self) -> None:
        before = {
            name
            for name in sys.modules
            if name == "isaacsim" or name.startswith(("omni", "pxr"))
        }
        for name in (
            "sim_isaac.articulation",
            "sim_isaac.asset_preparation",
            "sim_isaac.cameras",
            "sim_isaac.environment",
            "sim_isaac.object_spawning",
            "sim_isaac.recording",
            "sim_isaac.scene",
            "sim_isaac.success_evaluation",
            "sim_isaac.version_compat",
        ):
            importlib.import_module(name)
        after = {
            name
            for name in sys.modules
            if name == "isaacsim" or name.startswith(("omni", "pxr"))
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
