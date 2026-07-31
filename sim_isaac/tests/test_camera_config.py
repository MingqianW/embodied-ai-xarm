from __future__ import annotations

import unittest
from pathlib import Path

from sim_isaac.cameras import (
    _experimental_resolution,
    _usd_camera_apertures,
    load_camera_configs,
)


ROOT = Path(__file__).resolve().parents[2]


class CameraConfigTests(unittest.TestCase):
    def test_camera_contract(self) -> None:
        configs = load_camera_configs(ROOT / "sim_isaac/config/cameras.yaml")
        self.assertEqual(set(configs), {"base", "wrist"})
        for config in configs.values():
            self.assertEqual(config.resolution, (320, 240))
            self.assertEqual(config.preprocessing.width, 224)
            self.assertEqual(config.preprocessing.height, 224)
            self.assertEqual(config.preprocessing.input_color_order, "RGB")
            self.assertEqual(_experimental_resolution(config), (240, 320))
            horizontal, vertical = _usd_camera_apertures(config)
            self.assertAlmostEqual(horizontal / vertical, 4.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
