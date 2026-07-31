from __future__ import annotations

import unittest

import numpy as np

from policy_runtime.image_preprocessing import preprocess_policy_image
from sim_isaac.scripts.compare_mujoco_isaac import _comparison


class CameraComparisonTests(unittest.TestCase):
    def test_side_by_side_overlay_and_metadata(self) -> None:
        mujoco = np.zeros((24, 32, 3), dtype=np.uint8)
        isaac = np.full((12, 16, 3), 100, dtype=np.uint8)
        side, overlay, metadata = _comparison(
            mujoco,
            isaac,
            alpha=0.5,
            landmarks={"mujoco": [[5, 5]], "isaac": [[3, 3]]},
        )
        self.assertEqual(side.shape, (24, 48, 3))
        self.assertEqual(overlay.shape, mujoco.shape)
        self.assertGreater(metadata["mean_absolute_pixel_difference"], 0)
        self.assertEqual(metadata["mujoco"]["color_order"], "RGB")

    def test_shared_preprocessing_normalizes_raw_reference(self) -> None:
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        policy = preprocess_policy_image(raw)
        self.assertEqual(policy.shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()
