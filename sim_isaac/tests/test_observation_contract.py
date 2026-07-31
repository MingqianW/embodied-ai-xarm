from __future__ import annotations

import unittest

import numpy as np

from policy_runtime.observation_builder import build_policy_observation


class IsaacObservationContractTests(unittest.TestCase):
    def test_rgba_camera_frames_use_shared_policy_preprocessing(self) -> None:
        rgba = np.ones((240, 320, 4), dtype=np.uint8) * 255
        state = np.asarray([0, -0.6, -1.2, 0, 1.8, 0, 845], dtype=np.float32)
        observation = build_policy_observation(rgba, rgba, state, "pick")
        self.assertEqual(observation.base_image.shape, (224, 224, 3))
        self.assertEqual(observation.base_image.dtype, np.uint8)
        self.assertEqual(observation.color_order, "RGB")


if __name__ == "__main__":
    unittest.main()

