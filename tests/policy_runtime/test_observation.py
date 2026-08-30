from __future__ import annotations

import unittest

import numpy as np

from policy_runtime.image_preprocessing import (
    ImagePreprocessingConfig,
    ensure_rgb_uint8,
    preprocess_policy_image,
)
from policy_runtime.observation_builder import (
    build_policy_observation,
    validate_policy_observation,
)


class ImagePreprocessingTests(unittest.TestCase):
    def test_bgr_float_rgba_and_orientation_handling(self) -> None:
        image = np.zeros((4, 6, 4), dtype=np.float32)
        image[0, 0, :3] = [0.0, 0.0, 1.0]
        rgb = ensure_rgb_uint8(image, input_color_order="BGR")
        self.assertEqual(rgb.shape, (4, 6, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb[0, 0].tolist(), [255, 0, 0])

    def test_shared_resize_with_pad_contract(self) -> None:
        image = np.ones((480, 640, 3), dtype=np.uint8) * 127
        output = preprocess_policy_image(image)
        self.assertEqual(output.shape, (224, 224, 3))
        self.assertEqual(output.dtype, np.uint8)

    def test_crop_and_flip(self) -> None:
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[1, 2] = [1, 2, 3]
        config = ImagePreprocessingConfig(
            crop_top=1,
            crop_bottom=1,
            crop_left=2,
            crop_right=2,
            flip_horizontal=True,
            flip_vertical=True,
        )
        output = preprocess_policy_image(image, config)
        self.assertEqual(output.shape, (224, 224, 3))


class ObservationContractTests(unittest.TestCase):
    def test_builds_exact_openpi_contract(self) -> None:
        image = np.ones((480, 640, 3), dtype=np.uint8)
        state = np.asarray([0, -0.6, -1.2, 0, 1.8, 0, 845], dtype=np.float32)
        observation = build_policy_observation(image, image, state, "pick up the object")
        validate_policy_observation(observation)
        payload = observation.as_openpi_dict()
        self.assertEqual(
            set(payload),
            {"observation/image", "observation/wrist_image", "observation/state", "prompt"},
        )
        self.assertEqual(payload["observation/state"].shape, (7,))

    def test_rejects_nonfinite_state(self) -> None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        state = np.zeros(7, dtype=np.float32)
        state[0] = np.nan
        with self.assertRaises(ValueError):
            build_policy_observation(image, image, state, "test")


if __name__ == "__main__":
    unittest.main()
