from __future__ import annotations

import unittest

import numpy as np

from policy_runtime.recording import pad_to_aspect, tile_recording_frame


class RecordingTests(unittest.TestCase):
    def test_shared_frame_layout(self) -> None:
        viewer = np.ones((480, 640, 3), dtype=np.uint8)
        camera = np.ones((224, 224, 3), dtype=np.uint8)
        frame = tile_recording_frame(viewer, camera, camera)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_padding_rejects_bad_image(self) -> None:
        with self.assertRaises(ValueError):
            pad_to_aspect(np.zeros((10, 10), dtype=np.uint8), 100, 100)


if __name__ == "__main__":
    unittest.main()
