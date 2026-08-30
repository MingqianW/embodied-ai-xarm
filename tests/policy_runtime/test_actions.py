from __future__ import annotations

import unittest

import numpy as np

from policy_runtime.action_decoder import (
    action_prefix,
    decode_policy_response,
    validate_policy_actions,
)


class ActionDecoderTests(unittest.TestCase):
    def test_decodes_ten_by_seven_chunk(self) -> None:
        actions = np.zeros((10, 7), dtype=np.float64)
        chunk = decode_policy_response({"actions": actions}, inference_latency_s=0.2)
        self.assertEqual(chunk.actions.shape, (10, 7))
        self.assertEqual(chunk.actions.dtype, np.float32)
        self.assertEqual(chunk.inference_latency_s, 0.2)

    def test_rejects_wrong_shape_and_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_policy_actions(np.zeros((1, 7), dtype=np.float32))
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[2, 3] = np.inf
        with self.assertRaises(ValueError):
            validate_policy_actions(actions)

    def test_prefix_is_bounded(self) -> None:
        actions = np.arange(70, dtype=np.float32).reshape(10, 7)
        np.testing.assert_array_equal(action_prefix(actions, 2), actions[:2])
        with self.assertRaises(ValueError):
            action_prefix(actions, 11)


if __name__ == "__main__":
    unittest.main()
