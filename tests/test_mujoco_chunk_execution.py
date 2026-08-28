from __future__ import annotations

import unittest

from evaluation.sim.legacy.run_remote_policy_closed_loop import (
    MAX_EXECUTE_CHUNK_STEPS,
    validate_execute_chunk_steps,
)


class ChunkExecutionTests(unittest.TestCase):
    def test_accepts_full_policy_horizon(self) -> None:
        self.assertEqual(MAX_EXECUTE_CHUNK_STEPS, 10)
        for value in (1, 5, 10):
            self.assertEqual(validate_execute_chunk_steps(value), value)

    def test_rejects_values_outside_policy_horizon(self) -> None:
        for value in (0, -1, 11):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_execute_chunk_steps(value)


if __name__ == "__main__":
    unittest.main()
