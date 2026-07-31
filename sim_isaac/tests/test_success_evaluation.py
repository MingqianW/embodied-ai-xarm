from __future__ import annotations

import unittest

import numpy as np

from sim_isaac.success_evaluation import evaluate_lift


class SuccessEvaluationTests(unittest.TestCase):
    def test_success_and_partial_credit(self) -> None:
        initial = np.asarray([0.4, 0.0, 0.08])
        success = evaluate_lift(initial + [0.0, 0.0, 0.09], initial)
        partial = evaluate_lift(initial + [0.0, 0.0, 0.04], initial)
        none = evaluate_lift(initial, initial)
        self.assertTrue(success.success)
        self.assertEqual(success.score, 1.0)
        self.assertFalse(partial.success)
        self.assertAlmostEqual(partial.score, 0.5)
        self.assertEqual(none.score, 0.0)


if __name__ == "__main__":
    unittest.main()
