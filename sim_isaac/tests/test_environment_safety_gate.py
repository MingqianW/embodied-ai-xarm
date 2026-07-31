from __future__ import annotations

import unittest
from unittest.mock import Mock

from sim_isaac.environment import IsaacEnvironment


class EnvironmentSafetyGateTests(unittest.TestCase):
    def test_safe_state_is_accepted(self) -> None:
        environment = IsaacEnvironment.__new__(IsaacEnvironment)
        environment.is_safe = Mock(return_value=True)
        environment.safety_diagnostics = Mock(return_value={"finite_state": True})

        environment.require_safe("test")

        environment.safety_diagnostics.assert_not_called()

    def test_unsafe_state_is_rejected_with_diagnostics(self) -> None:
        environment = IsaacEnvironment.__new__(IsaacEnvironment)
        environment.is_safe = Mock(return_value=False)
        environment.safety_diagnostics = Mock(
            return_value={"max_joint_tracking_error_rad": 2.25}
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "test preflight failed.*max_joint_tracking_error_rad",
        ):
            environment.require_safe("test")


if __name__ == "__main__":
    unittest.main()
