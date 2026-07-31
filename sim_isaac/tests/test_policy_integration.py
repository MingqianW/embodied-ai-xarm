from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from policy_runtime.remote_policy_client import (
    PolicyTimeoutError,
    RemotePolicyClient,
    RemotePolicyConfig,
)
from policy_runtime.runners import (
    ClosedLoopConfig,
    DryLoopConfig,
    run_closed_loop,
    run_dry_loop,
)
from sim_isaac.environment import IsaacEnvironment
from sim_isaac.recording import create_recorder

try:
    import pytest

    pytestmark = [
        pytest.mark.isaac,
        pytest.mark.integration,
        pytest.mark.policy_server,
    ]
except ModuleNotFoundError:
    pytestmark = []


class _RejectedPolicy:
    last_inference_latency_s = 0.0

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        return {"actions": np.full((10, 7), 1e6, dtype=np.float32)}


class _TimeoutPolicy:
    last_inference_latency_s = None

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        raise PolicyTimeoutError("intentional integration timeout")


@unittest.skipUnless(
    os.environ.get("RUN_ISAAC_POLICY_TESTS") == "1",
    "set RUN_ISAAC_POLICY_TESTS=1 and run with Isaac Sim plus a policy server",
)
class IsaacPolicyIntegrationTests(unittest.TestCase):
    def test_dry_repeated_and_short_closed_loop(self) -> None:
        host = os.environ.get("OPENPI_POLICY_HOST", "127.0.0.1")
        port = int(os.environ.get("OPENPI_POLICY_PORT", "18000"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with IsaacEnvironment(headless=True, seed=0) as environment, RemotePolicyClient(
                RemotePolicyConfig(host=host, port=port)
            ) as policy:
                dry = run_dry_loop(
                    environment,
                    policy,
                    DryLoopConfig(
                        prompt="pick up the object",
                        iterations=2,
                        output_dir=root / "dry",
                    ),
                )
                self.assertEqual(dry["iterations"], 2)
                closed = run_closed_loop(
                    environment,
                    policy,
                    ClosedLoopConfig(
                        prompt="pick up the object",
                        max_policy_steps=2,
                        execute_chunk_steps=1,
                        output_dir=root / "closed",
                    ),
                    recorder=create_recorder(root / "recording", max_frames=10),
                )
                self.assertIn(
                    closed["termination_reason"],
                    {"max_policy_steps", "simulation_instability"},
                )
                self.assertTrue((root / "closed/episode.json").is_file())
                self.assertIn("recording", closed)

    def test_timeout_and_rejected_action_hold_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with IsaacEnvironment(headless=True, seed=0) as environment:
                timeout = run_closed_loop(
                    environment,
                    _TimeoutPolicy(),
                    ClosedLoopConfig(
                        prompt="pick up the object",
                        max_policy_steps=1,
                        output_dir=root / "timeout",
                    ),
                )
                self.assertEqual(timeout["termination_reason"], "policy_timeout")
                rejected = run_closed_loop(
                    environment,
                    _RejectedPolicy(),
                    ClosedLoopConfig(
                        prompt="pick up the object",
                        max_policy_steps=1,
                        output_dir=root / "rejected",
                    ),
                )
                self.assertEqual(rejected["termination_reason"], "unsafe_action")


if __name__ == "__main__":
    unittest.main()
