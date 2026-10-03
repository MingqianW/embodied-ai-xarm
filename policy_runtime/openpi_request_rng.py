"""Stateless request sampling through the pinned OpenPI public noise hook."""

from __future__ import annotations

from typing import Any

import numpy as np

from policy_runtime.action_decoder import validate_policy_actions
from policy_runtime.remote_policy_client import REQUEST_RNG_SEED_KEY


def request_noise(seed: int, horizon: int, internal_action_dim: int) -> np.ndarray:
    # Call only inside the allocation. Importing this module is GPU-safe.
    import jax

    # Pi0.sample_actions at OpenPI 15a9616 samples this exact normal shape.
    # Its explicit noise argument bypasses the stateful Policy RNG stream.
    return np.asarray(jax.random.normal(jax.random.key(seed), (1, horizon, internal_action_dim)))[0]


class RequestRngPolicy:
    """Wrap an actually loaded JAX Pi0/Pi0.5 policy, without changing transforms.

    The caller owns checkpoint/config restoration and provenance verification.
    This adapter never constructs or accepts a formal provenance document.
    """

    def __init__(self, policy: Any, *, horizon: int = 10, internal_action_dim: int = 32) -> None:
        if horizon < 1 or internal_action_dim < 7:
            raise ValueError("Invalid OpenPI sampling dimensions")
        self._policy = policy
        self._horizon = horizon
        self._internal_action_dim = internal_action_dim

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._policy.metadata, "request_rng_required": True}

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        inputs = dict(observation)
        seed = inputs.pop(REQUEST_RNG_SEED_KEY, None)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Each request requires an unsigned 32-bit integer RNG seed")
        noise = request_noise(seed, self._horizon, self._internal_action_dim)
        result = self._policy.infer(inputs, noise=noise)
        validate_policy_actions(result["actions"], action_horizon=self._horizon)
        return result
