from __future__ import annotations

import numpy as np
import pytest

import policy_runtime.openpi_request_rng as adapter
from policy_runtime.remote_policy_client import REQUEST_RNG_SEED_KEY


class LoadedPolicyFixture:
    metadata = {"fixture": True}

    def __init__(self) -> None:
        self.observations: list[dict] = []
        self.noises: list[np.ndarray] = []

    def infer(self, observation: dict, *, noise: np.ndarray) -> dict:
        self.observations.append(observation)
        self.noises.append(noise)
        return {"actions": noise[:, :7]}


def test_request_seed_reaches_sampling_without_becoming_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    seeds = []

    def noise(seed: int, horizon: int, internal_dim: int) -> np.ndarray:
        seeds.append((seed, horizon, internal_dim))
        return np.full((horizon, internal_dim), seed, dtype=np.float32)

    monkeypatch.setattr(adapter, "request_noise", noise)
    upstream = LoadedPolicyFixture()
    policy = adapter.RequestRngPolicy(upstream)
    observation = {"prompt": "fixture", REQUEST_RNG_SEED_KEY: 123}
    first = policy.infer(observation)["actions"]
    policy.infer({**observation, REQUEST_RNG_SEED_KEY: 456})
    repeated = policy.infer(observation)["actions"]
    restarted = adapter.RequestRngPolicy(LoadedPolicyFixture()).infer(observation)["actions"]
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, restarted)
    assert seeds == [(123, 10, 32), (456, 10, 32), (123, 10, 32), (123, 10, 32)]
    assert all(REQUEST_RNG_SEED_KEY not in obs for obs in upstream.observations)
    assert observation[REQUEST_RNG_SEED_KEY] == 123
    assert policy.metadata == {"fixture": True, "request_rng_required": True}
    assert "request_rng_required" not in upstream.metadata


@pytest.mark.parametrize("seed", [None, True, -1, 2**32, 1.5, "123"])
def test_invalid_request_seed_never_invokes_model(seed: object) -> None:
    upstream = LoadedPolicyFixture()
    policy = adapter.RequestRngPolicy(upstream)
    with pytest.raises(ValueError, match="RNG seed"):
        policy.infer({REQUEST_RNG_SEED_KEY: seed})
    assert not upstream.observations


@pytest.mark.parametrize("output", [np.zeros((9, 7)), np.full((10, 7), np.nan)])
def test_server_boundary_reuses_canonical_action_validation(
    monkeypatch: pytest.MonkeyPatch, output: np.ndarray,
) -> None:
    monkeypatch.setattr(adapter, "request_noise", lambda *_: np.zeros((10, 32)))
    upstream = LoadedPolicyFixture()
    monkeypatch.setattr(upstream, "infer", lambda *_, **__: {"actions": output})
    with pytest.raises(ValueError):
        adapter.RequestRngPolicy(upstream).infer({REQUEST_RNG_SEED_KEY: 0})
