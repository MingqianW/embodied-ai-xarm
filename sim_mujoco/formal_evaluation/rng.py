"""Stable request-scoped stochastic-policy RNG derivation."""

from __future__ import annotations

import hashlib


def policy_rng_seed(*, protocol_salt: str, task_id: str, evaluation_seed: int, policy_step: int) -> int:
    """Return a stable uint32 seed independent of process and episode order."""

    if evaluation_seed < 0 or policy_step < 0:
        raise ValueError("Evaluation seed and policy step must be non-negative")
    payload = "\0".join(
        ("xarm-formal-policy-rng-v1", protocol_salt, task_id, str(evaluation_seed), str(policy_step))
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")
