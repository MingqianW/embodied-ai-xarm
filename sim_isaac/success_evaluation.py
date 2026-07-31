from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiftEvaluation:
    success: bool
    score: float
    lift_height_m: float
    target_height_m: float

    def to_json(self) -> dict[str, float | bool]:
        return {
            "success": self.success,
            "score": self.score,
            "lift_height_m": self.lift_height_m,
            "target_height_m": self.target_height_m,
        }


def evaluate_lift(
    object_position_m: np.ndarray,
    initial_position_m: np.ndarray,
    *,
    target_lift_m: float = 0.08,
    partial_credit_lift_m: float = 0.04,
) -> LiftEvaluation:
    if target_lift_m <= 0 or not 0 < partial_credit_lift_m <= target_lift_m:
        raise ValueError("Lift thresholds must satisfy 0 < partial <= target")
    position = np.asarray(object_position_m, dtype=np.float64)
    initial = np.asarray(initial_position_m, dtype=np.float64)
    if position.shape != (3,) or initial.shape != (3,):
        raise ValueError("Object positions must have shape (3,)")
    if not np.isfinite(position).all() or not np.isfinite(initial).all():
        raise ValueError("Object positions must be finite")
    lift = float(position[2] - initial[2])
    if lift >= target_lift_m:
        score = 1.0
    elif lift <= 0:
        score = 0.0
    elif lift < partial_credit_lift_m:
        score = 0.5 * lift / partial_credit_lift_m
    else:
        score = 0.5 + 0.5 * (
            (lift - partial_credit_lift_m)
            / (target_lift_m - partial_credit_lift_m)
        )
    return LiftEvaluation(
        success=lift >= target_lift_m,
        score=float(np.clip(score, 0.0, 1.0)),
        lift_height_m=lift,
        target_height_m=float(target_lift_m),
    )
