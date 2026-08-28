#!/usr/bin/env python3
"""Run a small two-stage sliding-friction search on the frozen split-pad model."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402

from sim_mujoco.scripts.analyze_friction_ablation import _hold_metrics  # noqa: E402
from sim_mujoco.scripts.run_contact_model_realism_regression import (  # noqa: E402
    _run_placing,
    _run_pushing,
)
from sim_mujoco.scripts.run_friction_ablation import (  # noqa: E402
    FROZEN_MODEL,
    FROZEN_MODEL_SHA256,
    PAD_FRICTION_A,
    validate_frozen_model,
)
from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import (  # noqa: E402
    _run_trial,
)


ALLOWED_ROOT = Path("/work/nvme/bfmk/mw89")
SEED = 50000
HOLD_SECONDS = 5.0
TASKS = ("red_block", "blue_block", "red_pepper")
CANDIDATE_MU = (2.0, 0.2, 0.35, 0.5, 0.65, 0.85, 1.15, 1.5)


def _label(mu: float) -> str:
    return f"mu_{mu:.2f}".replace(".", "p")


def _setting(mu: float) -> dict[str, Any]:
    mapping = {name: float(mu) for name in PAD_FRICTION_A}
    return {
        "name": _label(mu),
        "condition": _label(mu),
        "sliding_friction": float(mu),
        "force_multiplier": 1.0,
        "kp_multiplier": 1.0,
        "friction_multiplier": 1.0,
        "pad_sliding_friction_by_name": mapping,
        "allowed_changed_invariants": []
        if math.isclose(mu, 2.0)
        else ["geom_friction"],
        "cone": "pyramidal",
        "impratio": 1.0,
        "gripper_closing_rate_raw_per_s": 244.0,
        "gripper_opening_rate_raw_per_s": 220.0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def validate_search() -> dict[str, Any]:
    frozen = validate_frozen_model()
    model = mujoco.MjModel.from_xml_path(str(FROZEN_MODEL))
    pad_values = {}
    for name in PAD_FRICTION_A:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        pad_values[name] = model.geom_friction[geom_id].tolist()
    return {
        "passed": True,
        "frozen_model": str(FROZEN_MODEL),
        "frozen_model_sha256": FROZEN_MODEL_SHA256,
        "candidate_mu": list(CANDIDATE_MU),
        "baseline_pad_friction": pad_values,
        "intentional_compiled_change": "geom_friction only",
        "frozen_validation": frozen,
    }


def _safe_output(path: Path | None) -> Path:
    if path is None:
        raise ValueError("--output-root is required")
    output = path.expanduser().resolve()
    if output == ALLOWED_ROOT or ALLOWED_ROOT not in output.parents:
        raise ValueError(f"Output must be below {ALLOWED_ROOT}: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    return output


def _trial_dir(
    output: Path, phase: str, protocol: str, task: str, setting: dict[str, Any]
) -> Path:
    path = output / phase / f"{protocol}_{task}_seed{SEED}_{setting['name']}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _screen_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        1.0 if metrics["drop"] or not metrics["retained"] else 0.0,
        float(metrics["maximum_downward_slip_m"]),
        float(metrics["maximum_penetration_m"] or 0.0),
    )


def main() -> None:
    args = _parser().parse_args()
    validation = validate_search()
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The friction search must run inside Slurm")
    output = _safe_output(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    settings = [_setting(mu) for mu in CANDIDATE_MU]
    manifest = {
        "schema_version": "xarm_split_pad_friction_search_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "seed": SEED,
        "hold_seconds": HOLD_SECONDS,
        "settings": settings,
        "maximum_trial_count": 29,
        "design": {
            "screen": "8 red-block suspended holds",
            "validation": "baseline plus best two screen candidates; 3 grasps, 3 pushes, 1 release each",
            "selection_uses_policy": False,
        },
        "model_validation": validation,
        "production_mjcf_modified": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    screen_results: list[dict[str, Any]] = []
    screen_metrics: list[dict[str, Any]] = []
    state_reference = None
    plan_reference = None
    try:
        for index, setting in enumerate(settings):
            result, state, plan = _run_trial(
                output_dir=_trial_dir(
                    output, "screen", "suspended_grasp", "red_block", setting
                ),
                task="red_block",
                seed=SEED,
                hold_kind="suspended",
                hold_seconds=HOLD_SECONDS,
                setting=setting,
                model_path=FROZEN_MODEL,
                record_video=False,
                initial_state_reference=state_reference,
                oracle_plan_reference=plan_reference,
                protocol="suspended_grasp",
            )
            if index == 0:
                state_reference, plan_reference = state, plan
            result["search_phase"] = "screen"
            screen_results.append(result)
            metrics = _hold_metrics(result)
            metrics["sliding_friction"] = setting["sliding_friction"]
            metrics["setting"] = setting["name"]
            screen_metrics.append(metrics)

        ranked = sorted(screen_metrics, key=_screen_score)
        top_two = [row["setting"] for row in ranked[:2]]
        validation_names = [_label(2.0), *top_two]
        validation_names = list(dict.fromkeys(validation_names))
        validation_settings = [
            next(setting for setting in settings if setting["name"] == name)
            for name in validation_names
        ]
        selection = {
            "screen_ranking": [
                {
                    "rank": index,
                    "setting": row["setting"],
                    "sliding_friction": row["sliding_friction"],
                    "retained": row["retained"],
                    "drop": row["drop"],
                    "maximum_downward_slip_m": row["maximum_downward_slip_m"],
                    "maximum_penetration_m": row["maximum_penetration_m"],
                }
                for index, row in enumerate(ranked, start=1)
            ],
            "top_two": top_two,
            "validation_settings": validation_names,
        }
        (output / "screen_selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        validation_results: list[dict[str, Any]] = []
        for task in TASKS:
            paired_state = None
            paired_plan = None
            for index, setting in enumerate(validation_settings):
                result, state, plan = _run_trial(
                    output_dir=_trial_dir(
                        output,
                        "validation",
                        "suspended_grasp",
                        task,
                        setting,
                    ),
                    task=task,
                    seed=SEED,
                    hold_kind="suspended",
                    hold_seconds=HOLD_SECONDS,
                    setting=setting,
                    model_path=FROZEN_MODEL,
                    record_video=False,
                    initial_state_reference=paired_state,
                    oracle_plan_reference=paired_plan,
                    protocol="suspended_grasp",
                )
                if index == 0:
                    paired_state, paired_plan = state, plan
                result["search_phase"] = "validation"
                validation_results.append(result)

        for task in TASKS:
            paired_state = None
            paired_plan = None
            for index, setting in enumerate(validation_settings):
                result, state, plan = _run_pushing(
                    _trial_dir(output, "validation", "pushing", task, setting),
                    task=task,
                    seed=SEED,
                    setting=setting,
                    state_reference=paired_state,
                    plan_reference=paired_plan,
                    model_path=FROZEN_MODEL,
                )
                if index == 0:
                    paired_state, paired_plan = state, plan
                result["search_phase"] = "validation"
                validation_results.append(result)

        paired_state = None
        paired_plan = None
        for index, setting in enumerate(validation_settings):
            result, state, plan = _run_placing(
                _trial_dir(
                    output,
                    "validation",
                    "placing_release",
                    "place_red_pepper_in_ring",
                    setting,
                ),
                seed=SEED,
                setting=setting,
                state_reference=paired_state,
                plan_reference=paired_plan,
                model_path=FROZEN_MODEL,
            )
            if index == 0:
                paired_state, paired_plan = state, plan
            result["search_phase"] = "validation"
            validation_results.append(result)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error": repr(exc),
            "screen_results": screen_results,
            "screen_metrics": screen_metrics,
            "validation_results": locals().get("validation_results", []),
        }
        (output / "results.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    results = {
        "status": "complete",
        "screen_trial_count": len(screen_results),
        "validation_trial_count": len(validation_results),
        "screen_results": screen_results,
        "screen_metrics": screen_metrics,
        "validation_results": validation_results,
        "screen_selection": selection,
    }
    (output / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
