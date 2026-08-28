#!/usr/bin/env python3
"""Run the fixed legacy split-pad fingertip-friction A/B diagnostic."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_mujoco.scripts.run_contact_model_realism_regression import (  # noqa: E402
    _run_placing,
    _run_pushing,
)
from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import (  # noqa: E402
    _model_invariant_hashes,
    _run_trial,
    _sha256,
)


ALLOWED_OUTPUT_ROOT = Path("/work/nvme/bfmk/mw89")
FROZEN_MODEL = Path(
    "/work/nvme/bfmk/mw89/runs/embodied-ai-xarm/gripper_slip/"
    "split_pad_geometry_20260814_v2/models/diagnostic_split_pad.xml"
)
FROZEN_MODEL_SHA256 = "6dc05639f9dee6709dd1c6a39e95785da85b410ac445ec1d13610381381e52fa"
TASKS = ("red_block", "blue_block", "red_pepper")
SEED = 50000
HOLD_SECONDS = 5.0
PAD_FRICTION_A = {
    "left_fingertip_pad": 2.0,
    "left_fingertip_pad_upper": 2.0,
    "right_fingertip_pad": 2.0,
    "right_fingertip_pad_upper": 2.0,
}
PAD_FRICTION_B = {
    "left_fingertip_pad": 0.7,
    "left_fingertip_pad_upper": 0.6,
    "right_fingertip_pad": 0.7,
    "right_fingertip_pad_upper": 0.6,
}


def conditions() -> tuple[dict[str, Any], dict[str, Any]]:
    common: dict[str, Any] = {
        "force_multiplier": 1.0,
        "kp_multiplier": 1.0,
        "friction_multiplier": 1.0,
        "cone": "pyramidal",
        "impratio": 1.0,
        "gripper_closing_rate_raw_per_s": 244.0,
        "gripper_opening_rate_raw_per_s": 220.0,
    }
    return (
        {
            **common,
            "name": "baseline_friction",
            "condition": "A",
            "allowed_changed_invariants": [],
        },
        {
            **common,
            "name": "menagerie_like_friction",
            "condition": "B",
            "pad_sliding_friction_by_name": PAD_FRICTION_B,
            "allowed_changed_invariants": ["geom_friction"],
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _pad_ids(model: mujoco.MjModel) -> dict[str, int]:
    result = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in PAD_FRICTION_A
    }
    missing = sorted(name for name, geom_id in result.items() if geom_id < 0)
    if missing:
        raise RuntimeError(f"Frozen split-pad model is missing pads: {missing}")
    return result


def validate_frozen_model() -> dict[str, Any]:
    if not FROZEN_MODEL.is_file():
        raise FileNotFoundError(FROZEN_MODEL)
    actual_sha = _sha256(FROZEN_MODEL)
    if actual_sha != FROZEN_MODEL_SHA256:
        raise RuntimeError(
            f"Frozen model SHA mismatch: {actual_sha} != {FROZEN_MODEL_SHA256}"
        )
    baseline = mujoco.MjModel.from_xml_path(str(FROZEN_MODEL))
    candidate = mujoco.MjModel.from_xml_path(str(FROZEN_MODEL))
    if int(baseline.nu) != 7:
        raise RuntimeError(f"Expected 7 actuators, found {baseline.nu}")
    left_slide = mujoco.mj_name2id(
        baseline, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_slide"
    )
    menagerie_driver = mujoco.mj_name2id(
        baseline, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint"
    )
    if left_slide < 0 or menagerie_driver >= 0:
        raise RuntimeError("Model is not the frozen simplified slide gripper")
    baseline_ids = _pad_ids(baseline)
    candidate_ids = _pad_ids(candidate)
    effective_a = {
        name: baseline.geom_friction[geom_id].tolist()
        for name, geom_id in baseline_ids.items()
    }
    for name, expected in PAD_FRICTION_A.items():
        if not np.allclose(effective_a[name], [expected, 0.02, 0.002], atol=1e-12):
            raise RuntimeError(
                f"Unexpected baseline pad friction: {name}={effective_a[name]}"
            )
    before = _model_invariant_hashes(candidate)
    for name, sliding in PAD_FRICTION_B.items():
        candidate.geom_friction[candidate_ids[name], 0] = sliding
    after = _model_invariant_hashes(candidate)
    changed = sorted(key for key in before if before[key] != after[key])
    if changed != ["geom_friction"]:
        raise RuntimeError(f"Runtime isolation failed: changed={changed}")
    effective_b = {
        name: candidate.geom_friction[geom_id].tolist()
        for name, geom_id in candidate_ids.items()
    }
    return {
        "passed": True,
        "model_path": str(FROZEN_MODEL),
        "model_sha256": actual_sha,
        "actuator_count": int(baseline.nu),
        "timestep_s": float(baseline.opt.timestep),
        "cone": int(baseline.opt.cone),
        "impratio": float(baseline.opt.impratio),
        "condition_a": effective_a,
        "condition_b": effective_b,
        "changed_compiled_invariants": changed,
    }


def _validate_output(path: Path | None) -> Path:
    if path is None:
        raise ValueError("--output-root is required unless --validate-only is used")
    output = path.expanduser().resolve()
    if output == ALLOWED_OUTPUT_ROOT or ALLOWED_OUTPUT_ROOT not in output.parents:
        raise ValueError(f"Output must be below {ALLOWED_OUTPUT_ROOT}: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    return output


def _trial_dir(output: Path, protocol: str, task: str, condition: str) -> Path:
    path = output / "trials" / f"{protocol}_{task}_seed{SEED}_{condition}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def main() -> None:
    args = _parser().parse_args()
    validation = validate_frozen_model()
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The friction A/B must run inside a Slurm allocation")
    output = _validate_output(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "model_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    setting_a, setting_b = conditions()
    manifest = {
        "schema_version": "xarm_split_pad_friction_ablation_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "repository": str(PROJECT_ROOT),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status_short": _git(["status", "--short"]),
        "python": sys.version,
        "mujoco_version": mujoco.__version__,
        "model_validation": validation,
        "conditions": [setting_a, setting_b],
        "tasks": list(TASKS),
        "seed": SEED,
        "hold_seconds": HOLD_SECONDS,
        "matrix": {
            "suspended_grasp": 6,
            "pushing": 6,
            "placing_release": 2,
            "total_trials": 14,
        },
        "intentional_physics_changes": ["four fingertip geom sliding-friction values"],
        "unchanged_friction_components": {"torsional": 0.02, "rolling": 0.002},
        "production_mjcf_modified": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    try:
        for task in TASKS:
            state_a = None
            plan_a = None
            for setting in (setting_a, setting_b):
                result, state, plan = _run_trial(
                    output_dir=_trial_dir(
                        output, "suspended_grasp", task, setting["condition"]
                    ),
                    task=task,
                    seed=SEED,
                    hold_kind="suspended",
                    hold_seconds=HOLD_SECONDS,
                    setting=setting,
                    model_path=FROZEN_MODEL,
                    record_video=False,
                    initial_state_reference=state_a,
                    oracle_plan_reference=plan_a,
                    protocol="suspended_grasp",
                )
                if setting["condition"] == "A":
                    state_a, plan_a = state, plan
                results.append(result)

        for task in TASKS:
            state_a = None
            plan_a = None
            for setting in (setting_a, setting_b):
                result, state, plan = _run_pushing(
                    _trial_dir(output, "pushing", task, setting["condition"]),
                    task=task,
                    seed=SEED,
                    setting=setting,
                    state_reference=state_a,
                    plan_reference=plan_a,
                    model_path=FROZEN_MODEL,
                )
                if setting["condition"] == "A":
                    state_a, plan_a = state, plan
                results.append(result)

        state_a = None
        plan_a = None
        for setting in (setting_a, setting_b):
            result, state, plan = _run_placing(
                _trial_dir(
                    output,
                    "placing_release",
                    "place_red_pepper_in_ring",
                    setting["condition"],
                ),
                seed=SEED,
                setting=setting,
                state_reference=state_a,
                plan_reference=plan_a,
                model_path=FROZEN_MODEL,
            )
            if setting["condition"] == "A":
                state_a, plan_a = state, plan
            results.append(result)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error": repr(exc),
            "completed_trial_count": len(results),
            "trials": results,
        }
        (output / "results.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    complete = {"status": "complete", "trial_count": len(results), "trials": results}
    (output / "results.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
