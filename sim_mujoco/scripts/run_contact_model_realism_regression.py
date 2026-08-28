#!/usr/bin/env python3
"""Run the fixed A/B contact-model realism and manipulation regression."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
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

from sim_mujoco.data_collection.conversions import (  # noqa: E402
    policy_state_from_mujoco,
)
from sim_mujoco.data_collection.ik_solver import solve_site_pose  # noqa: E402
from sim_mujoco.data_collection.oracle_controller import (  # noqa: E402
    PlaceOracleConfig,
    PlaceOracleStage,
    PlaceRedPepperOracleController,
)
from sim_mujoco.environment import MuJoCoEnvironment  # noqa: E402
from sim_mujoco.gripper_slip_diagnostics import (  # noqa: E402
    CommandContext,
    PhysicsTraceRecorder,
)
from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import (  # noqa: E402
    BASE_MODEL_PATH,
    GRASP_OFFSET_BY_TASK,
    _apply_overrides,
    _capture_initial_state,
    _interpolate_arm_targets,
    _oracle_action_manifest,
    _restore_initial_state,
    _run_trial,
    _sha256,
    _step,
)
from sim_mujoco.task_scenes import resolve_task  # noqa: E402


ALLOWED_OUTPUT_ROOT = Path("/work/nvme/bfmk/mw89")
TASKS = ("red_block", "blue_block", "red_pepper")
PLACE_TASK = "place_red_pepper_in_ring"
PROTOCOLS = ("suspended_grasp", "pushing", "placing_release", "tabletop_sliding")
SEEDS = (50000, 50001, 50002)
ACTION_DT_S = 0.1
PUSH_START_OFFSET_M = -0.065
PUSH_END_OFFSET_M = 0.065
PUSH_SETTLE_S = 1.0
SLIDE_INITIAL_VELOCITY_MPS = 0.25
SLIDE_DURATION_S = 2.0


def contact_conditions() -> list[dict[str, Any]]:
    """Return the only two allowed interventions for this experiment."""
    return [
        {
            "name": "production_pyramidal_impratio1",
            "condition": "A",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "cone": "pyramidal",
            "impratio": 1.0,
        },
        {
            "name": "candidate_elliptic_impratio10",
            "condition": "B",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "cone": "elliptic",
            "impratio": 10.0,
        },
    ]


def experiment_matrix(seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        tasks = (PLACE_TASK,) if protocol == "placing_release" else TASKS
        for task in tasks:
            for seed in seeds:
                for setting in contact_conditions():
                    rows.append(
                        {
                            "protocol": protocol,
                            "task": task,
                            "seed": seed,
                            "condition": setting["condition"],
                            "setting": setting["name"],
                        }
                    )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prerequisite-root", type=Path, required=True)
    parser.add_argument(
        "--seed", action="append", dest="seeds", type=int, required=True
    )
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    return parser


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    output = args.output_root.expanduser().resolve()
    prerequisite = args.prerequisite_root.expanduser().resolve()
    for label, path in (("output", output), ("prerequisite", prerequisite)):
        if path == ALLOWED_OUTPUT_ROOT or ALLOWED_OUTPUT_ROOT not in path.parents:
            raise ValueError(
                f"{label} must be a child of {ALLOWED_OUTPUT_ROOT}: {path}"
            )
    if output.exists():
        raise FileExistsError(f"Refusing existing output root: {output}")
    if args.hold_seconds != 5.0:
        raise ValueError("This fixed experiment requires --hold-seconds 5")
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"This fixed experiment requires seeds in order {SEEDS}")
    results_path = prerequisite / "results.json"
    validation_path = prerequisite / "analysis" / "validation.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if results.get("status") != "complete" or results.get("suite") != "contact":
        raise ValueError(
            f"Prerequisite is not a complete contact suite: {results_path}"
        )
    if validation.get("passed") is not True:
        raise ValueError(
            f"Prerequisite contact validation did not pass: {validation_path}"
        )
    return output, prerequisite


def _state_summary(state: dict[str, Any], source: str) -> dict[str, Any]:
    return {**state, "state": None, "source_condition": source}


def _action_manifest(stages: list[tuple[str, list[np.ndarray]]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for stage, values in stages:
        actions = np.asarray(values, dtype=np.float64).reshape(-1, 7)
        digest.update(stage.encode("utf-8"))
        digest.update(np.asarray(actions.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(actions).tobytes())
        rows.append(
            {
                "stage": stage,
                "action_count": int(actions.shape[0]),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(actions).tobytes()
                ).hexdigest(),
            }
        )
    return {
        "action_dtype": "float64",
        "action_width": 7,
        "total_action_count": sum(row["action_count"] for row in rows),
        "sha256": digest.hexdigest(),
        "stages": rows,
    }


def _common_trial(
    environment: MuJoCoEnvironment,
    *,
    model_path: Path = BASE_MODEL_PATH,
    protocol: str,
    task: str,
    seed: int,
    setting: dict[str, Any],
    overrides: dict[str, Any],
    state: dict[str, Any],
    action_manifest: dict[str, Any],
    action_source: str,
) -> dict[str, Any]:
    runtime = environment.task_runtime
    assert runtime is not None
    target_id = mujoco.mj_name2id(
        environment.context.model, mujoco.mjtObj.mjOBJ_BODY, runtime.target_body
    )
    return {
        "protocol": protocol,
        "task": task,
        "seed": seed,
        "setting": setting,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "production_model_file_modified": False,
        "overrides": overrides,
        "target_body": runtime.target_body,
        "target_mass_kg": float(environment.context.model.body_mass[target_id]),
        "initial_target_z_m": runtime.initial_target_z,
        "initial_target_position_m": np.asarray(
            environment.context.data.xpos[target_id], dtype=np.float64
        ).tolist(),
        "paired_initial_state": _state_summary(
            state, "A" if setting["condition"] == "A" else "A-restored"
        ),
        "action_plan_source_condition": action_source,
        "action_manifest": action_manifest,
    }


def _run_actions(
    environment: MuJoCoEnvironment,
    recorder: PhysicsTraceRecorder,
    stages: list[tuple[str, list[np.ndarray]]],
) -> None:
    action_step = 0
    for stage, actions in stages:
        for action in actions:
            environment.apply_action(action)
            command = CommandContext(
                source="scripted_realism_regression",
                stage=stage,
                action_step=action_step,
                gripper_returned_raw=float(action[6]),
                gripper_clamped_raw=float(action[6]),
                gripper_ctrl=float(environment.context.data.ctrl[6]),
            )
            _step(environment, recorder, command, ACTION_DT_S)
            action_step += 1


def _write_result(
    output_dir: Path,
    recorder: PhysicsTraceRecorder,
    result: dict[str, Any],
) -> dict[str, Any]:
    artifacts = recorder.write(output_dir)
    result = {
        "status": "complete",
        **result,
        "sample_count": len(recorder.rows),
        "event_names": [event.event for event in recorder.events],
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _initialize_environment(
    stack: ExitStack,
    *,
    task: str,
    seed: int,
    setting: dict[str, Any],
    state_reference: dict[str, Any] | None,
    model_path: Path = BASE_MODEL_PATH,
) -> tuple[MuJoCoEnvironment, dict[str, Any], dict[str, Any]]:
    environment = stack.enter_context(
        MuJoCoEnvironment(task=task, settle_steps=500, model_path=model_path)
    )
    _, task_spec = resolve_task(task)
    overrides = _apply_overrides(
        environment, setting, target_body=str(task_spec["target_body"])
    )
    environment.reset(seed=seed)
    if state_reference is not None:
        _restore_initial_state(environment, state_reference)
    state = _capture_initial_state(environment)
    if (
        state_reference is not None
        and state["state_sha256"] != state_reference["state_sha256"]
    ):
        raise RuntimeError("Restored paired state differs from condition A")
    return environment, overrides, state


def _push_plan(
    environment: MuJoCoEnvironment, task: str
) -> list[tuple[str, list[np.ndarray]]]:
    model = environment.context.model
    data = environment.context.data
    runtime = environment.task_runtime
    assert runtime is not None
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.target_body)
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point")
    object_position = np.asarray(data.xpos[target_id], dtype=np.float64).copy()
    tcp_rotation = (
        np.asarray(data.site_xmat[tcp_id], dtype=np.float64).reshape(3, 3).copy()
    )
    initial = policy_state_from_mujoco(model, data).astype(np.float64)
    push_z = float(object_position[2] + GRASP_OFFSET_BY_TASK.get(task, -0.011))
    start_position = np.asarray(
        [object_position[0] + PUSH_START_OFFSET_M, object_position[1], push_z]
    )
    end_position = np.asarray(
        [object_position[0] + PUSH_END_OFFSET_M, object_position[1], push_z]
    )
    start = solve_site_pose(
        model,
        data,
        site_name="tool_center_point",
        target_position=start_position,
        target_rotation=tcp_rotation,
        seed_joint_qpos=initial[:6],
    )
    push_solutions = []
    seed_qpos = start.joint_qpos
    for position in np.linspace(start_position, end_position, 27)[1:]:
        solution = solve_site_pose(
            model,
            data,
            site_name="tool_center_point",
            target_position=position,
            target_rotation=tcp_rotation,
            seed_joint_qpos=seed_qpos,
        )
        push_solutions.append(solution)
        seed_qpos = solution.joint_qpos
    failed_waypoints = [
        index
        for index, solution in enumerate(push_solutions, start=1)
        if not solution.success
    ]
    if not start.success or failed_waypoints:
        raise RuntimeError(
            f"Pushing IK failed: start={start.success}, waypoints={failed_waypoints}"
        )
    closed = 50.0
    close_count = max(1, int(np.ceil(abs(initial[6] - closed) / 25.0)))
    close_values = np.linspace(initial[6], closed, close_count + 1)[1:]
    approach = _interpolate_arm_targets(
        initial[:6], start.joint_qpos, max_step_rad=0.025
    )
    push = [solution.joint_qpos for solution in push_solutions]
    end_qpos = push[-1]
    settle_count = int(round(PUSH_SETTLE_S / ACTION_DT_S))
    return [
        ("CLOSE_IN_AIR", [np.concatenate([initial[:6], [x]]) for x in close_values]),
        ("MOVE_TO_PUSH_START", [np.concatenate([x, [closed]]) for x in approach]),
        ("PUSH", [np.concatenate([x, [closed]]) for x in push]),
        (
            "PUSH_SETTLE",
            [np.concatenate([end_qpos, [closed]]) for _ in range(settle_count)],
        ),
    ]


def _run_pushing(
    output_dir: Path,
    *,
    task: str,
    seed: int,
    setting: dict[str, Any],
    state_reference: dict[str, Any] | None,
    plan_reference: list[tuple[str, list[np.ndarray]]] | None,
    model_path: Path = BASE_MODEL_PATH,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, list[np.ndarray]]]]:
    with ExitStack() as stack:
        environment, overrides, state = _initialize_environment(
            stack,
            task=task,
            seed=seed,
            setting=setting,
            state_reference=state_reference,
            model_path=model_path,
        )
        stages = (
            _push_plan(environment, task)
            if plan_reference is None
            else deepcopy(plan_reference)
        )
        manifest = _action_manifest(stages)
        trial = _common_trial(
            environment,
            model_path=model_path,
            protocol="pushing",
            task=task,
            seed=seed,
            setting=setting,
            overrides=overrides,
            state=state,
            action_manifest=manifest,
            action_source="A" if plan_reference is None else "A-reused",
        )
        runtime = environment.task_runtime
        assert runtime is not None
        recorder = PhysicsTraceRecorder(
            model=environment.context.model,
            data=environment.context.data,
            target_body=runtime.target_body,
            camera_config=environment.context.config,
            initial_target_z_m=runtime.initial_target_z,
            trial=trial,
        )
        _run_actions(environment, recorder, stages)
        result = _write_result(
            output_dir,
            recorder,
            {"protocol": "pushing", "task": task, "seed": seed, "setting": setting},
        )
        return result, state, deepcopy(stages)


def _set_planar_velocity(environment: MuJoCoEnvironment) -> None:
    model = environment.context.model
    data = environment.context.data
    runtime = environment.task_runtime
    assert runtime is not None
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.target_body)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("Sliding target must have a free joint")
    dof = int(model.jnt_dofadr[joint_id])
    data.qvel[dof : dof + 6] = 0.0
    data.qvel[dof] = SLIDE_INITIAL_VELOCITY_MPS
    mujoco.mj_forward(model, data)


def _run_sliding(
    output_dir: Path,
    *,
    task: str,
    seed: int,
    setting: dict[str, Any],
    state_reference: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], None]:
    with ExitStack() as stack:
        environment, overrides, _ = _initialize_environment(
            stack, task=task, seed=seed, setting=setting, state_reference=None
        )
        if state_reference is None:
            _set_planar_velocity(environment)
        else:
            _restore_initial_state(environment, state_reference)
        state = _capture_initial_state(environment)
        if (
            state_reference is not None
            and state["state_sha256"] != state_reference["state_sha256"]
        ):
            raise RuntimeError("Restored paired sliding state differs from condition A")
        manifest = _action_manifest([])
        trial = _common_trial(
            environment,
            protocol="tabletop_sliding",
            task=task,
            seed=seed,
            setting=setting,
            overrides=overrides,
            state=state,
            action_manifest=manifest,
            action_source="A-fixed-velocity" if state_reference is None else "A-reused",
        )
        trial["initial_planar_velocity_mps"] = [SLIDE_INITIAL_VELOCITY_MPS, 0.0]
        runtime = environment.task_runtime
        assert runtime is not None
        recorder = PhysicsTraceRecorder(
            model=environment.context.model,
            data=environment.context.data,
            target_body=runtime.target_body,
            camera_config=environment.context.config,
            initial_target_z_m=runtime.initial_target_z,
            trial=trial,
        )
        command = CommandContext(
            source="scripted_realism_regression", stage="FREE_SLIDE", action_step=0
        )
        _step(environment, recorder, command, SLIDE_DURATION_S)
        result = _write_result(
            output_dir,
            recorder,
            {
                "protocol": "tabletop_sliding",
                "task": task,
                "seed": seed,
                "setting": setting,
            },
        )
        return result, state, None


def _reuse_place_plan(
    controller: PlaceRedPepperOracleController, reference: Any
) -> None:
    controller.stage = PlaceOracleStage.RESET
    controller.failure_reason = None
    controller.action_steps = 0
    controller.transitions = controller.transitions[:1]
    controller.plan = deepcopy(reference)
    controller._stage_actions = controller._build_stage_actions()
    controller._stage_action_index = 0
    controller.release_step = None
    controller.retreat_detected = False
    controller._verification_samples = []
    controller._stability_result = None


def _run_placing(
    output_dir: Path,
    *,
    seed: int,
    setting: dict[str, Any],
    state_reference: dict[str, Any] | None,
    plan_reference: Any | None,
    model_path: Path = BASE_MODEL_PATH,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    with ExitStack() as stack:
        environment, overrides, state = _initialize_environment(
            stack,
            task=PLACE_TASK,
            seed=seed,
            setting=setting,
            state_reference=state_reference,
            model_path=model_path,
        )
        config = PlaceOracleConfig(task=PLACE_TASK)
        controller = PlaceRedPepperOracleController(environment, config)
        if plan_reference is not None:
            _reuse_place_plan(controller, plan_reference)
        if controller.failure_reason is not None:
            raise RuntimeError(f"Place planning failed: {controller.failure_reason}")
        manifest = _oracle_action_manifest(controller)
        trial = _common_trial(
            environment,
            model_path=model_path,
            protocol="placing_release",
            task=PLACE_TASK,
            seed=seed,
            setting=setting,
            overrides=overrides,
            state=state,
            action_manifest=manifest,
            action_source="A" if plan_reference is None else "A-reused",
        )
        trial.update(
            {
                "oracle_config": asdict(config),
                "oracle_plan": controller.plan.to_json(),
            }
        )
        runtime = environment.task_runtime
        assert runtime is not None
        recorder = PhysicsTraceRecorder(
            model=environment.context.model,
            data=environment.context.data,
            target_body=runtime.target_body,
            camera_config=environment.context.config,
            initial_target_z_m=runtime.initial_target_z,
            trial=trial,
        )
        while not controller.terminal:
            action = controller.next_action()
            if action is None:
                break
            stage = controller.stage.value
            environment.apply_action(action)
            command = CommandContext(
                source="scripted_place_oracle",
                stage=stage,
                action_step=controller.action_steps,
                gripper_returned_raw=float(action[6]),
                gripper_clamped_raw=float(action[6]),
                gripper_ctrl=float(environment.context.data.ctrl[6]),
            )
            collision = _step(environment, recorder, command, config.action_dt_s)
            controller.notify_post_step(
                task_metrics=runtime.metrics(),
                collision=collision,
                simulation_finite=bool(
                    np.isfinite(environment.context.data.qpos).all()
                    and np.isfinite(environment.context.data.qvel).all()
                ),
            )
        result = _write_result(
            output_dir,
            recorder,
            {
                "protocol": "placing_release",
                "task": PLACE_TASK,
                "seed": seed,
                "setting": setting,
                "oracle_terminal_stage": controller.stage.value,
                "place_stability": controller.stability_metadata(),
                "oracle_transitions": controller.transition_log(),
            },
        )
        return result, state, deepcopy(controller.plan)


def main() -> None:
    args = _parser().parse_args()
    output, prerequisite = _validate_args(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The realism regression must run inside a Slurm allocation")
    output.mkdir(parents=True, exist_ok=False)
    matrix = experiment_matrix(args.seeds)
    manifest = {
        "schema_version": "xarm_contact_realism_regression_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "suite": "realism",
        "conditions": contact_conditions(),
        "protocols": list(PROTOCOLS),
        "tasks": list(TASKS),
        "place_task": PLACE_TASK,
        "seeds": args.seeds,
        "trial_count": len(matrix),
        "matrix": matrix,
        "prerequisite_root": str(prerequisite),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "repository": str(PROJECT_ROOT),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status_short": _git(["status", "--short"]),
        "mujoco_version": mujoco.__version__,
        "python": sys.version,
        "paired_design": {
            "reference_condition": "A",
            "state_pairing": "exact non-warm-start MuJoCo state SHA-256",
            "action_pairing": "condition A plan/actions reused byte-for-byte in B",
            "changes_only": ["cone", "impratio"],
            "forbidden_changes": [
                "kp",
                "friction",
                "geometry",
                "mass",
                "timestep",
                "solver",
                "solref",
                "solimp",
            ],
        },
        "protocol_parameters": {
            "suspended_grasp_hold_s": args.hold_seconds,
            "pushing_axis": "+x",
            "pushing_tcp_offsets_m": [PUSH_START_OFFSET_M, PUSH_END_OFFSET_M],
            "pushing_settle_s": PUSH_SETTLE_S,
            "sliding_initial_velocity_mps": [SLIDE_INITIAL_VELOCITY_MPS, 0.0],
            "sliding_duration_s": SLIDE_DURATION_S,
        },
        "primary_metrics": [
            "penetration_depth_and_duration",
            "normal_and_tangential_contact_force",
            "relative_grasp_slip",
            "release_latency_after_opening",
            "pushing_displacement",
            "sliding_displacement",
        ],
        "prior_penetration_guardrail_m": 0.00544,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    references: dict[tuple[str, str, int], tuple[dict[str, Any], Any]] = {}
    try:
        for protocol in PROTOCOLS:
            tasks = (PLACE_TASK,) if protocol == "placing_release" else TASKS
            for task in tasks:
                for seed in args.seeds:
                    pair_key = (protocol, task, seed)
                    for setting in contact_conditions():
                        reference = references.get(pair_key)
                        state_reference = None if reference is None else reference[0]
                        plan_reference = None if reference is None else reference[1]
                        trial_name = (
                            f"{protocol}_{task}_seed{seed}_{setting['condition']}"
                        )
                        trial_dir = output / "trials" / trial_name
                        trial_dir.mkdir(parents=True, exist_ok=False)
                        if protocol == "suspended_grasp":
                            result, state, plan = _run_trial(
                                output_dir=trial_dir,
                                task=task,
                                seed=seed,
                                hold_kind="suspended",
                                hold_seconds=args.hold_seconds,
                                setting=setting,
                                model_path=BASE_MODEL_PATH,
                                record_video=False,
                                initial_state_reference=state_reference,
                                oracle_plan_reference=plan_reference,
                                protocol=protocol,
                            )
                        elif protocol == "pushing":
                            result, state, plan = _run_pushing(
                                trial_dir,
                                task=task,
                                seed=seed,
                                setting=setting,
                                state_reference=state_reference,
                                plan_reference=plan_reference,
                            )
                        elif protocol == "placing_release":
                            result, state, plan = _run_placing(
                                trial_dir,
                                seed=seed,
                                setting=setting,
                                state_reference=state_reference,
                                plan_reference=plan_reference,
                            )
                        else:
                            result, state, plan = _run_sliding(
                                trial_dir,
                                task=task,
                                seed=seed,
                                setting=setting,
                                state_reference=state_reference,
                            )
                        if setting["condition"] == "A":
                            references[pair_key] = (state, plan)
                        results.append(result)
    except BaseException as exc:
        (output / "results.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "completed_trial_count": len(results),
                    "trials": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    complete = {
        "status": "complete",
        "suite": "realism",
        "trial_count": len(results),
        "trials": results,
    }
    (output / "results.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
