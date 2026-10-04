"""No-render physical Place reset checks; run in a compute allocation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from evaluation.sim.config import load_protocol
from evaluation.sim.success import validate_initial_place_grasp
from simulation.configuration import load_simulation_config
from simulation.physics.collision import collision_diagnostics, target_gripper_contact_count
from simulation.robot.gripper import actuator_ctrl_from_raw_hardware
from simulation.runtime import initialize_scene
from simulation.scene import configure_task_scene


def _reset(seed: int):
    protocol = load_protocol()
    model = mujoco.MjModel.from_xml_path(str(protocol.robot_xml_path))
    data = mujoco.MjData(model)
    initialize_scene(model, data, settle_steps=0)
    runtime, initial = configure_task_scene(
        model, data, task="place_red_pepper_in_ring", seed=seed,
        object_xy_range=protocol.object_xy_range_m,
        object_yaw_range_deg=protocol.object_yaw_range_deg,
        joint_noise=protocol.joint_noise_rad, layout_profile=protocol.layout_profile,
        config_path=protocol.task_scene_config_path,
    )
    return protocol, model, data, runtime, initial


def _verify_grasp(seed: int):
    protocol, model, data, runtime, initial = _reset(seed)
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_pepper")
    joint = int(model.body_jntadr[target])
    assert int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
    assert runtime.active_target_body == "red_pepper"
    assert "held_red_pepper" not in initial["active_bodies"]
    assert initial["place_initialization"] == {
        "mode": "physical_free_body", "settle_steps": 500,
        "initial_gripper_raw": 450.0, "pose_assignments": 1, "attachment_used": False,
    }
    assert not any(
        int(model.eq_type[i]) in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD))
        and target in (int(model.eq_obj1id[i]), int(model.eq_obj2id[i]))
        for i in range(model.neq)
    )
    result = validate_initial_place_grasp(runtime=runtime, initial_conditions=initial, protocol=protocol)
    assert result["validated"], result
    assert result["maximum_relative_drift_m"] <= protocol.placement_initial_max_relative_drift_m
    assert all(c["gripper_contact_count"] >= protocol.placement_initial_min_gripper_contacts for c in result["checks"])
    assert all(c["height_above_table_m"] >= protocol.placement_initial_min_height_above_table_m for c in result["checks"])
    assert all(not c["forbidden_collision"] and not c["table_contact"] for c in result["checks"])
    evidence_root = os.environ.get("XARM_PLACE_RESET_EVIDENCE")
    if evidence_root:
        evidence = Path(evidence_root)
        evidence.mkdir(parents=True, exist_ok=True)
        with (evidence / f"seed_{seed}.json").open("x") as stream:
            json.dump({"seed": seed, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "initial_conditions": initial, "validation": result,
                       "collision_after_validation": collision_diagnostics(model, data)}, stream, indent=2)
    return protocol, model, data, runtime, initial


@pytest.mark.parametrize("seed", [50018, 50019])
def test_place_smoke_seeds_hold_without_initialization_assistance(seed):
    protocol, model, data, runtime, _ = _verify_grasp(seed)
    # Commands and observations retain the normal raw/action contract.
    observation = {"observation/state": np.array([0., 0., 0., 0., 0., 0., 620.])}
    runtime.adjust_observation(observation)
    assert observation["observation/state"][6] == 620.
    assert runtime.physical_gripper_raw_target(440.) == 440.
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    assert not runtime.release_if_requested(600.)
    assert runtime.release_if_requested(700.)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert runtime.active_target_body == "red_pepper"
    # Opening must physically release the same free object: no hidden weld,
    # kinematic fixture, repeated pose write, or raw-command lock can retain it.
    before = runtime._target_pose().copy()
    data.ctrl[6] = actuator_ctrl_from_raw_hardware(845., load_simulation_config())
    for _ in range(round(1. / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert runtime._target_pose()[2] < before[2] - .04
    assert target_gripper_contact_count(collision_diagnostics(model, data), "red_pepper") == 0


@pytest.mark.parametrize("seed", range(50020, 50038))
def test_remaining_formal_place_seeds_hold_without_assistance(seed):
    _verify_grasp(seed)
