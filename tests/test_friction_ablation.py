from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim_mujoco.gripper_mapping import (
    gripper_state_to_raw,
    raw_gripper_to_sim_slide,
    sim_slide_to_raw_gripper,
)
from sim_mujoco.scripts.run_friction_ablation import (
    FROZEN_MODEL,
    PAD_FRICTION_A,
    PAD_FRICTION_B,
    conditions,
    validate_frozen_model,
)
from sim_mujoco.scripts.prepare_friction_policy_variant import validate_variant
from sim_mujoco.scripts.run_friction_search import (
    CANDIDATE_MU,
    _setting,
    validate_search,
)


CONFIG_PATH = Path(
    "sim_mujoco/config/diagnostics/legacy_split_pad_camera_calibration.yaml"
)
requires_frozen_split_pad_model = pytest.mark.skipif(
    not FROZEN_MODEL.is_file(),
    reason="legacy frozen split-pad artifact is available only on the Delta cluster",
)


@requires_frozen_split_pad_model
def test_frozen_model_runtime_diff_is_only_pad_friction() -> None:
    validation = validate_frozen_model()
    assert validation["passed"] is True
    assert validation["changed_compiled_invariants"] == ["geom_friction"]
    assert validation["condition_a"] == {
        name: [value, 0.02, 0.002] for name, value in PAD_FRICTION_A.items()
    }
    assert validation["condition_b"] == {
        name: [value, 0.02, 0.002] for name, value in PAD_FRICTION_B.items()
    }


def test_fixed_ab_conditions_preserve_nonfriction_physics() -> None:
    condition_a, condition_b = conditions()
    for key in (
        "force_multiplier",
        "kp_multiplier",
        "cone",
        "impratio",
        "gripper_closing_rate_raw_per_s",
        "gripper_opening_rate_raw_per_s",
    ):
        assert condition_a[key] == condition_b[key]
    assert condition_a["allowed_changed_invariants"] == []
    assert condition_b["allowed_changed_invariants"] == ["geom_friction"]


@requires_frozen_split_pad_model
def test_legacy_slide_mapping_round_trip_and_state() -> None:
    from sim_mujoco.remote_policy_observation import load_camera_config

    config = load_camera_config(CONFIG_PATH)
    model = mujoco.MjModel.from_xml_path(str(FROZEN_MODEL))
    data = mujoco.MjData(model)
    for raw in (50.0, 200.0, 500.0, 845.0):
        slide = raw_gripper_to_sim_slide(raw, config)
        reconstructed = sim_slide_to_raw_gripper(slide, config)
        assert np.isclose(reconstructed, raw, atol=1e-6)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_slide")
    data.qpos[int(model.jnt_qposadr[joint_id])] = raw_gripper_to_sim_slide(
        200.0, config
    )
    assert np.isclose(gripper_state_to_raw(model, data, config), 200.0, atol=1e-6)


@requires_frozen_split_pad_model
def test_small_friction_search_is_fixed_and_one_dimensional() -> None:
    assert CANDIDATE_MU == (2.0, 0.2, 0.35, 0.5, 0.65, 0.85, 1.15, 1.5)
    validation = validate_search()
    assert validation["passed"] is True
    for mu in CANDIDATE_MU:
        setting = _setting(mu)
        assert set(setting["pad_sliding_friction_by_name"].values()) == {mu}
        assert setting["cone"] == "pyramidal"
        assert setting["impratio"] == 1.0
        expected = [] if mu == 2.0 else ["geom_friction"]
        assert setting["allowed_changed_invariants"] == expected


@requires_frozen_split_pad_model
def test_selected_policy_variant_changes_only_sliding_friction() -> None:
    validation = validate_variant(0.5, 0.5)
    assert validation["changed_compiled_invariants"] == ["geom_friction"]
    assert {
        tuple(value) for value in validation["effective_pad_friction"].values()
    } == {(0.5, 0.02, 0.002)}
