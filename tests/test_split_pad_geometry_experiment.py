from __future__ import annotations

from sim_mujoco.scripts.analyze_contact_model_realism_regression import (
    _manifold_metrics,
)
from sim_mujoco.scripts.analyze_split_pad_geometry_experiment import (
    _validate_pad_topology,
)
from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import _sha256
from sim_mujoco.scripts.run_split_pad_geometry_experiment import (
    BASE_MODEL_PATH,
    build_split_pad_model,
    geometry_conditions,
    validate_split_pad_model,
)


def test_split_pad_model_is_runtime_only_and_topology_only(tmp_path):
    production_hash = _sha256(BASE_MODEL_PATH)
    target = tmp_path / "runtime_model" / "diagnostic_split_pad.xml"

    descriptor = build_split_pad_model(target)
    validation = validate_split_pad_model(target)

    assert target.is_file()
    assert descriptor["production_model_file_modified"] is False
    assert descriptor["production_model_sha256"] == production_hash
    assert _sha256(BASE_MODEL_PATH) == production_hash
    assert validation["passed"] is True
    assert validation["added_geom_count"] == 2
    assert validation["invariants_identical"] is True
    assert validation["finger_body_mass_and_inertia_identical"] is True


def test_split_pad_conditions_change_only_geometry():
    conditions = geometry_conditions()

    assert [value["condition"] for value in conditions] == ["A", "B"]
    for value in conditions:
        assert value["cone"] == "pyramidal"
        assert value["impratio"] == 1.0
        assert value["force_multiplier"] == 1.0
        assert value["friction_multiplier"] == 1.0
    assert conditions[0].get("geometry_variant") is None
    assert conditions[1]["geometry_variant"] == "split_pad_two_zone_same_envelope"


def test_manifold_metrics_preserve_first_bilateral_positions():
    rows = [
        {
            "contacts": {
                "target_gripper_contact_count": 5,
                "left_target_count": 1,
                "right_target_count": 4,
                "bilateral": True,
                "left_target_contact_positions_world_m": [[0.4, -0.01, 0.05]],
                "right_target_contact_positions_world_m": [
                    [0.4, 0.01, 0.04],
                    [0.4, 0.01, 0.05],
                    [0.4, 0.01, 0.06],
                    [0.4, 0.01, 0.07],
                ],
            }
        }
    ]

    metrics = _manifold_metrics(rows)

    assert metrics["first_bilateral_left_contact_count"] == 1
    assert metrics["first_bilateral_right_contact_count"] == 4
    assert metrics["first_bilateral_left_contact_positions_world_m"] == [
        [0.4, -0.01, 0.05]
    ]
    assert metrics["mean_bilateral_contact_count_asymmetry"] == 3.0
    assert metrics["mean_bilateral_contact_count_symmetry"] == 0.4
    assert metrics["bilateral_exact_count_symmetry_fraction"] == 0


def test_pad_topology_validation_allows_only_pad_identity_size_and_position_changes():
    def pad(name, size, position):
        return {
            "name": name,
            "size_m": size,
            "pos_m": position,
            "friction": [2.0, 0.02, 0.002],
            "condim": 3,
            "solref": [0.02, 1.0],
            "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
            "margin_m": 0.0,
            "gap_m": 0.0,
            "type": 6,
        }

    a = {
        "finger_pads": [
            pad(
                "left_fingertip_pad",
                [0.016, 0.003, 0.018],
                [0.0, -0.0075, 0.052],
            ),
            pad(
                "right_fingertip_pad",
                [0.016, 0.003, 0.018],
                [0.0, 0.0075, 0.052],
            ),
        ]
    }
    b = {
        "finger_pads": [
            pad(
                "left_fingertip_pad",
                [0.016, 0.003, 0.009],
                [0.0, -0.0075, 0.043],
            ),
            pad(
                "left_fingertip_pad_upper",
                [0.016, 0.003, 0.009],
                [0.0, -0.0075, 0.061],
            ),
            pad(
                "right_fingertip_pad",
                [0.016, 0.003, 0.009],
                [0.0, 0.0075, 0.043],
            ),
            pad(
                "right_fingertip_pad_upper",
                [0.016, 0.003, 0.009],
                [0.0, 0.0075, 0.061],
            ),
        ]
    }

    assert _validate_pad_topology(a, b) == []
