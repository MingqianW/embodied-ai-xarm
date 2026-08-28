from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import mujoco

from data.sim.generation.oracle import OracleStage

from sim_mujoco.gripper_slip_diagnostics import (
    inverse_quantile_normalize,
    reconstruct_network_action,
)
from sim_mujoco.scripts.analyze_real_sim_gripper_trajectories import (
    _phase_boundaries,
)
from sim_mujoco.scripts.analyze_policy_gripper_slip_matrix import _action_rows
from sim_mujoco.scripts.analyze_scripted_gripper_slip_experiments import (
    _analyze_trial,
    _contact_condition_summary,
    _contact_validation,
    _event_times,
    _maximum_finger_speed_mps,
)
from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import (
    BASE_MODEL_PATH,
    _apply_overrides,
    _capture_initial_state,
    _geometry_variant_xml,
    _interpolate_arm_targets,
    _oracle_action_manifest,
    _restore_initial_state,
    _reuse_paired_oracle_plan,
    _settings,
    _trial_settings,
)
from sim_mujoco.scripts.run_contact_model_realism_regression import (
    PROTOCOLS,
    SEEDS,
    contact_conditions,
    experiment_matrix,
)
from sim_mujoco.scripts.analyze_contact_model_realism_regression import (
    _release_latency,
    analyze_trial as _analyze_realism_trial,
    validate_pairs,
)


def test_inverse_quantile_normalization_endpoints() -> None:
    q01 = 206.2904512664795
    q99 = 844.8720790008545
    assert inverse_quantile_normalize(q01, q01=q01, q99=q99) == pytest.approx(-1.0)
    assert inverse_quantile_normalize(q99 + 1e-6, q01=q01, q99=q99) == pytest.approx(
        1.0
    )


def test_reconstruct_network_action_inverts_xarm_output_pipeline() -> None:
    q01 = np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 206.0])
    q99 = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 845.0])
    state = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 500.0])
    expected = np.asarray([-0.8, -0.4, 0.0, 0.2, 0.4, 0.8, -0.25])
    transformed = (expected + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    returned = transformed.copy()
    returned[:6] += state[:6]

    actual = reconstruct_network_action(
        returned,
        state,
        q01=q01,
        q99=q99,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_reconstruct_network_action_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        reconstruct_network_action(
            np.zeros(6),
            np.zeros(7),
            q01=np.zeros(7),
            q99=np.ones(7),
        )


def test_real_sim_phase_detection_finds_closure_lift_and_hold() -> None:
    gripper = np.concatenate(
        [
            np.full(30, 845.0),
            np.linspace(845.0, 200.0, 11)[1:],
            np.full(60, 200.0),
        ]
    )
    tcp_z = np.concatenate(
        [
            np.full(50, 0.10),
            np.linspace(0.10, 0.20, 21)[1:],
            np.full(30, 0.20),
        ]
    )

    boundaries = _phase_boundaries(gripper, tcp_z)

    assert boundaries is not None
    assert 30 <= boundaries["closure_start"] <= 40
    assert boundaries["lift_start"] >= 50
    assert boundaries["hold_start"] > boundaries["lift_start"]
    assert boundaries["hold_end"] == len(gripper)


def test_real_sim_phase_detection_rejects_no_gripper_closure() -> None:
    assert _phase_boundaries(np.full(50, 400.0), np.linspace(0.1, 0.2, 50)) is None


def test_dynamic_suite_covers_required_motion_profiles() -> None:
    settings = _settings("dynamics")
    assert {row.get("motion_profile") for row in settings} == {
        None,
        "horizontal_transport",
        "direction_change",
        "rotation",
    }
    assert {row.get("lift_step_multiplier") for row in settings} == {
        None,
        0.5,
        2.0,
    }


def test_force_suite_scales_servo_gain_not_inactive_force_limit() -> None:
    settings = _settings("force")
    assert [row["kp_multiplier"] for row in settings] == [2.0, 5.0]
    assert [row["force_multiplier"] for row in settings] == [1.0, 1.0]
    assert [row["name"] for row in settings] == ["gripper_kp_2x", "gripper_kp_5x"]


def test_force_override_scales_both_position_servo_terms() -> None:
    model = mujoco.MjModel.from_xml_path(str(BASE_MODEL_PATH))
    environment = SimpleNamespace(
        context=SimpleNamespace(model=model, data=mujoco.MjData(model))
    )
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
    )
    baseline_forcerange = model.actuator_forcerange[actuator_id].copy()

    values = _apply_overrides(
        environment,
        _settings("force")[0],
        target_body="small_block",
    )

    assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(240.0)
    assert model.actuator_biasprm[actuator_id, 1] == pytest.approx(-240.0)
    np.testing.assert_allclose(
        model.actuator_forcerange[actuator_id], baseline_forcerange
    )
    assert values["baseline"]["actuator"]["gainprm"][0] == pytest.approx(120.0)
    assert values["effective"]["actuator"]["gainprm"][0] == pytest.approx(240.0)


def test_contact_suite_changes_only_cone_and_impratio() -> None:
    settings = _settings("contact")
    assert [row["condition"] for row in settings] == ["A", "B", "C"]
    assert [(row["cone"], row["impratio"]) for row in settings] == [
        ("pyramidal", 1.0),
        ("elliptic", 1.0),
        ("elliptic", 10.0),
    ]

    for setting in settings:
        model = mujoco.MjModel.from_xml_path(str(BASE_MODEL_PATH))
        environment = SimpleNamespace(
            context=SimpleNamespace(model=model, data=mujoco.MjData(model))
        )
        values = _apply_overrides(
            environment,
            setting,
            target_body="small_block",
        )
        assert values["changed_invariant_hashes"] == []
        assert values["invariant_hashes_before"] == values["invariant_hashes_after"]
        assert values["effective"]["simulation"]["cone"] == setting["cone"]
        assert values["effective"]["simulation"]["impratio"] == pytest.approx(
            setting["impratio"]
        )
        assert values["effective"]["actuator"] == values["baseline"]["actuator"]
        assert values["effective"]["finger_pads"] == values["baseline"]["finger_pads"]
        assert values["effective"]["target"] == values["baseline"]["target"]


def test_contact_suite_pairs_both_validated_gripper_commands() -> None:
    settings = _trial_settings("contact")
    assert len(settings) == 6
    assert {(row["condition"], row["command_variant"]) for row in settings} == {
        (condition, command)
        for condition in ("A", "B", "C")
        for command in ("oracle_command", "max_closed_raw50")
    }
    assert all(
        row.get("closed_gripper_raw_override") is None
        for row in settings
        if row["command_variant"] == "oracle_command"
    )
    assert all(
        row["closed_gripper_raw_override"] == 50.0
        for row in settings
        if row["command_variant"] == "max_closed_raw50"
    )


def test_realism_suite_is_exact_fixed_sixty_trial_ab_matrix() -> None:
    conditions = contact_conditions()
    assert [(row["condition"], row["cone"], row["impratio"]) for row in conditions] == [
        ("A", "pyramidal", 1.0),
        ("B", "elliptic", 10.0),
    ]
    assert all(row["force_multiplier"] == 1.0 for row in conditions)
    assert all(row["friction_multiplier"] == 1.0 for row in conditions)
    assert set(PROTOCOLS) == {
        "suspended_grasp",
        "pushing",
        "placing_release",
        "tabletop_sliding",
    }
    matrix = experiment_matrix(list(SEEDS))
    assert len(matrix) == 60
    assert {
        protocol: sum(row["protocol"] == protocol for row in matrix)
        for protocol in PROTOCOLS
    } == {
        "suspended_grasp": 18,
        "pushing": 18,
        "placing_release": 6,
        "tabletop_sliding": 18,
    }
    assert all(row["condition"] in {"A", "B"} for row in matrix)


def test_release_latency_requires_sustained_contact_loss() -> None:
    rows = []
    for index in range(100):
        rows.append(
            {
                "sim_time_s": index * 0.002,
                "command": {
                    "stage": "RELEASE" if index >= 10 else "LOWER_TO_TARGET",
                    "gripper_ctrl": 0.15 if index >= 15 else 0.35742,
                    "gripper_returned_raw": 700.0 if index >= 15 else 492.58,
                },
                "contacts": {
                    "target_gripper_contact_count": (
                        0 if index >= 35 or index in {20, 21} else 1
                    )
                },
            }
        )
    onset, latency = _release_latency(rows, 0.002)
    assert onset == pytest.approx(0.03)
    assert latency == pytest.approx(0.04)


def test_realism_pair_validation_checks_state_action_model_and_invariants(
    tmp_path,
) -> None:
    results = []
    trial_paths = {}
    conditions = {row["condition"]: row for row in contact_conditions()}
    for matrix_row in experiment_matrix(list(SEEDS)):
        key = (
            matrix_row["protocol"],
            matrix_row["task"],
            matrix_row["seed"],
        )
        condition = matrix_row["condition"]
        setting = conditions[condition]
        trial_dir = tmp_path / f"{'_'.join(map(str, key))}_{condition}"
        trial_dir.mkdir()
        trial = {
            "protocol": key[0],
            "setting": setting,
            "model_sha256": "same-model",
            "paired_initial_state": {"state_sha256": f"state-{key}"},
            "action_manifest": {"sha256": f"actions-{key}"},
            "overrides": {
                "changed_invariant_hashes": [],
                "invariant_hashes_after": {"fixed": "same"},
                "effective": {
                    "simulation": {
                        "cone": setting["cone"],
                        "impratio": setting["impratio"],
                    }
                },
            },
        }
        trial_path = trial_dir / "trial.json"
        trial_path.write_text(json.dumps(trial) + "\n", encoding="utf-8")
        trial_paths[(*key, condition)] = trial_path
        results.append(
            {
                "protocol": key[0],
                "task": key[1],
                "seed": key[2],
                "setting": setting,
                "artifacts": {"trial": str(trial_path)},
            }
        )

    assert validate_pairs(results)["status"] == "passed"

    changed_path = trial_paths[("pushing", "red_block", 50000, "B")]
    changed = json.loads(changed_path.read_text(encoding="utf-8"))
    changed["overrides"]["invariant_hashes_after"]["fixed"] = "different"
    changed_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    validation = validate_pairs(results)
    assert validation["status"] == "failed"
    assert any(
        "fixed-model invariant hash mismatch" in error for error in validation["errors"]
    )


def test_realism_analysis_computes_all_six_primary_metrics(tmp_path) -> None:
    def row(index: int, *, protocol: str) -> dict:
        if protocol == "placing_release":
            stage = "LOWER_TO_TARGET" if index < 10 else "RELEASE"
            opening_target = 0.038 if index >= 15 else 0.012
            gripper_contacts = 0 if index >= 35 else 1
        elif protocol == "pushing":
            stage = "PUSH"
            opening_target = 0.006
            gripper_contacts = 1
        elif protocol == "tabletop_sliding":
            stage = "FREE_SLIDE"
            opening_target = None
            gripper_contacts = 0
        else:
            stage = "DIAGNOSTIC_SUSPENDED_HOLD"
            opening_target = 0.014
            gripper_contacts = 2
        x_position = 0.001 * (index + 1)
        return {
            "sample_index": index,
            "sim_time_s": 0.002 * index,
            "trial": {"initial_target_position_m": [0.0, 0.0, 0.02]},
            "command": {
                "source": (
                    "scripted_hold"
                    if protocol == "suspended_grasp"
                    else "scripted_realism_regression"
                ),
                "stage": stage,
                "gripper_ctrl": opening_target,
                "gripper_returned_raw": (
                    700.0
                    if protocol == "placing_release" and index >= 15
                    else 492.58
                    if protocol == "placing_release"
                    else None
                ),
            },
            "contacts": {
                "target_gripper_contact_count": gripper_contacts,
                "bilateral": gripper_contacts == 2,
                "maximum_all_target_penetration_m": 0.006,
                "maximum_target_penetration_m": 0.0055,
                "maximum_target_table_penetration_m": 0.001,
                "all_target_normal_sum_n": 3.0,
                "all_target_tangential_sum_n": 0.4,
                "target_gripper_normal_sum_n": 2.0,
                "target_gripper_tangential_sum_n": 0.3,
                "left_target_normal_sum_n": 1.0,
                "right_target_normal_sum_n": 1.0,
                "left_target_tangential_sum_n": 0.15,
                "right_target_tangential_sum_n": 0.15,
                "target_table_normal_sum_n": 1.0,
                "target_table_tangential_sum_n": 0.1,
            },
            "relative": {"drift_m": 0.0001 * index},
            "object": {
                "position_m": [x_position, 0.0, 0.02],
                "linear_velocity_world_mps": [0.1, 0.0, 0.0],
            },
            "tcp": {
                "position_m": [0.4, 0.0, 0.1],
                "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            },
            "simulation": {
                "warning_count": 0,
                "maximum_abs_qvel": 0.1,
                "maximum_abs_qacc": 1.0,
            },
        }

    summaries = {}
    for protocol in PROTOCOLS:
        trace_path = tmp_path / f"{protocol}.jsonl"
        trace_path.write_text(
            "".join(
                json.dumps(row(index, protocol=protocol)) + "\n" for index in range(100)
            ),
            encoding="utf-8",
        )
        summaries[protocol] = _analyze_realism_trial(
            {
                "protocol": protocol,
                "task": (
                    "place_red_pepper_in_ring"
                    if protocol == "placing_release"
                    else "red_block"
                ),
                "seed": 50000,
                "setting": contact_conditions()[0],
                "place_stability": {"stable_place_success": True},
                "artifacts": {"trace": str(trace_path)},
            }
        )

    grasp = summaries["suspended_grasp"]
    assert grasp["maximum_relative_grasp_slip_m"] == pytest.approx(0.0099)
    assert grasp["maximum_normal_contact_force_n"] == pytest.approx(3.0)
    assert grasp["maximum_tangential_contact_force_n"] == pytest.approx(0.4)
    assert grasp["target_gripper_penetration_over_prior_5_44mm"] is True
    assert grasp["target_gripper_penetration_duration_s"] == pytest.approx(0.2)

    placing = summaries["placing_release"]
    assert placing["opening_target_onset_s"] == pytest.approx(0.03)
    assert placing["release_latency_s"] == pytest.approx(0.04)
    assert placing["place_stable_success"] is True
    assert summaries["pushing"]["pushing_x_displacement_m"] == pytest.approx(0.1)
    assert summaries["tabletop_sliding"]["sliding_x_displacement_m"] == pytest.approx(
        0.1
    )
    assert all(summary["simulation_finite"] for summary in summaries.values())


def test_contact_analysis_reports_bilateral_slip_velocity_and_force() -> None:
    rows = []
    for index in range(20):
        slip = 0.003 * index / 19
        rows.append(
            {
                "sample_index": index,
                "sim_time_s": 1.0 + 0.002 * index,
                "trial": {
                    "target_mass_kg": 0.012,
                    "overrides": {
                            "effective": {
                                "simulation": {
                                    "cone": "pyramidal",
                                    "impratio": 1.0,
                                },
                                "actuator": {
                                    "forcerange_actuator_space": [-8.0, 8.0]
                                },
                            }
                    },
                },
                "command": {"source": "scripted_hold"},
                "contacts": {
                    "left_target_count": 1,
                    "right_target_count": 1,
                    "bilateral": True,
                    "target_table_count": 0,
                    "target_gripper_normal_sum_n": 2.0,
                    "target_gripper_tangential_sum_n": 0.2,
                    "left_target_normal_sum_n": 1.0,
                    "right_target_normal_sum_n": 1.0,
                    "left_target_tangential_sum_n": 0.1,
                    "right_target_tangential_sum_n": 0.1,
                    "maximum_target_penetration_m": 0.0002,
                },
                "relative": {
                    "downward_slip_m": slip,
                    "vertical_slip_m": slip,
                    "vertical_slip_velocity_mps": 0.001,
                },
                "actuator": {
                    "force_fraction": 0.1,
                    "force_actuator_space": 4.0,
                },
                "object": {
                    "lift_height_m": 0.1,
                    "linear_acceleration_world_mps2": [0.0, 0.0, 0.0],
                    "angular_velocity_world_radps": [0.0, 0.0, 0.0],
                    "position_m": [0.4, 0.0, 0.1],
                    "linear_velocity_world_mps": [0.0, 0.0, -0.001],
                },
                "tcp": {
                    "linear_acceleration_world_mps2": [0.0, 0.0, 0.0],
                    "linear_velocity_world_mps": [0.0, 0.0, 0.0],
                    "position_m": [0.4, 0.0, 0.2],
                },
                "fingers": {"left_qvel_mps": 0.0, "right_qvel_mps": 0.0},
                "simulation": {
                    "solver_iterations": 2,
                    "solver_fwdinv": [0.0, 0.0],
                    "warning_count": 0,
                    "maximum_abs_qvel": 0.001,
                    "maximum_abs_qacc": 0.01,
                },
            }
        )
    result = {
        "task": "red_block",
        "seed": 50000,
        "hold_kind": "suspended",
        "setting": {
            "name": "pyramidal_impratio1",
            "condition": "A",
            "command_variant": "oracle_command",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
        },
    }

    summary = _analyze_trial(result, rows)

    assert summary["diagnostic_failure_label"] == "STATIC_CONTACT_SLIP"
    assert summary["maximum_bilateral_downward_slip_m"] == pytest.approx(0.003)
    assert summary["mean_stationary_slip_velocity_mps"] == pytest.approx(0.001)
    assert summary["peak_stationary_slip_velocity_mps"] == pytest.approx(0.001)
    assert summary["mean_bilateral_normal_force_sum_n"] == pytest.approx(2.0)
    assert summary["mean_abs_actuator_force_actuator_space"] == pytest.approx(4.0)
    assert summary["simulation_finite"] is True

    contact_summary = _contact_condition_summary(
        [
            {
                **summary,
                "condition": condition,
                "setting": setting,
                "effective_cone": cone,
                "effective_impratio": impratio,
            }
            for condition, setting, cone, impratio in (
                ("A", "pyramidal_impratio1", "pyramidal", 1.0),
                ("B", "elliptic_impratio1", "elliptic", 1.0),
                ("C", "elliptic_impratio10", "elliptic", 10.0),
            )
        ]
    )
    assert [row["bilateral_slip_count"] for row in contact_summary] == [1, 1, 1]


def test_menagerie_finger_speed_uses_half_aperture_rate() -> None:
    rows = [
        {"sim_time_s": 1.0, "fingers": {"opening_width_m": 0.04}},
        {"sim_time_s": 1.1, "fingers": {"opening_width_m": 0.038}},
        {"sim_time_s": 1.2, "fingers": {"opening_width_m": 0.034}},
    ]

    assert _maximum_finger_speed_mps(rows) == pytest.approx(0.02)


def test_contact_validation_covers_both_command_variants(tmp_path) -> None:
    results = {"trials": []}
    summaries = []
    comparisons = []
    for command_variant, baseline_setting, command_raw in (
        ("oracle_command", "baseline_oracle_command", 200.0),
        ("max_closed_raw50", "baseline_max_closed_raw50", 50.0),
    ):
        comparisons.append(
            {
                "task": "red_block",
                "seed": 50000,
                "hold_kind": "suspended",
                "setting": baseline_setting,
                "maximum_downward_slip_m": 0.01,
                "hold_bilateral_contact_fraction": 0.5,
                "maximum_abs_actuator_force_actuator_space": 2.0,
                "drop_time_from_hold_start_s": 1.5,
                "diagnostic_failure_label": "CONTACT_LOSS",
                "hold_success": False,
            }
        )
        for condition, setting, cone, impratio in (
            ("A", "pyramidal_impratio1", "pyramidal", 1.0),
            ("B", "elliptic_impratio1", "elliptic", 1.0),
            ("C", "elliptic_impratio10", "elliptic", 10.0),
        ):
            trial_dir = tmp_path / f"{setting}_{command_variant}"
            trial_dir.mkdir()
            base_configuration = {
                "simulation": {
                    "cone": "elliptic",
                    "cone_enum": 1,
                    "impratio": 10.0,
                    "solver": "mjSOL_NEWTON",
                },
                "actuator": {"kp_n_per_m": 500.0},
                "target": {"body": "small_block", "mass_kg": 0.012},
            }
            effective_configuration = json.loads(json.dumps(base_configuration))
            effective_configuration["simulation"].update(
                {
                    "cone": cone,
                    "cone_enum": 0 if cone == "pyramidal" else 1,
                    "impratio": impratio,
                }
            )
            setting_value = {
                "name": setting,
                "condition": condition,
                "command_variant": command_variant,
                "closed_gripper_raw_override": (
                    50.0 if command_variant == "max_closed_raw50" else None
                ),
            }
            trial = {
                "setting": setting_value,
                "model_path": "/model.xml",
                "model_sha256": "model-hash",
                "oracle_config": {"closed_gripper_raw": command_raw},
                "oracle_plan": {"trajectory": [1, 2, 3]},
                "oracle_action_manifest": {
                    "sha256": f"actions-{command_variant}",
                    "total_action_count": 3,
                },
                "oracle_plan_source_condition": (
                    "A" if condition == "A" else "A-reused"
                ),
                "target_body": "small_block",
                "target_mass_kg": 0.012,
                "initial_target_z_m": 0.02,
                "paired_initial_state": {
                    "state_spec": 4095,
                    "state_size": 42,
                    "state_sha256": f"state-{command_variant}",
                    "state": None,
                    "initial_target_z_m": 0.02,
                    "initial_conditions": {
                        "task": "red_block",
                        "seed": 50000,
                    },
                    "source_condition": ("A" if condition == "A" else "A-restored"),
                },
                "overrides": {
                    "baseline": base_configuration,
                    "effective": effective_configuration,
                    "invariant_hashes_before": {"all": "same"},
                    "invariant_hashes_after": {"all": "same"},
                    "changed_invariant_hashes": [],
                },
            }
            trial_path = trial_dir / "trial.json"
            trial_path.write_text(json.dumps(trial) + "\n", encoding="utf-8")
            trace_path = trial_dir / "physics_trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "object": {
                            "position_m": [0.4, 0.0, 0.02],
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        "tcp": {"position_m": [0.4, 0.0, 0.12]},
                        "fingers": {
                            "left_driver_qpos_rad": 0.35,
                            "right_driver_qpos_rad": 0.35,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            results["trials"].append(
                {
                    "task": "red_block",
                    "seed": 50000,
                    "hold_kind": "suspended",
                    "setting": setting_value,
                    "artifacts": {
                        "trial": str(trial_path),
                        "trace": str(trace_path),
                    },
                }
            )
            summaries.append(
                {
                    "task": "red_block",
                    "seed": 50000,
                    "hold_kind": "suspended",
                    "setting": setting,
                    "condition": condition,
                    "command_variant": command_variant,
                    "maximum_downward_slip_m": 0.01,
                    "hold_bilateral_contact_fraction": 0.5,
                    "maximum_abs_actuator_force_actuator_space": 2.0,
                    "contact_loss_from_hold_start_s": 1.5,
                    "diagnostic_failure_label": "CONTACT_LOSS",
                    "hold_success": False,
                }
            )

    validation = _contact_validation(
        results,
        summaries,
        [*summaries, *comparisons],
    )

    assert validation["passed"] is True
    assert validation["actual_trial_count"] == 6
    assert len(validation["pairing"]) == 2
    assert len(validation["baseline_reproduction"]) == 2
    assert validation["only_cone_and_impratio_changed"] is True


def test_paired_initial_state_round_trip_is_exact() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="object" pos="0 0 0.2">
              <freejoint/>
              <geom type="box" size="0.01 0.01 0.01" mass="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[0] = 0.125
    data.qvel[2] = -0.25
    data.time = 1.25
    mujoco.mj_forward(model, data)
    reference_object_position = data.xpos[1].copy()
    environment = SimpleNamespace(
        context=SimpleNamespace(model=model, data=data),
        task_runtime=SimpleNamespace(initial_target_z=0.19),
        initial_conditions={"task": "red_block", "seed": 50000},
        _last_step_started_s=0.0,
        _last_step_duration_s=1.0,
    )
    reference = _capture_initial_state(environment)

    data.qpos[0] = -2.0
    data.qvel[2] = 3.0
    data.time = 9.0
    environment.task_runtime.initial_target_z = -1.0
    environment.initial_conditions = {"mutated": True}
    _restore_initial_state(environment, reference)
    restored = _capture_initial_state(environment)

    assert restored["state_sha256"] == reference["state_sha256"]
    assert restored["state"] == reference["state"]
    assert environment.task_runtime.initial_target_z == pytest.approx(0.19)
    assert environment.initial_conditions == {"task": "red_block", "seed": 50000}
    assert environment._last_step_started_s == pytest.approx(1.25)
    assert environment._last_step_duration_s == 0.0
    np.testing.assert_array_equal(data.xpos[1], reference_object_position)


def test_condition_a_oracle_plan_is_reused_exactly() -> None:
    controller = SimpleNamespace(
        stage="FAILED",
        failure_reason="candidate_plan_failure",
        action_steps=7,
        transitions=["reset", "failed"],
        plan={"trajectory": [9.0]},
        _stage_action_index=4,
    )
    controller._build_stage_actions = lambda: {
        "trajectory": controller.plan["trajectory"]
    }
    reference = {"trajectory": [1.0, 2.0, 3.0]}

    _reuse_paired_oracle_plan(controller, reference)

    assert controller.stage == OracleStage.RESET
    assert controller.failure_reason is None
    assert controller.action_steps == 0
    assert controller.transitions == ["reset"]
    assert controller.plan == reference
    assert controller.plan is not reference
    assert controller._stage_actions == {"trajectory": [1.0, 2.0, 3.0]}
    assert controller._stage_action_index == 0


def test_oracle_action_manifest_detects_any_command_change() -> None:
    sequence = (OracleStage.RESET, OracleStage.OPEN_GRIPPER)
    first = SimpleNamespace(
        _SEQUENCE=sequence,
        _stage_actions={
            OracleStage.RESET: [],
            OracleStage.OPEN_GRIPPER: [np.arange(7), np.arange(7) + 1],
        },
    )
    identical = SimpleNamespace(
        _SEQUENCE=sequence,
        _stage_actions={
            OracleStage.RESET: [],
            OracleStage.OPEN_GRIPPER: [np.arange(7), np.arange(7) + 1],
        },
    )
    changed = SimpleNamespace(
        _SEQUENCE=sequence,
        _stage_actions={
            OracleStage.RESET: [],
            OracleStage.OPEN_GRIPPER: [
                np.arange(7),
                np.arange(7) + np.asarray([1, 1, 1, 1, 1, 1, 1.001]),
            ],
        },
    )

    first_manifest = _oracle_action_manifest(first)
    assert first_manifest == _oracle_action_manifest(identical)
    assert first_manifest["sha256"] != _oracle_action_manifest(changed)["sha256"]
    assert first_manifest["total_action_count"] == 2


def test_arm_interpolation_obeys_maximum_joint_step() -> None:
    targets = _interpolate_arm_targets(
        np.zeros(6),
        np.asarray([0.051, -0.02, 0.0, 0.0, 0.0, 0.0]),
        max_step_rad=0.01,
    )
    steps = np.diff(np.vstack([np.zeros(6), targets]), axis=0)
    assert np.max(np.abs(steps)) <= 0.01 + 1e-12
    np.testing.assert_allclose(targets[-1], [0.051, -0.02, 0, 0, 0, 0])


def test_policy_action_rows_deduplicate_physics_samples() -> None:
    rows = [
        {
            "command": {
                "inference_index": inference,
                "action_index_in_chunk": action,
                "action_step": step,
            }
        }
        for inference, action, step in (
            (0, 0, 0),
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 2),
            (1, 0, 2),
        )
    ]
    assert len(_action_rows(rows)) == 3


def test_static_table_contact_is_not_classified_as_post_lift_impact() -> None:
    rows = [
        {
            "sample_index": index,
            "sim_time_s": 0.002 * index,
            "object": {"lift_height_m": 0.0},
            "relative": {"downward_slip_m": 0.0},
            "contacts": {
                "left_target_count": 1,
                "right_target_count": 1,
                "bilateral": True,
                "target_table_count": 1,
            },
            "actuator": {"force_fraction": 0.5},
        }
        for index in range(20)
    ]

    events = _event_times(rows)

    assert events["bilateral_grasp_s"] == 0.0
    assert events["lift_onset_s"] is None
    assert events["table_impact_s"] is None


def test_legacy_three_patch_geometry_variant_rejects_local_four_bar_model() -> None:
    setting = next(
        row
        for row in _settings("geometry")
        if row.get("geometry_variant") == "three_patch_pad_same_envelope"
    )
    with pytest.raises(RuntimeError, match="incompatible with the Menagerie hand"):
        _geometry_variant_xml(setting)
