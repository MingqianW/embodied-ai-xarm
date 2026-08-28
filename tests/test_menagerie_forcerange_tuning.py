from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import _settings


def test_forcerange_suite_is_exact_and_isolated():
    settings = _settings("menagerie_forcerange")

    assert [row["force_limit_actuator_space"] for row in settings] == [
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
    ]
    assert all(row["force_multiplier"] == 1.0 for row in settings)
    assert all(row["friction_multiplier"] == 1.0 for row in settings)
    assert all(row["gripper_closing_rate_raw_per_s"] == 244.0 for row in settings)
    assert all(row["gripper_opening_rate_raw_per_s"] == 220.0 for row in settings)
