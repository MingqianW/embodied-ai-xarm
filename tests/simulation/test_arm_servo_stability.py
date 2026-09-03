from __future__ import annotations

import mujoco
import numpy as np

from simulation.resources import DEFAULT_MODEL_PATH


def test_arm_servo_uses_xarm6_calibrated_graded_pd_parameters() -> None:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))

    np.testing.assert_allclose(
        model.dof_damping[:6],
        np.asarray([10.0, 10.0, 5.0, 5.0, 5.0, 2.0]),
        atol=0.0,
    )
    np.testing.assert_allclose(model.dof_armature[:6], 0.1, atol=0.0)
    np.testing.assert_allclose(
        model.actuator_gainprm[:6, 0],
        np.asarray([120.0, 120.0, 100.0, 70.0, 50.0, 30.0]),
        atol=0.0,
    )
    np.testing.assert_allclose(
        model.actuator_biasprm[:6, 2],
        np.asarray([-1.0, -3.0, -8.0, 0.0, 0.0, 0.0]),
        atol=0.0,
    )


def test_arm_position_servo_settles_after_a_stopped_target() -> None:
    """Guard against under-damped arm position control in the canonical MJCF."""

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    target = data.qpos[:6].copy() + np.asarray(
        [0.10, 0.10, -0.10, 0.10, -0.10, 0.10],
        dtype=np.float64,
    )
    data.ctrl[:6] = target
    samples: list[np.ndarray] = []
    for _ in range(2_500):
        mujoco.mj_step(model, data)
        samples.append(data.qpos[:6].copy())

    settled_error = np.abs(np.asarray(samples)[500:] - target)
    assert float(settled_error.max()) < 0.0015
