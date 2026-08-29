from __future__ import annotations

import numpy as np

from diagnostics.simulation.camera.calibration import camera_axes
from diagnostics.simulation.camera.calibration import parameter_vector
from diagnostics.simulation.camera.calibration import vector_parameters
from diagnostics.simulation.camera.cli import _parser


def test_camera_axes_is_orthonormal_and_points_at_target() -> None:
    position = np.asarray([1.0, -0.2, 0.7])
    target = np.asarray([0.4, 0.0, 0.2])

    axes = camera_axes(position, target, roll_deg=2.0)

    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
    expected_camera_z = -(target - position) / np.linalg.norm(target - position)
    np.testing.assert_allclose(axes[:, 2], expected_camera_z, atol=1e-12)


def test_camera_parameter_vector_round_trip() -> None:
    original = {
        "position": [1.0, 2.0, 3.0],
        "target": [0.1, 0.2, 0.3],
        "roll_deg": 2.0,
        "fovy_deg": 57.0,
        "retained_metadata": "yes",
    }

    restored = vector_parameters(parameter_vector(original), original)

    assert restored == original


def test_camera_cli_has_only_maintained_workflows() -> None:
    parser = _parser()
    for command in ("discover", "select", "fit", "evaluate", "render"):
        assert parser.parse_args([command]).command == command
