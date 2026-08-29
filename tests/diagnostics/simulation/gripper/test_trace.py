from __future__ import annotations

import numpy as np
import pytest

from diagnostics.simulation.gripper.trace import inverse_quantile_normalize
from diagnostics.simulation.gripper.trace import reconstruct_network_action


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

    actual = reconstruct_network_action(returned, state, q01=q01, q99=q99)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_reconstruct_network_action_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        reconstruct_network_action(
            np.zeros(6),
            np.zeros(7),
            q01=np.zeros(7),
            q99=np.ones(7),
        )
