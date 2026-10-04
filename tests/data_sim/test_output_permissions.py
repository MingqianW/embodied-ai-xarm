import os

import pytest

from data.sim.generation import safety


def test_owner_only_preserves_exact_root_guard_and_private_modes(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    monkeypatch.setenv("XARM_OUTPUT_PERMISSION_POLICY", "owner_only")
    monkeypatch.setattr(safety, "AUTHORIZED_ROOTS", frozenset({root}))
    monkeypatch.setattr(safety, "LOG_PARENT", tmp_path / "logs")
    monkeypatch.setattr(safety, "_require_delta_group", lambda: pytest.fail("Delta group lookup"))
    with pytest.raises(ValueError, match="exact authorized"):
        safety.replace_authorized_roots([tmp_path], overwrite=True, git_sha="test", config_path=tmp_path / "plan")
    safety.replace_authorized_roots([root], overwrite=True, git_sha="test", config_path=tmp_path / "plan")
    frame = root / "frame"
    frame.write_text("fixture")
    os.chmod(frame, 0o666)
    safety.apply_group_permissions([root])
    assert root.stat().st_mode & 0o777 == 0o700
    assert frame.stat().st_mode & 0o777 == 0o600


def test_unknown_permission_override_fails_closed(monkeypatch):
    monkeypatch.setenv("XARM_OUTPUT_PERMISSION_POLICY", "skip")
    with pytest.raises(ValueError, match="owner_only"):
        safety.permission_policy()
