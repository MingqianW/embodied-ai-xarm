"""CLI regression tests for the compiled-physics diagnostic."""

from __future__ import annotations

import json

from diagnostics.simulation.physics import consistency


def test_main_writes_json_to_stdout_by_default(monkeypatch, capsys) -> None:
    expected = {"compiled_physics_exact_match_all_tasks": True}
    monkeypatch.setattr(consistency, "audit", lambda *_args: expected)
    monkeypatch.setattr("sys.argv", ["consistency"])

    consistency.main()

    assert json.loads(capsys.readouterr().out) == expected
