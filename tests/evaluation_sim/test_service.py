import subprocess
from types import SimpleNamespace

import pytest

from evaluation.sim.service import require_allocation, stop_owned_process, wait_ready


def test_service_refuses_login_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        require_allocation()


def test_readiness_detects_early_server_exit() -> None:
    process = SimpleNamespace(poll=lambda: 1, returncode=1)
    with pytest.raises(RuntimeError, match="before readiness"):
        wait_ready(process, SimpleNamespace(readiness_timeout=1), {})


def test_cleanup_kills_owned_process_after_terminate_timeout() -> None:
    calls = []
    def wait(*, timeout):
        calls.append("wait")
        if calls.count("wait") == 1:
            raise subprocess.TimeoutExpired("owned-model", timeout)
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("terminate"), kill=lambda: calls.append("kill"), wait=wait)
    stop_owned_process(process)
    assert calls == ["terminate", "wait", "kill", "wait"]
