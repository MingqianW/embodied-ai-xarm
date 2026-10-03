import subprocess
import json
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np

import pytest

from evaluation.sim.service import require_allocation, stop_owned_process, wait_ready
from evaluation.sim.service import restore_policy
from evaluation.sim.service import configure_allocated_egl
from evaluation.common.models import ModelSpec


def test_service_refuses_login_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        require_allocation()


def test_readiness_detects_early_server_exit() -> None:
    process = SimpleNamespace(poll=lambda: 1, returncode=1)
    with pytest.raises(RuntimeError, match="before readiness"):
        wait_ready(process, SimpleNamespace(readiness_timeout=1), {})


def test_egl_uses_allocated_global_gpu_instead_of_cuda_remapped_zero(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("SLURM_JOB_GPUS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    configure_allocated_egl()
    assert __import__("os").environ["MUJOCO_EGL_DEVICE_ID"] == "2"


@pytest.mark.parametrize("devices", ["", "0,1", "GPU-unknown"])
def test_egl_refuses_ambiguous_allocation(devices, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("SLURM_JOB_GPUS", devices)
    with pytest.raises(RuntimeError, match="exactly one numeric"):
        configure_allocated_egl()


def test_cleanup_kills_owned_process_after_terminate_timeout() -> None:
    calls = []
    def wait(*, timeout):
        calls.append("wait")
        if calls.count("wait") == 1:
            raise subprocess.TimeoutExpired("owned-model", timeout)
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("terminate"), kill=lambda: calls.append("kill"), wait=wait)
    stop_owned_process(process)
    assert calls == ["terminate", "wait", "kill", "wait"]


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_identity_is_generated_only_after_selected_restore_and_warmup(tmp_path, monkeypatch, warmup_fails) -> None:
    from training.openpi import inference
    from evaluation.sim import service
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    norm = tmp_path / "checkpoint/assets/own_asset/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text(json.dumps({"norm_stats": {"state": {"mean": [0] * 7}}}))
    model = ModelSpec("HF_TEST", "selected_config", norm.parents[2], None, "own_asset")
    config = SimpleNamespace(model=SimpleNamespace(action_horizon=10, action_dim=32))
    calls = []
    def resolve(name, *, openpi_root):
        calls.append(("config", name, openpi_root))
        return config
    def infer(observation):
        calls.append(("warmup",))
        if warmup_fails:
            raise RuntimeError("Warmup failed")
        return {"actions": np.zeros((10, 7))}
    loaded = SimpleNamespace(metadata={"request_rng_required": True}, infer=infer)
    def restore(selected, checkpoint):
        calls.append(("restore", selected, checkpoint))
        return loaded
    upstream = ModuleType("openpi.policies.policy_config")
    upstream.create_trained_policy = restore
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", upstream)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda: [SimpleNamespace(platform="gpu")], __version__="fixture"))
    monkeypatch.setattr(inference, "resolve_inference_config", resolve)
    monkeypatch.setattr(service, "RequestRngPolicy", lambda policy, **kwargs: policy)
    args = SimpleNamespace(openpi_root=tmp_path / "openpi", load_report=tmp_path / "load.json")
    provenance = {"protocol": {"tasks": [{"prompt": "canonical fixture prompt"}]}, "evaluation_protocol_version": "fixture", "protocol_sha256": "protocol", "model_spec_sha256": "model", "provenance_sha256": "provenance", "model": model.to_json()}
    if warmup_fails:
        with pytest.raises(RuntimeError, match="Warmup failed"):
            restore_policy(args, model, provenance)
        assert not args.load_report.exists()
    else:
        _, metadata = restore_policy(args, model, provenance)
        assert metadata["formal_evaluation_provenance"]["model"] == model.to_json()
        assert json.loads(args.load_report.read_text())["model_restored"] is True
    assert calls[:2] == [("config", "selected_config", args.openpi_root), ("restore", config, model.manager_directory)]
