"""Allocation-local OpenPI lifecycle for the canonical formal evaluator.

The server restores the configured checkpoint itself and derives its identity
after restoration. The controller invokes evaluation.sim.cli unchanged.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np

from evaluation.sim.cli import _parser, _resolve, _verify_server
from evaluation.sim.provenance import server_provenance
from policy_runtime.openpi_request_rng import RequestRngPolicy
from policy_runtime.remote_policy_client import (
    PolicyConnectionError, PolicyTimeoutError, RemotePolicyClient, RemotePolicyConfig,
    REQUEST_RNG_SEED_KEY,
)


def require_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Model service and rendering require a Slurm allocation")


def _egl_cuda_identities() -> tuple[str, list[dict[str, Any]]]:
    """Query identities without creating EGL displays or probing other GPUs.

    EGL_NV_device_cuda exposes a CUDA device handle, not a Slurm global ID:
    https://registry.khronos.org/EGL/extensions/NV/EGL_NV_device_cuda.txt
    """
    from mujoco.egl import egl_ext as egl

    cuda = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value != 1:
        raise RuntimeError("EGL selection requires exactly one visible allocated CUDA device")

    def uuid(device: int) -> str | None:
        value = (ctypes.c_ubyte * 16)()
        if cuda.cuDeviceGetUuid(ctypes.byref(value), ctypes.c_int(device)) != 0:
            return None
        return bytes(value).hex()

    allocated = uuid(0)
    if allocated is None:
        raise RuntimeError("Cannot read the allocated CUDA GPU UUID")
    string_address = egl.eglGetProcAddress("eglQueryDeviceStringEXT")
    attribute_address = egl.eglGetProcAddress("eglQueryDeviceAttribEXT")
    if not string_address or not attribute_address:
        raise RuntimeError("EGL device identity query extensions are unavailable")
    query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(string_address)
    query_attribute = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
    )(attribute_address)
    rows = []
    for index, device in enumerate(egl.eglQueryDevicesEXT()):
        pointer = ctypes.cast(device, ctypes.c_void_p)
        extensions = query_string(pointer, 0x3055) or b""  # EGL_EXTENSIONS
        identity = None
        if b"EGL_NV_device_cuda" in extensions.split():
            ordinal = ctypes.c_ssize_t(-1)
            if query_attribute(pointer, 0x323A, ctypes.byref(ordinal)):  # EGL_CUDA_DEVICE_NV
                identity = uuid(ordinal.value)
            egl.eglGetError()
        rows.append({"egl_index": index, "cuda_uuid_hex": identity})
    return allocated, rows


def configure_allocated_egl() -> dict[str, Any]:
    require_allocation()
    selected = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS", "")
    if not selected.isdecimal():
        raise RuntimeError("Local service requires exactly one numeric Slurm GPU device ID")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("EGL selection requires exactly one visible allocated CUDA device")
    allocated, rows = _egl_cuda_identities()
    matching = [row["egl_index"] for row in rows if row["cuda_uuid_hex"] == allocated]
    if len(matching) != 1:
        raise RuntimeError(f"Allocated GPU UUID has no unique EGL match: {matching}")
    # MuJoCo indexes EGL's enumeration, which may differ from both Slurm and
    # CUDA indices. Only the hardware-identity-matched display may initialize.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(matching[0])
    os.environ["XARM_ALLOCATED_GPU_UUID_HEX"] = allocated
    return {"slurm_gpu_id": selected, "allocated_cuda_uuid_hex": allocated,
            "egl_device_index": matching[0], "egl_devices": rows}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def restore_policy(args: Any, model: Any, provenance: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    require_allocation()
    from training.openpi.inference import resolve_inference_config, strict_parameter_config
    from openpi.policies.policy_config import create_trained_policy
    import jax

    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-backed JAX is required; found {devices}")
    config = resolve_inference_config(model.training_config, openpi_root=args.openpi_root)
    if (config.model.action_horizon, config.model.action_dim) != (10, 32):
        raise ValueError("This service supports the audited JAX Pi0/Pi0.5 10x32 sampling contract")
    started = time.monotonic()
    loaded = create_trained_policy(strict_parameter_config(config), model.manager_directory)
    policy = RequestRngPolicy(loaded, horizon=config.model.action_horizon, internal_action_dim=config.model.action_dim)
    norm_path = model.manager_directory / "assets" / model.norm_asset_id / "norm_stats.json"
    state = np.asarray(json.loads(norm_path.read_text())["norm_stats"]["state"]["mean"], dtype=np.float32)
    observation = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": state,
        "prompt": provenance["protocol"]["tasks"][0]["prompt"],
        REQUEST_RNG_SEED_KEY: 0,
    }
    actions = np.asarray(policy.infer(observation)["actions"])
    # Identity is attached only after real restoration and a validated warm-up.
    metadata = {**policy.metadata, "formal_evaluation_provenance": server_provenance(provenance)}
    _write_new(args.load_report, {
        "model_restored": True, "warmup_action_shape": list(actions.shape),
        "strict_parameter_restore": True,
        "warmup_actions_finite": bool(np.isfinite(actions).all()),
        "restore_and_warmup_seconds": time.monotonic() - started,
        "jax_devices": [str(device) for device in devices],
        "jax_version": jax.__version__, "metadata": metadata,
        "checkpoint": str(model.manager_directory),
        "gpu_selection": {key: os.environ.get(key) for key in (
            "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES",
            "MUJOCO_EGL_DEVICE_ID", "XARM_ALLOCATED_GPU_UUID_HEX",
        )},
    })
    return policy, metadata


def stop_owned_process(process: Any) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def wait_ready(process: Any, args: Any, provenance: dict[str, Any]) -> None:
    deadline = time.monotonic() + args.readiness_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited before readiness: {process.returncode}")
        try:
            with RemotePolicyClient(RemotePolicyConfig(args.host, args.port, 2, args.timeout)) as client:
                _verify_server(client, provenance)
            return
        except (PolicyConnectionError, PolicyTimeoutError):
            time.sleep(5)
    raise TimeoutError("Policy server restoration/warm-up exceeded readiness deadline")


def verification_observation(protocol: Any) -> dict[str, Any]:
    from simulation.runtime import initialize_scene, load_simulation
    from simulation.observation.policy import build_policy_observation

    simulation = load_simulation(protocol.robot_xml_path, protocol.camera_config_path)
    try:
        initialize_scene(simulation.model, simulation.data)
        observation = build_policy_observation(
            simulation.model, simulation.data, simulation.renderer, simulation.config,
            prompt=protocol.tasks[0].prompt,
        )
        for key in ("observation/image", "observation/wrist_image"):
            image = observation[key]
            if image.shape != (224, 224, 3) or image.dtype != np.uint8 or float(image.std()) == 0:
                raise ValueError(f"Invalid EGL-rendered policy image: {key}")
        if not np.isfinite(observation["observation/state"]).all():
            raise ValueError("Nonfinite canonical simulation state")
        return observation
    finally:
        simulation.close()


def check_requests(args: Any, provenance: dict[str, Any], observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    with RemotePolicyClient(RemotePolicyConfig(args.host, args.port, 5, args.timeout)) as client:
        _verify_server(client, provenance)
        outputs = [np.asarray(client.infer(observation, rng_seed=seed)["actions"]) for seed in (50018, 50019, 50018)]
        if not np.array_equal(outputs[0], outputs[2]):
            raise RuntimeError("Repeated request seed changed actions after an intervening request")
        if np.array_equal(outputs[0], outputs[1]):
            raise RuntimeError("Different request seeds produced identical actions")
        missing_seed_rejected = False
        try:
            client.infer(observation)
        except RuntimeError as exc:
            if "requires an unsigned 32-bit integer RNG seed" not in str(exc):
                raise
            missing_seed_rejected = True
        if not missing_seed_rejected:
            raise RuntimeError("Service accepted a request with no RNG seed")
        return outputs[0], {
            "same_seed_exact_equal": True, "different_seed_changes_actions": True,
            "missing_seed_rejected": True,
            "action_shape": list(outputs[0].shape),
            "actions_finite": bool(np.isfinite(outputs[0]).all()),
            "actions_sha256": hashlib.sha256(outputs[0].tobytes()).hexdigest(),
            "metadata": client.server_metadata,
        }


def main() -> None:
    parser = _parser()
    parser.description = __doc__
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--readiness-timeout", type=float, default=1200)
    parser.add_argument("--load-report", type=Path)
    parser.add_argument("--verification-report", type=Path)
    args = parser.parse_args()
    require_allocation()
    configure_allocated_egl()
    if args.host != "127.0.0.1" or not 1024 <= args.port <= 65535:
        raise ValueError("Allocation-local service requires 127.0.0.1 and an unprivileged port")
    if args.timeout <= 0 or args.readiness_timeout <= 0:
        raise ValueError("Service timeouts must be positive")
    if args.dry_run or args.prepare_server_provenance:
        raise ValueError("Use evaluation.sim.cli for preflight")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    def terminate(_signal: int, _frame: Any) -> None:
        raise SystemExit("Slurm terminated the allocation-local service")
    signal.signal(signal.SIGTERM, terminate)
    protocol, model, provenance = _resolve(args)
    if args.serve_only:
        if args.load_report is None:
            raise ValueError("--serve-only requires --load-report")
        policy, metadata = restore_policy(args, model, provenance)
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()
        return

    job = os.environ["SLURM_JOB_ID"]
    with socket.socket() as probe:
        probe.bind((args.host, args.port))
    log_root = Path(os.environ["XARM_WORK_ROOT"]) / "logs" / "policy_service" / job
    log_root.mkdir(parents=True, exist_ok=False)
    evaluator_args = [
        "--model-spec", str(args.model_spec), "--protocol", str(args.protocol),
        "--openpi-root", str(args.openpi_root), "--embodied-ai-root", str(args.embodied_ai_root),
        "--host", args.host, "--port", str(args.port), "--timeout", str(args.timeout),
    ]
    if args.output_root:
        evaluator_args.extend(("--output-root", str(args.output_root)))
    if args.resume:
        evaluator_args.append("--resume")
    evidence: list[dict[str, Any]] = []
    reference = None
    observation = verification_observation(protocol) if args.verification_report else None
    for index in range(2 if args.verification_report else 1):
        command = [sys.executable, "-m", "evaluation.sim.service", *evaluator_args, "--serve-only", "--load-report", str(log_root / f"load-{index}.json")]
        with (log_root / f"server-{index}.log").open("x") as server_log:
            process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
            try:
                wait_ready(process, args, provenance)
                if args.verification_report:
                    actions, check = check_requests(args, provenance, observation)
                    if reference is not None and not np.array_equal(reference, actions):
                        raise RuntimeError("Restoring the same model changed actions for the same request")
                    reference = actions
                    evidence.append(check)
                    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv"], check=True)
                else:
                    subprocess.run([sys.executable, "-m", "evaluation.sim.cli", *evaluator_args], check=True)
            finally:
                stop_owned_process(process)
    if args.verification_report:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv"], capture_output=True, text=True, check=True)
        _write_new(args.verification_report, {
            "verified": True, "slurm_job_id": job, "restart_exact_equal": True,
            "requests": evidence, "gpu": gpu.stdout, "egl_render_verified": True,
            "observation_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in observation.items() if isinstance(value, np.ndarray)},
            "service_log_root": str(log_root), "provenance": provenance,
        })


if __name__ == "__main__":
    main()
