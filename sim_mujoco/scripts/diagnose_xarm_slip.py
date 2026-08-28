#!/usr/bin/env python3
"""Run one isolated, opt-in xArm slip-diagnostic episode.

This entry point deliberately does not accept a formal evaluation output root.
It preserves the selected protocol's task scoring and RNG salt while allowing
an explicit c1/c2/c5 executed prefix and diagnostic-only post-success trace.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy_runtime.remote_policy_client import RemotePolicyClient  # noqa: E402
from policy_runtime.remote_policy_client import RemotePolicyConfig  # noqa: E402
from evaluation.sim.config import FORMAL_PROTOCOL_VERSION  # noqa: E402
from evaluation.sim.config import (  # noqa: E402
    FORMAL_STABLE_HOLD_PROTOCOL_VERSION,
)
from evaluation.sim.config import load_protocol  # noqa: E402
from evaluation.sim.episode_runner import EpisodeRequest  # noqa: E402
from evaluation.sim.episode_runner import run_formal_episode  # noqa: E402
from evaluation.common.models import load_model_spec  # noqa: E402
from evaluation.common.models import validate_training_config_asset  # noqa: E402
from evaluation.sim.outputs import episode_output_root  # noqa: E402
from evaluation.sim.outputs import initialize_output  # noqa: E402
from evaluation.sim.outputs import write_json  # noqa: E402
from evaluation.sim.provenance import build_provenance  # noqa: E402
from evaluation.sim.provenance import file_hash  # noqa: E402
from evaluation.sim.provenance import server_provenance  # noqa: E402
from evaluation.sim.slip_trace import POST_SUCCESS_SECONDS_ENV  # noqa: E402
from evaluation.sim.slip_trace import DIAGNOSTIC_LATCH_RAW_ENV  # noqa: E402
from evaluation.sim.slip_trace import SLIP_TRACE_ENV  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--execute-chunk-steps", type=int, choices=(1, 2, 5), required=True
    )
    parser.add_argument("--post-success-seconds", type=float, default=2.0)
    parser.add_argument(
        "--gripper-latch-raw",
        type=float,
        help=(
            "Diagnostic-only raw gripper command applied after bilateral target contact. "
            "Omit for the unmodified policy run."
        ),
    )
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=Path(os.environ.get("OPENPI_ROOT", "/u/mw89/repos/openpi")),
    )
    parser.add_argument("--embodied-ai-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video while retaining all physics-cadence diagnostic logs.",
    )
    parser.add_argument(
        "--prepare-server-provenance",
        type=Path,
        help="Write server identity and exit without simulation or inference.",
    )
    parser.add_argument(
        "--friction-variant-manifest",
        type=Path,
        help="Read-only provenance for an already prepared runtime friction model.",
    )
    return parser


def _resolve(args: argparse.Namespace):
    source_protocol_path = args.protocol.expanduser().resolve()
    source = load_protocol(source_protocol_path)
    supported_protocols = {
        FORMAL_PROTOCOL_VERSION,
        FORMAL_STABLE_HOLD_PROTOCOL_VERSION,
    }
    if source.protocol_version not in supported_protocols:
        raise ValueError(
            "Slip diagnosis requires a formal v1 or stable-hold formal v2 protocol"
        )
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not math.isfinite(args.post_success_seconds) or args.post_success_seconds < 0.0:
        raise ValueError("--post-success-seconds must be finite and non-negative")
    if args.gripper_latch_raw is not None and (
        not math.isfinite(args.gripper_latch_raw)
        or not 50.0 <= args.gripper_latch_raw <= 845.0
    ):
        raise ValueError("--gripper-latch-raw must be in [50, 845]")
    selected = tuple(task for task in source.tasks if task.task_id == args.task)
    if len(selected) != 1:
        raise ValueError(f"Task is not in the formal protocol: {args.task!r}")
    output_root = args.output_root.expanduser().resolve()
    if not str(output_root).startswith("/work/nvme/bfmk/mw89/"):
        raise ValueError("Diagnostic output root must be under /work/nvme/bfmk/mw89")
    if output_root == source.output_root or source.output_root in output_root.parents:
        raise ValueError(
            "Diagnostic output must be isolated from the historical formal output root"
        )
    protocol = replace(
        source,
        tasks=selected,
        seed_start=args.seed,
        seed_count=1,
        execute_chunk_steps=args.execute_chunk_steps,
        output_root=output_root,
        video_policy="all",
    )
    model = load_model_spec(args.model_spec)
    validate_training_config_asset(model, openpi_root=args.openpi_root)
    friction_variant = None
    if args.friction_variant_manifest is not None:
        friction_path = args.friction_variant_manifest.expanduser().resolve()
        friction_variant = json.loads(friction_path.read_text(encoding="utf-8"))
        if friction_variant.get("schema_version") != (
            "xarm_runtime_friction_policy_variant_v1"
        ):
            raise ValueError(f"Invalid friction variant manifest: {friction_path}")
        if Path(friction_variant["runtime_model"]).resolve() != (
            protocol.robot_xml_path
        ):
            raise ValueError("Friction manifest runtime model differs from protocol")
    provenance = build_provenance(
        protocol=protocol,
        model=model,
        openpi_root=args.openpi_root,
        embodied_ai_root=args.embodied_ai_root,
    )
    manifest = {
        "diagnostic_type": "xarm_tcp_relative_slip_v1",
        "scientific_result_semantics": (
            "diagnostic_intervention_not_formal_policy_performance"
            if args.gripper_latch_raw is not None
            else (
                "historical_formal_v1_frozen_at_original_success"
                if source.protocol_version == FORMAL_PROTOCOL_VERSION
                else "formal_v2_stable_hold_frozen_at_original_success"
            )
        ),
        "source_protocol_path": str(source_protocol_path),
        "source_protocol_sha256": file_hash(source_protocol_path),
        "model": model.to_json(),
        "task": args.task,
        "seed": args.seed,
        "execute_chunk_steps": args.execute_chunk_steps,
        "control_duration_s": protocol.control_duration_s,
        "post_success_seconds": args.post_success_seconds,
        "rng_salt": protocol.rng_salt,
        "output_root": str(output_root),
        "simulator_ground_truth_exposed_to_policy": False,
        "friction_override": friction_variant,
        "record_video": not args.no_video,
        "diagnostic_gripper_latch_raw": args.gripper_latch_raw,
        "diagnostic_gripper_latch_trigger": (
            None
            if args.gripper_latch_raw is None
            else "five_consecutive_bilateral_target_contact_physics_samples"
        ),
        "provenance_sha256": provenance["provenance_sha256"],
    }
    return protocol, model, provenance, manifest


def _verify_server(client: RemotePolicyClient, provenance: dict[str, object]) -> None:
    metadata = client.server_metadata
    if metadata.get("formal_evaluation_provenance") != server_provenance(provenance):
        raise RuntimeError("Policy server provenance differs from this slip diagnostic")
    if metadata.get("request_rng_required") is not True:
        raise RuntimeError("Slip diagnosis requires request-scoped policy RNG")


def main() -> None:
    args = _parser().parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise ValueError("--timeout must be finite and positive")
    protocol, model, provenance, manifest = _resolve(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.prepare_server_provenance is not None:
        write_json(
            args.prepare_server_provenance.expanduser().resolve(),
            server_provenance(provenance),
        )
        return
    if args.dry_run:
        return

    os.environ[SLIP_TRACE_ENV] = "1"
    os.environ[POST_SUCCESS_SECONDS_ENV] = str(args.post_success_seconds)
    if args.gripper_latch_raw is not None:
        os.environ[DIAGNOSTIC_LATCH_RAW_ENV] = str(args.gripper_latch_raw)
    else:
        os.environ.pop(DIAGNOSTIC_LATCH_RAW_ENV, None)
    if protocol.output_root.exists() and any(protocol.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing a non-empty slip-diagnostic output root: {protocol.output_root}"
        )
    model_root = initialize_output(
        output_root=protocol.output_root,
        model_id=model.model_id,
        provenance=provenance,
        resume=False,
    )
    write_json(protocol.output_root / "diagnostic_manifest.json", manifest)
    episode_dir = episode_output_root(
        protocol.output_root, model.model_id, args.task, args.seed
    )
    episode_dir.mkdir(parents=True, exist_ok=False)
    with RemotePolicyClient(
        RemotePolicyConfig(
            host=args.host,
            port=args.port,
            connect_timeout_s=args.timeout,
            inference_timeout_s=args.timeout,
        )
    ) as policy:
        _verify_server(policy, provenance)
        result = run_formal_episode(
            EpisodeRequest(
                protocol=protocol,
                model=model,
                provenance=provenance,
                task=protocol.tasks[0],
                seed=args.seed,
                output_dir=episode_dir,
                model_output_dir=model_root,
                record_video=not args.no_video,
            ),
            policy=policy,
        )
    trace = episode_dir / "slip_trace.csv"
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(f"Slip trace was not written: {trace}")
    print(
        json.dumps(
            {
                "result": str(episode_dir / "result.json"),
                "slip_trace": str(trace),
                "episode": result["episode"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
