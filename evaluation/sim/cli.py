"""Primary CLI for reproducible formal xArm π0.5 MuJoCo evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy_runtime.remote_policy_client import RemotePolicyClient  # noqa: E402
from policy_runtime.remote_policy_client import RemotePolicyConfig  # noqa: E402
from evaluation.sim.config import default_protocol_path  # noqa: E402
from evaluation.sim.config import load_protocol  # noqa: E402
from evaluation.sim.episode_runner import EpisodeRequest  # noqa: E402
from evaluation.sim.episode_runner import run_formal_episode  # noqa: E402
from evaluation.common.models import load_model_spec  # noqa: E402
from evaluation.common.models import validate_training_config_asset  # noqa: E402
from evaluation.sim.outputs import EPISODE_SCHEMA_VERSION  # noqa: E402
from evaluation.sim.outputs import episode_output_root  # noqa: E402
from evaluation.sim.outputs import initialize_output  # noqa: E402
from evaluation.sim.outputs import read_json  # noqa: E402
from evaluation.sim.outputs import validate_episode_result  # noqa: E402
from evaluation.sim.outputs import write_json  # noqa: E402
from evaluation.sim.provenance import build_provenance  # noqa: E402
from evaluation.sim.provenance import server_provenance  # noqa: E402
from evaluation.sim.summary import write_comparison  # noqa: E402
from evaluation.sim.summary import write_model_summary  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=default_protocol_path())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, default=Path(os.environ.get("OPENPI_ROOT", "/u/mw89/repos/openpi")))
    parser.add_argument("--embodied-ai-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-server-provenance",
        type=Path,
        help="Validate model/protocol and write the identity JSON used at server startup; do not evaluate.",
    )
    return parser


def _resolve(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    protocol = load_protocol(args.protocol)
    if args.output_root.expanduser().resolve() != protocol.output_root:
        raise ValueError(
            f"Output root {args.output_root.expanduser().resolve()} differs from the protocol's "
            f"isolated formal output root {protocol.output_root}"
        )
    model = load_model_spec(args.model_spec)
    validate_training_config_asset(model, openpi_root=args.openpi_root)
    provenance = build_provenance(
        protocol=protocol,
        model=model,
        openpi_root=args.openpi_root,
        embodied_ai_root=args.embodied_ai_root,
    )
    return protocol, model, provenance


def _print_resolution(protocol: Any, model: Any, provenance: dict[str, Any], output_root: Path) -> None:
    print(
        json.dumps(
            {
                "model": model.to_json(),
                "protocol": protocol.to_json(),
                "protocol_sha256": provenance["protocol_sha256"],
                "provenance_sha256": provenance["provenance_sha256"],
                "output_root": str(output_root.expanduser().resolve()),
                "episode_count": len(protocol.tasks) * len(protocol.seeds),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _verify_server(client: RemotePolicyClient, provenance: dict[str, Any]) -> None:
    metadata = client.server_metadata
    expected = server_provenance(provenance)
    actual = metadata.get("formal_evaluation_provenance")
    if actual != expected:
        raise RuntimeError(
            "Connected policy server provenance differs from requested formal model/protocol; "
            "refusing to evaluate an unverified checkpoint"
        )
    if metadata.get("request_rng_required") is not True:
        raise RuntimeError("Formal evaluation requires a request-RNG-enforcing policy server")


def main() -> None:
    args = _parser().parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    protocol, model, provenance = _resolve(args)
    if args.prepare_server_provenance is not None:
        target = args.prepare_server_provenance.expanduser().resolve()
        write_json(target, server_provenance(provenance))
        print(f"Wrote formal server provenance: {target}")
        return
    _print_resolution(protocol, model, provenance, args.output_root)
    if args.dry_run:
        return

    model_root = initialize_output(
        output_root=args.output_root,
        model_id=model.model_id,
        provenance=provenance,
        resume=args.resume,
    )
    invalid = 0
    with RemotePolicyClient(
        RemotePolicyConfig(
            host=args.host,
            port=args.port,
            connect_timeout_s=args.timeout,
            inference_timeout_s=args.timeout,
        )
    ) as policy:
        _verify_server(policy, provenance)
        for task in protocol.tasks:
            for seed in protocol.seeds:
                episode_dir = episode_output_root(args.output_root, model.model_id, task.task_id, seed)
                result_path = episode_dir / "result.json"
                if result_path.exists():
                    if not args.resume:
                        raise FileExistsError(f"Episode output already exists: {result_path}")
                    result = read_json(result_path)
                    validate_episode_result(result)
                    if result["schema_version"] != EPISODE_SCHEMA_VERSION:
                        raise ValueError(
                            "Refusing to resume a legacy episode schema into the v2 formal pipeline; "
                            "use the derived reclassification helper for historical results instead."
                        )
                    if result["provenance"] != provenance:
                        raise ValueError(f"Refusing resume with different provenance: {result_path}")
                else:
                    episode_dir.mkdir(parents=True, exist_ok=False)
                    record_video = protocol.video_policy in {"category_representative", "all"} or (
                        protocol.video_policy == "periodic"
                        and (seed - protocol.seed_start) % protocol.periodic_video_every == 0
                    )
                    result = run_formal_episode(
                        EpisodeRequest(
                            protocol=protocol,
                            model=model,
                            provenance=provenance,
                            task=task,
                            seed=seed,
                            output_dir=episode_dir,
                            model_output_dir=model_root,
                            record_video=record_video,
                        ),
                        policy=policy,
                    )
                invalid += int(not result["episode"]["valid"])
                print(
                    f"model={model.model_id} task={task.task_id} seed={seed} "
                    f"valid={result['episode']['valid']} success={result['episode']['success']} "
                    f"reason={result['episode']['termination_reason']}"
                )
    summary = write_model_summary(model_root)
    write_comparison(args.output_root)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    if protocol.fail_on_invalid and invalid:
        raise SystemExit(f"Formal evaluation produced {invalid} invalid episodes")


if __name__ == "__main__":
    main()
