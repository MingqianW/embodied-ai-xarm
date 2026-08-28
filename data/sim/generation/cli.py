"""Stable CLI for generation, conversion, audit, and inspection."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from data.sim.generation.audit import (
    audit_converted,
    audit_raw,
    write_audit_reports,
    write_raw_audit_reports,
    write_smoke_reports,
)
from data.sim.generation.collection import collect
from data.sim.generation.config import load_pipeline_config, repository_root
from data.sim.generation.conversion import convert_dataset
from data.sim.generation.manifest import atomic_write_json
from data.sim.generation.reporting import (
    write_final_handoff,
    write_initial_audit,
)
from data.sim.generation.safety import (
    apply_group_permissions,
    path_inventory,
    replace_authorized_roots,
    validate_authorized_root,
)
from data.sim.generation.status import git_sha, write_status


DEFAULT_CONFIG = Path("configs/data/sim/generation/clean_multitask_stable_v3.yaml")


def _config(args: argparse.Namespace):
    return load_pipeline_config(args.config)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m data.sim.generation.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    generate = subparsers.add_parser("generate")
    add_config(generate)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--smoke", action="store_true")

    convert = subparsers.add_parser("convert")
    add_config(convert)
    convert.add_argument("--raw", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--overwrite", action="store_true")

    audit = subparsers.add_parser("audit")
    add_config(audit)
    audit.add_argument("--raw", type=Path, required=True)
    audit.add_argument("--converted", type=Path)
    audit.add_argument("--report-dir", type=Path, required=True)
    audit.add_argument("--decode-all-images", action="store_true")
    audit.add_argument("--smoke", action="store_true")

    inspect = subparsers.add_parser("inspect")
    add_config(inspect)
    inspect.add_argument("--initialize-log-root", action="store_true")
    inspect.add_argument("--overwrite", action="store_true")

    permissions = subparsers.add_parser("permissions")
    add_config(permissions)
    permissions.add_argument("paths", nargs="+", type=Path)

    status = subparsers.add_parser("status")
    add_config(status)
    status.add_argument("--phase", required=True)
    status.add_argument("--completed-work", action="append", default=[])
    status.add_argument("--job-id", action="append", default=[])
    status.add_argument("--job-state", default="UNKNOWN")
    status.add_argument("--known-failure", action="append", default=[])
    status.add_argument("--next-action", required=True)
    status.add_argument("--resume-command", action="append", default=[])

    handoff = subparsers.add_parser("handoff")
    add_config(handoff)

    args = parser.parse_args()
    config = _config(args)
    if args.command == "generate":
        result = collect(
            config, args.output, overwrite=args.overwrite, resume=args.resume,
            smoke=args.smoke,
        )
    elif args.command == "convert":
        result = convert_dataset(config, args.raw, args.output, overwrite=args.overwrite)
    elif args.command == "audit":
        raw_report = audit_raw(
            config,
            args.raw,
            decode_all_images=args.decode_all_images,
            smoke=args.smoke,
        )
        if args.smoke and args.converted:
            raise ValueError("Smoke audit does not accept a converted dataset")
        converted_report = (
            audit_converted(
                config, args.converted, decode_all_images=args.decode_all_images
            )
            if args.converted
            else None
        )
        result = (
            write_smoke_reports(config, raw_report, args.report_dir)
            if args.smoke
            else (
                write_audit_reports(
                    config, raw_report, converted_report, args.report_dir
                )
                if converted_report is not None
                else write_raw_audit_reports(config, raw_report, args.report_dir)
            )
        )
    elif args.command == "permissions":
        apply_group_permissions(args.paths)
        result = {"paths": [str(validate_authorized_root(path)) for path in args.paths]}
    elif args.command == "status":
        repository = repository_root()
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        existing_path = config.outputs.log / "CODEX_STATUS.json"
        existing = (
            json.loads(existing_path.read_text(encoding="utf-8"))
            if existing_path.is_file()
            else {}
        )
        job_ids = list(
            dict.fromkeys([*(existing.get("submitted_job_ids") or []), *args.job_id])
        )
        job_states = dict(existing.get("job_states_at_last_check") or {})
        job_states.update({job_id: args.job_state for job_id in args.job_id})
        result = {
            "repository": str(repository),
            "branch": branch,
            "commit_sha": git_sha(repository),
            "current_phase": args.phase,
            "completed_work": list(
                dict.fromkeys([*(existing.get("completed_work") or []), *args.completed_work])
            ),
            "submitted_job_ids": job_ids,
            "job_states_at_last_check": job_states,
            "resolved_raw_path": str(config.outputs.raw),
            "resolved_converted_path": str(config.outputs.converted),
            "resolved_smoke_path": str(config.outputs.smoke),
            "resolved_log_path": str(config.outputs.log),
            "known_failures": list(
                dict.fromkeys(
                    [*(existing.get("known_failures") or []), *args.known_failure]
                )
            ),
            "next_required_action": args.next_action,
            "exact_resume_commands": args.resume_command,
        }
        status_id = args.job_id[-1] if args.job_id else "login"
        atomic_write_json(
            config.outputs.log / "status" / f"{args.phase}-{status_id}.json",
            result,
        )
        write_status(config.outputs.log, result)
    elif args.command == "handoff":
        result = {"final_handoff": str(write_final_handoff(config))}
    else:
        if args.initialize_log_root:
            result = replace_authorized_roots(
                [config.outputs.log], overwrite=args.overwrite,
                git_sha=git_sha(repository_root()), config_path=config.path,
            )
            (config.outputs.log / "slurm").mkdir(exist_ok=True)
            write_initial_audit(config)
        else:
            result = {
                "schema_version": config.schema_version,
                "dataset_version": config.dataset_version,
                "config": str(config.path),
                "camera_config": str(config.camera_config),
                "total_episodes": config.total_episodes,
                "tasks": [
                    {"task_id": task.task_id, "prompt": task.prompt, "episodes": task.episodes}
                    for task in config.tasks
                ],
                "outputs": {
                    name: path_inventory(path)
                    for name, path in {
                        "raw": config.outputs.raw,
                        "converted": config.outputs.converted,
                        "smoke": config.outputs.smoke,
                        "log": config.outputs.log,
                    }.items()
                },
            }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
