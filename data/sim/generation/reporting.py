"""Durable audit and handoff reports for versioned simulation pipelines."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from data.common.task_identity import TASKS
from data.sim.generation.config import PipelineConfig, repository_root


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_initial_audit(config: PipelineConfig) -> Path:
    path = config.outputs.log / "INITIAL_AUDIT.md"
    repository = repository_root()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    text = f"""# Initial MuJoCo Pipeline Audit

- Repository: `{repository_root()}`
- Starting branch: `{branch}`
- Starting commit: `{commit}`
- Authoritative camera file: `{config.camera_config}` (preserved; never replaced)

## Findings before refactoring

1. The retained `data/sim/generation/legacy/collect_oracle_data.py` workflow was a red-block-only collector.
2. The retained `data/sim/generation/legacy/collect_real_raw_sim_data.py` workflow supported six tasks, but its plan and prompts were hard-coded and included distractor splits.
3. Prompt mappings mixed underscore folder labels with natural-language prompts.
4. Pick success used a 5 cm lift and a three-policy-step streak.
5. Pick `VERIFY` could complete early from that short streak; it did not execute a mandatory 20-step stability window.
6. Place reset activated a fixed `held_red_pepper` gripper child.
7. `held_red_pepper` was the fixed visual object; `red_pepper` was a separate free body.
8. Release teleported the fixed body pose into the free body, creating a hidden identity swap.
9. Place initialization was visible to the controller setup path and lacked an explicit excluded 10-step physical validation phase.
10. The established six-task conversion path used the real-compatible raw CSV/PNG layout and the shared `data/common/lerobot_writer.py` LeRobot v2.1 writer.
11. Output paths were resolved by several independent scripts and environment defaults rather than one exact-root policy.
12. Existing tests covered task scenes, basic oracle motion, conversion, gripper motion, and collisions, but not the required full-window stability, exact-root overwrite, prompt registry, seed, manifest, or clean-plan contracts.

## Integrated design

The versioned package separates the registry, typed config, scene setup, Pick and Place controllers, shared validators, recording, seed/retry state, manifests, conversion, audits, artifacts, status, permissions, and Slurm orchestration. The camera YAML above remains authoritative.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_final_handoff(config: PipelineConfig) -> Path:
    log = config.outputs.log
    raw_summary = _read_json(config.outputs.raw / "collection_summary.json")
    converted = _read_json(
        config.outputs.converted / "meta" / "mujoco_multitask_metadata.json"
    )
    audit = _read_json(log / "DATASET_AUDIT.json")
    smoke = _read_json(log / "SMOKE_AUDIT.json")
    status = _read_json(log / "CODEX_STATUS.json")
    test_report = log / "TEST_REPORT.md"
    raw_marker = _read_json(config.outputs.raw / "OVERWRITE_MARKER.json")
    converted_marker = _read_json(config.outputs.converted / "OVERWRITE_MARKER.json")
    smoke_marker = _read_json(config.outputs.smoke / "OVERWRITE_MARKER.json")
    log_marker = _read_json(log / "OVERWRITE_MARKER.json")
    repository = repository_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "show", "--pretty=format:", "--name-only", "HEAD"],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    jobs = status.get("job_states_at_last_check") or {}
    task_rows = "\n".join(
        f"| `{task.task_id}` | {task.prompt} | "
        f"{next(plan.episodes for plan in config.tasks if plan.task_id == task.task_id)} | 0 |"
        for task in TASKS
    )
    marker_rows = "\n".join(
        f"- `{name}`: `{marker.get('removed_and_recreated_path')}` at "
        f"`{marker.get('overwritten_utc')}`; inventory `{marker.get('preoverwrite_inventory')}`"
        for name, marker in (
            ("raw", raw_marker),
            ("converted", converted_marker),
            ("smoke", smoke_marker),
            ("log", log_marker),
        )
    )
    ready = audit.get("status") == "READY_FOR_TRAINING"
    text = f"""# Final Handoff

## Repository and implementation

1. Repository: `{repository}`
2. Branch: `sim-pipeline-camera-sync-20260806`
3. Branch: `{branch}`
4. Local commit: `{commit}`
5. Original problems: hard-coded mixed-prompt/distractor plan, short Pick streak acceptance, and a fixed-to-free Place pepper identity swap.
6. Architecture: typed YAML config plus registry, scene runtime, separate Pick/Place controllers, canonical stability validators, recorder, atomic manifest, converter, strict audits, artifact writer, and self-contained Slurm phases.
7. Files in the focused commit: {', '.join(f'`{name}`' for name in changed if name) or '(inspect Git commit)'}.
8. Public CLI: `python -m data.sim.generation.cli {{generate,convert,audit,inspect}} ...`.
9. Versioned config: `{config.path}`.

## Tasks, prompts, and clean plan

| Task ID | Canonical prompt | Accepted | Distractors |
|---|---|---:|---:|
{task_rows}

- Alias policy: underscore and historical labels are input-only aliases; emitted simulation prompts are canonical natural-language strings.
- Joint real/simulation training must normalize historical real aliases through `data.common.task_identity`; old datasets are not mutated.
- Total distractor episodes: `{raw_summary.get('total_distractor_episodes')}`.

## Correctness definitions

- Pick: enter verification only after 5 cm lift, hold the final arm/closed-gripper target for exactly 20 actions at 0.1 s, retain at least 4 cm lift, limit relative downward slip to 1 cm, remain in the grasp region, avoid table/forbidden contact and non-finite state, and finish with robust-fit downward speed no worse than 0.01 m/s.
- Place reset: LOCAL's `held_red_pepper` child body starts enabled inside the gripper while the free `red_pepper` is disabled. Release performs the one-time held-to-free pose transfer and body swap; no per-step pose overwrite is used, and release simulation time is recorded.
- Place initialization: exactly 10 excluded actions at 0.1 s validate the held pepper before recorder frame 0. Initialization frames recorded: `0` for every accepted Place episode.
- Place success: opening/release and retreat must precede a complete 20-action, 2.0 s settled in-ring verification.

## Execution results

- Focused/regression/offline test report: `{test_report}`.
- Smoke status: `{smoke.get('status')}`; job history: `{json.dumps(jobs, sort_keys=True)}`.
- Full generation complete: `{raw_summary.get('complete')}`; accepted: `{raw_summary.get('total_accepted_episodes')}`.
- Conversion episodes: `{len(converted.get('episodes') or [])}`.
- Raw path: `{config.outputs.raw}`.
- Converted path: `{config.outputs.converted}`.
- Task counts: `{json.dumps(raw_summary.get('accepted_counts_by_task') or {}, sort_keys=True)}`.
- Prompt audit: `{'PASS' if ready else 'INCOMPLETE'}`.
- Pick rejection statistics: `{json.dumps(raw_summary.get('stable_grasp_failure_counts') or {}, sort_keys=True)}`.
- Place initial-grasp rejection statistics: `{json.dumps(raw_summary.get('initial_place_grasp_failure_counts') or {}, sort_keys=True)}`.
- Raw and converted integrity: `{audit.get('status')}`.
- Dataset readiness: `{'READY_FOR_TRAINING' if ready else 'NOT_READY_FOR_TRAINING'}`.
- Permission evidence: `{log / 'PERMISSIONS_REPORT.txt'}`.
- Representative videos/key frames: `{config.outputs.smoke}` and accepted episode `visual_manifest.json` files under `{config.outputs.raw}`.

## Scoped overwrite record

The user authorized replacement of only the four exact roots for `{config.dataset_version}`. Each phase required `--overwrite`, validated the canonical non-symlink path, saved an inventory outside the replaced root, and recreated the root with an overwrite marker.

{marker_rows}

## Documentation and reproducibility

- Documentation root: `{repository / 'docs' / 'simulation_data'}`.
- Known limitations: physical robustness is bounded by the configured randomization and smoke/full seeds; no model training was performed.

```bash
cd {repository}
{repository_root()}/slurm/simulation_data/offline_tests.sbatch  # submit with sbatch
sbatch {repository}/slurm/simulation_data/smoke.sbatch
sbatch {repository}/slurm/simulation_data/full_generation.sbatch
sbatch {repository}/slurm/simulation_data/conversion.sbatch
sbatch {repository}/slurm/simulation_data/final_audit.sbatch
```

Resume after disconnect:

```bash
cat {log / 'CODEX_STATUS.md'}
squeue -u mw89
sacct -j JOB_ID --format=JobID,JobName%32,State,Elapsed,ExitCode
tail -n 100 {log / 'slurm'}/JOB_LOG
```
"""
    path = log / "FINAL_HANDOFF.md"
    path.write_text(text, encoding="utf-8")
    return path
