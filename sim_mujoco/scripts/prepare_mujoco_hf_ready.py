"""Prepare a local, hashed Hugging Face-ready MuJoCo LeRobot directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.data_collection.episode_recorder import REAL_TRAINING_PROMPT
from sim_mujoco.paths import mujoco_dataset_root, mujoco_output_root


DEFAULT_DATASET = mujoco_dataset_root() / "xarm_mujoco_red_block_lerobot"
DEFAULT_RAW = mujoco_dataset_root() / "xarm_mujoco_red_block_raw"
DEFAULT_OUTPUT = mujoco_output_root() / "hf_ready" / "xarm_mujoco_red_block_v1"
DEFAULT_REPO_ID = "MingqianW/xarm_mujoco_red_block_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _dataset_card(
    *,
    info: dict[str, Any],
    raw_config: dict[str, Any],
    raw_manifest: dict[str, Any],
    mujoco_version: str,
) -> str:
    return f"""# xArm MuJoCo red-block scripted-oracle dataset v1

## Purpose

This is a **simulation-only** LeRobot v2.1 dataset for adapting OpenPI π0.5
to an xArm6 red-block grasping task. It is not real-robot data.

## Dataset facts

- Simulator: MuJoCo {mujoco_version}
- Robot model: xArm6 with simulated parallel gripper
- Task prompt: `{REAL_TRAINING_PROMPT}`
- Successful episodes: {info['total_episodes']}
- Training frames: {info['total_frames']}
- FPS: {info['fps']} (one sample every 0.1 s)
- Failed attempts excluded from training: {len(raw_manifest.get('failed_attempts') or [])}
- State: float32 `[joint1..joint6 radians, gripper_raw]`
- Action: float32 absolute next-interval target with the same 7D ordering
- Cameras: `image` and `wrist_image`, RGB uint8, 480×640 stored;
  existing OpenPI preprocessing resizes to 224×224

## Oracle and randomization

The demonstrations use a deterministic finite-state scripted oracle with
ground-truth object pose, damped-least-squares TCP IK, bounded joint
interpolation, collision checks, finite-value checks, timeouts, and sustained
lift verification. The collection randomizes object XY by
±{raw_config['object_xy_range_m']} m, yaw by
±{raw_config['object_yaw_range_deg']} degrees, and initial joints with
standard deviation {raw_config['joint_noise_rad']} rad.

Only successful `COMPLETE` episodes appear in the LeRobot train split.
Raw failures, when present, remain debug artifacts and are not imitation
targets.

## Intended use

The intended use is π0.5 simulation adaptation, evaluation of the existing
xArm observation/action contract, and controlled real-vs-simulation studies.

## Limitations and domain gap

- Rendered backgrounds, lighting, textures, camera optics, and gripper
  appearance differ substantially from the real dataset.
- Scripted IK trajectories can be smoother and less behaviorally diverse than
  human demonstrations.
- The real raw dataset has no object ground-truth pose, so object-position
  distributions cannot be compared directly.
- The dataset does not prove safe transfer to a physical robot.
- Existing real normalization statistics are not modified by this dataset.

Always review the generated real-vs-sim contact sheets and distribution flags
before training or physical deployment.
"""


def _mixed_plan() -> str:
    return """# Real + simulation preparation plan

No datasets are merged by this repository workflow.

## A. Separate simulation specialist

Train a new simulation-specialist checkpoint from the existing π0.5 base or
from a copy of the real-trained checkpoint. Keep the real dataset, its
normalization assets, and the production checkpoint immutable. Recommended
name: `pi05_xarm_mujoco_red_block_v1`.

This is the safest first experiment because simulation regressions cannot
silently replace real behavior.

## B. Mixed real + simulation fine-tuning

Prefer multi-dataset sampling over physically rewriting either dataset.
Start with a **4:1 real:simulation frame sampling ratio** and ablate 2:1 and
8:1. Both sources retain the exact prompt `pick up the red block`, but task
and source identities should remain traceable in experiment configuration.

Do not recompute and replace the existing real checkpoint normalization
statistics implicitly. First compare:

1. continuing with frozen real normalization statistics;
2. source-specific normalization before sampling;
3. explicitly recomputed mixed statistics saved under a new asset ID.

Changing normalization while continuing a real-trained checkpoint changes
the coordinate system seen by the model and can destabilize training.

Use a low peak learning rate (initial proposal: `1e-5`, with `5e-6` and
`2e-5` ablations), short validation intervals, and separate real/sim success
metrics. Simulation oversampling can cause catastrophic forgetting of real
textures, camera artifacts, compliant grasp behavior, and recovery motions.

Recommended checkpoint names:

- `pi05_xarm_real_then_sim_v1`
- `pi05_xarm_real4_sim1_v1`
- `pi05_xarm_real4_sim1_frozen_real_norm_v1`
"""


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    raw_dir = args.raw_input_dir.resolve()
    output_dir = args.output_dir.resolve()
    info = _read_json(dataset_dir / "meta" / "info.json")
    raw_config = _read_json(raw_dir / "run_config.json")
    raw_manifest = _read_json(raw_dir / "manifest.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Only canonical LeRobot v2.1 input is accepted")
    if int(info.get("fps", -1)) != 10:
        raise ValueError("Expected 10 FPS")
    if int(info.get("total_episodes", -1)) != len(
        raw_manifest.get("completed_episodes") or []
    ):
        raise ValueError("Canonical and raw successful episode counts differ")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"HF-ready output is not empty; pass --overwrite: {output_dir}"
        )
    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        raise FileExistsError(
            f"Stale staging directory requires manual inspection: {staging}"
        )
    staging.mkdir(parents=True)
    _copy_tree(dataset_dir, staging)
    try:
        import mujoco

        mujoco_version = getattr(mujoco, "__version__", "unknown")
    except ModuleNotFoundError:
        mujoco_version = "unknown"
    card = _dataset_card(
        info=info,
        raw_config=raw_config,
        raw_manifest=raw_manifest,
        mujoco_version=mujoco_version,
    )
    (staging / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    (staging / "README.md").write_text(
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "  - robotics\n"
        "tags:\n"
        "  - lerobot\n"
        "  - mujoco\n"
        "  - xarm6\n"
        "  - openpi\n"
        "---\n\n"
        + card,
        encoding="utf-8",
    )
    (staging / "UPLOAD_PLAN.md").write_text(
        f"""# Upload plan

Proposed repository: `{args.repo_id}`
Repository type: `dataset`
Local directory: `{output_dir}`

Dry run:

```bash
python sim_mujoco/scripts/upload_mujoco_dataset_to_hf.py \
  --local-dir '{output_dir}' \
  --repo-id '{args.repo_id}' \
  --private \
  --dry-run
```

No upload occurs unless the user later runs the same script with both
`--upload` and `--yes`, and without `--dry-run`. The uploader never deletes
remote files and refuses an existing repository with a different manifest
identity.
""",
        encoding="utf-8",
    )
    (staging / "MIXED_REAL_SIM_PLAN.md").write_text(
        _mixed_plan(),
        encoding="utf-8",
    )
    file_rows = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            file_rows.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "manifest_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "dataset_identity": {
            "name": "xarm_mujoco_red_block_v1",
            "task": "red_block",
            "prompt": REAL_TRAINING_PROMPT,
            "simulation_only": True,
            "codebase_version": info["codebase_version"],
        },
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "files": file_rows,
        "total_hashed_size_bytes": sum(row["size_bytes"] for row in file_rows),
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if output_dir.exists():
        if not (output_dir / "MANIFEST.json").is_file():
            raise ValueError(
                "Refusing --overwrite because the target is not a prior "
                f"HF-ready output (missing MANIFEST.json): {output_dir}"
            )
        shutil.rmtree(output_dir)
    staging.rename(output_dir)
    result = {
        "output_dir": str(output_dir),
        "repo_id": args.repo_id,
        "total_episodes": manifest["total_episodes"],
        "total_frames": manifest["total_frames"],
        "hashed_files": len(file_rows),
        "total_hashed_size_bytes": manifest["total_hashed_size_bytes"],
        "uploaded": False,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-input-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
