"""Convert successful raw MuJoCo oracle episodes through the real xArm writer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tune.xarm_lerobot_writer import write_xarm_lerobot_dataset
from sim_mujoco.data_collection.lerobot_adapter import (
    CONVERSION_MANIFEST_VERSION,
    RawOracleEpisode,
    discover_successful_episodes,
    load_episode_records,
    read_json,
    validate_temporal_alignment,
)
from sim_mujoco.paths import mujoco_dataset_root


DEFAULT_INPUT = mujoco_dataset_root() / "xarm_mujoco_red_block_raw"
DEFAULT_OUTPUT = mujoco_dataset_root() / "xarm_mujoco_red_block_lerobot"
DEFAULT_REPO_ID = "MingqianW/xarm_mujoco_red_block_v1"
MANIFEST_RELATIVE_PATH = Path("meta") / "mujoco_conversion_manifest.json"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_conversion_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": CONVERSION_MANIFEST_VERSION,
            "converted_source_ids": [],
            "episodes": [],
        }
    value = read_json(path)
    if value.get("schema_version") != CONVERSION_MANIFEST_VERSION:
        raise ValueError(f"Unsupported conversion manifest: {path}")
    if not isinstance(value.get("converted_source_ids"), list):
        raise ValueError(f"Invalid converted_source_ids in {path}")
    if not isinstance(value.get("episodes"), list):
        raise ValueError(f"Invalid episodes in {path}")
    return value


def _select_episodes(
    episodes: list[RawOracleEpisode],
    episode_limit: int | None,
) -> tuple[list[RawOracleEpisode], dict[str, Any]]:
    if episode_limit is None:
        return episodes, {
            "strategy": "all_successful_by_episode_index",
            "episode_limit": None,
        }
    if episode_limit <= 0:
        raise ValueError("--episode-limit must be positive")
    if len(episodes) < episode_limit:
        raise ValueError(
            f"Requested {episode_limit} successful episodes, "
            f"but only {len(episodes)} are available"
        )
    return episodes[:episode_limit], {
        "strategy": "first_successful_by_episode_index",
        "episode_limit": episode_limit,
    }


def _copy_debug_videos(
    episodes,
    *,
    output_dir: Path,
    overwrite: bool,
) -> int:
    copied = 0
    for episode in episodes:
        destination_dir = (
            output_dir
            / "source_videos"
            / f"episode_{episode.episode_index:06d}"
        )
        for filename in ("overview.mp4", "combined.mp4"):
            source = episode.directory / filename
            if not source.is_file():
                continue
            destination = destination_dir / filename
            if destination.exists() and not overwrite:
                if destination.stat().st_size != source.stat().st_size:
                    raise FileExistsError(
                        f"Existing copied video differs: {destination}"
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    return copied


def convert(args: argparse.Namespace) -> dict[str, Any]:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    available_episodes = discover_successful_episodes(input_dir)
    if not available_episodes:
        raise ValueError(f"No successful completed episodes: {input_dir}")
    episodes, source_selection = _select_episodes(
        available_episodes,
        args.episode_limit,
    )

    manifest_path = output_dir / MANIFEST_RELATIVE_PATH
    manifest = _load_conversion_manifest(manifest_path) if args.resume else {
        "schema_version": CONVERSION_MANIFEST_VERSION,
        "converted_source_ids": [],
        "episodes": [],
    }
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(
            f"Cannot resume a missing output directory: {output_dir}"
        )
    if args.resume:
        if manifest.get("repo_id") != args.repo_id:
            raise ValueError(
                "Resume repo ID differs from the original conversion"
            )
        if manifest.get("dataset_name") != args.dataset_name:
            raise ValueError(
                "Resume dataset name differs from the original conversion"
            )
        if Path(str(manifest.get("input_dir", ""))).resolve() != input_dir:
            raise ValueError(
                "Resume input directory differs from the original conversion"
            )
        if manifest.get("source_selection") != source_selection:
            raise ValueError(
                "Resume source episode selection differs from the original "
                "conversion"
            )
    elif output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output is not empty; pass --resume or --overwrite: {output_dir}"
        )

    converted_ids = set(str(value) for value in manifest["converted_source_ids"])
    pending = [
        episode for episode in episodes if episode.source_id not in converted_ids
    ]
    records_by_episode: list[list[dict[str, Any]]] = []
    alignment_reports: list[dict[str, Any]] = []
    for episode in pending:
        records = load_episode_records(episode, validate_images=True)
        alignment = validate_temporal_alignment(records)
        alignment_reports.append(
            {
                "source_episode_index": episode.episode_index,
                **alignment,
            }
        )
        records_by_episode.append(records)

    raw_validation = {
        "input_dir": str(input_dir),
        "available_successful_manifest_episodes": len(available_episodes),
        "selected_successful_manifest_episodes": len(episodes),
        "source_selection": source_selection,
        "already_converted_episodes": len(episodes) - len(pending),
        "pending_episodes": len(pending),
        "pending_frames": sum(len(records) for records in records_by_episode),
        "prompt": "pick up the red block",
        "fps": 10,
        "alignment": alignment_reports,
        "failed_attempt_directories_considered": 0,
    }
    if args.validate_only:
        print(json.dumps(raw_validation, indent=2))
        return {"validate_only": True, **raw_validation}
    if args.resume and not pending:
        copied_videos = (
            _copy_debug_videos(
                episodes,
                output_dir=output_dir,
                overwrite=False,
            )
            if args.copy_videos
            else 0
        )
        print("No new successful raw episodes to convert.")
        return {
            "validate_only": False,
            "output_dir": str(output_dir),
            "written_episodes": 0,
            "copied_debug_videos": copied_videos,
            **raw_validation,
        }

    write_result = write_xarm_lerobot_dataset(
        records_by_episode,
        repo_id=args.repo_id,
        output_path=output_dir,
        robot_type="xarm6",
        fps=10,
        overwrite=args.overwrite,
        resume=args.resume,
        image_writer_threads=args.num_workers,
        image_writer_processes=0,
        push_to_hub=False,
    )
    previous_episode_rows = list(manifest.get("episodes") or [])
    new_episode_rows = [
        {
            "source_episode_index": episode.episode_index,
            "source_relative_path": episode.relative_path,
            "source_id": episode.source_id,
            "source_seed": int(episode.metadata["seed"]),
            "number_of_frames": int(episode.metadata["number_of_samples"]),
            "success": True,
        }
        for episode in pending
    ]
    manifest = {
        "schema_version": CONVERSION_MANIFEST_VERSION,
        "dataset_name": args.dataset_name,
        "repo_id": args.repo_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "fps": 10,
        "task": "red_block",
        "prompt": "pick up the red block",
        "success_only": True,
        "source_selection": source_selection,
        "available_successful_manifest_episodes": len(available_episodes),
        "converted_source_ids": [
            row["source_id"]
            for row in [*previous_episode_rows, *new_episode_rows]
        ],
        "episodes": [*previous_episode_rows, *new_episode_rows],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "shared_writer": "fine_tune/xarm_lerobot_writer.py",
        "canonical_write_result": write_result,
    }
    _write_json(manifest_path, manifest)
    copied_videos = (
        _copy_debug_videos(
            episodes,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
        if args.copy_videos
        else 0
    )
    result = {
        "validate_only": False,
        **raw_validation,
        **write_result,
        "conversion_manifest": str(manifest_path),
        "copied_debug_videos": copied_videos,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--dataset-name",
        default="xarm_mujoco_red_block_v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--copy-videos", action="store_true")
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help=(
            "Convert only the first N successful episodes in deterministic "
            "episode-index order."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
