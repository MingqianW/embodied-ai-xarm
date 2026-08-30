"""Convert this repository's raw xArm demonstrations to an OpenPI-friendly dataset.

The converter produces a lightweight LeRobot-style directory that mirrors the
fields OpenPI expects from a LeRobot dataset:

    image, wrist_image, state, actions, task

For compatibility with the canonical training OpenPI adapter, the
JSONL export also includes `prompt` as an alias of `task`.

If `lerobot` is installed, it also writes a real LeRobotDataset using
LeRobotDataset.create(...). The JSONL output is always written so the conversion
can be inspected without extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from data.common.lerobot_writer import (
    write_xarm_lerobot_dataset,
)
from data.common.records import EpisodeRecord, FrameRecord, SourceBackend
from data.common.schema import XARM_STATE_COLUMNS
from data.real.collection.raw_episodes import (
    RawEpisode as Episode,
    discover_raw_episodes as _find_episodes,
    instruction_from_task as _instruction_from_task,
    read_json as _read_json,
    state_from_row as _state_from_row,
)
from data.real.config import get_raw_data_root, resolve_from_repo


STATE_COLUMNS = XARM_STATE_COLUMNS

TCP_COLUMNS = (
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_rx_rad",
    "tcp_ry_rad",
    "tcp_rz_rad",
)

def _copy_image(src: Path, dst: Path, *, overwrite: bool) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    if overwrite or not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst.resolve().as_posix()


def _write_jsonl(path: Path, records: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_metadata(output_dir: Path, episodes: list[Episode], records: list[dict[str, Any]]) -> None:
    metadata = {
        "format": "xarm_openpi_lerobot_light",
        "source_backend": "real",
        "state_columns": STATE_COLUMNS,
        "action_columns": STATE_COLUMNS,
        "image_key": "image",
        "wrist_image_key": "wrist_image",
        "num_episodes": len({record["episode_index"] for record in records}),
        "num_frames": len(records),
        "tasks": sorted({record["raw_task"] for record in records} or {episode.task for episode in episodes}),
        "tcp_columns_available_in_raw": TCP_COLUMNS,
        "notes": ["actions are the next frame state; the final raw row of each episode is dropped"],
    }
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_rgb(path: str) -> Any:
    try:
        import cv2

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except ModuleNotFoundError:
        from PIL import Image
        import numpy as np

        return np.asarray(Image.open(path).convert("RGB"))


def _try_write_hf_dataset(output_dir: Path, records: list[dict[str, Any]]) -> bool:
    try:
        from datasets import Dataset, Features, Image, Sequence, Value  # type: ignore
    except Exception as exc:
        print(f"skip Hugging Face dataset export: {exc}")
        return False

    features = Features(
        {
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "timestamp": Value("float64"),
            "raw_task": Value("string"),
            "task": Value("string"),
            "prompt": Value("string"),
            "image": Image(),
            "wrist_image": Image(),
            "state": Sequence(Value("float32")),
            "actions": Sequence(Value("float32")),
        }
    )
    dataset_records = []
    for record in records:
        dataset_records.append(
            {
                "episode_index": record["episode_index"],
                "frame_index": record["frame_index"],
                "timestamp": record["timestamp"],
                "raw_task": record["raw_task"],
                "task": record["task"],
                "prompt": record["prompt"],
                "image": record["image"],
                "wrist_image": record["wrist_image"],
                "state": record["state"],
                "actions": record["actions"],
            }
        )

    dataset = Dataset.from_list(dataset_records, features=features)
    dataset.save_to_disk(output_dir / "hf_dataset")
    return True


def _default_manifest_path(repo_id: str, output_dir: Path) -> Path:
    hf_home = os.environ.get("HF_LEROBOT_HOME")
    if hf_home:
        return Path(hf_home) / repo_id / "meta" / "xarm_raw_manifest.json"
    return output_dir / "meta" / "xarm_raw_manifest.json"


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"converted_raw_episodes": []}
    data = _read_json(path)
    if not isinstance(data.get("converted_raw_episodes"), list):
        raise ValueError(f"invalid converted_raw_episodes in manifest: {path}")
    return data


def _write_manifest(path: Path, *, converted_raw_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": "xarm_raw_conversion_manifest_v1",
        "source_backend": "real",
        "converted_raw_episodes": sorted(converted_raw_ids),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _path_arg(value: str | None, *, arg_name: str, default: Path | None = None) -> Path | None:
    if value is None:
        return default

    suspicious_tokens = (" --", "\n", "\r", "python ")
    if any(token in value for token in suspicious_tokens):
        raise SystemExit(
            f"{arg_name} looks like it swallowed later shell arguments:\n"
            f"  {value!r}\n"
            "Check shell quoting and line continuations. In WSL/Linux, use a POSIX path like "
            '"/mnt/d/2026 summer project/embodied-ai-xarm/converted/xarm_pi05_light", not a Windows path ending in a backslash.'
        )

    if os.name != "nt" and len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/"):
        drive = value[0].lower()
        rest = value[3:].replace("\\", "/")
        raise SystemExit(
            f"{arg_name} is a Windows-style path in a POSIX shell: {value!r}\n"
            f"Use '/mnt/{drive}/{rest}' under WSL, or a relative path such as 'converted/xarm_pi05_light'."
        )

    return Path(value)


def _try_write_lerobot_dataset(
    records_by_episode: list[EpisodeRecord],
    *,
    repo_id: str,
    robot_type: str,
    fps: int,
    push_to_hub: bool,
    hub_private: bool,
    overwrite: bool,
    append_new: bool,
    image_writer_threads: int,
    image_writer_processes: int,
) -> bool:
    try:
        from data.common.lerobot_writer import default_lerobot_output_path
    except Exception as exc:
        print(f"skip LeRobotDataset export: {exc}")
        return False

    try:
        output_path = default_lerobot_output_path(repo_id)
        result = write_xarm_lerobot_dataset(
            records_by_episode,
            repo_id=repo_id,
            output_path=output_path,
            robot_type=robot_type,
            fps=fps,
            overwrite=overwrite,
            resume=append_new and not overwrite,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
            push_to_hub=push_to_hub,
            hub_private=hub_private,
        )
    except Exception as exc:
        print(f"skip LeRobotDataset export: {exc}")
        return False
    print(f"lerobot dataset: {result['output_path']}")
    return True


def convert(
    raw_root: Path,
    output_dir: Path,
    *,
    repo_id: str,
    robot_type: str,
    fps: int,
    push_to_hub: bool,
    hub_private: bool,
    overwrite: bool,
    append_new: bool,
    manifest_path: Path | None,
    skip_light_image_copy: bool,
    skip_hf_dataset: bool,
    image_writer_threads: int,
    image_writer_processes: int,
) -> None:
    episodes = _find_episodes(raw_root)
    if not episodes:
        raise SystemExit(f"No episodes found under {raw_root}")

    manifest_path = manifest_path or _default_manifest_path(repo_id, output_dir)
    converted_raw_ids: set[str] = set()
    if append_new and not overwrite:
        dataset_dir = manifest_path.parent.parent
        if dataset_dir.exists() and not manifest_path.exists():
            raise SystemExit(
                f"append-new cannot safely continue because an existing dataset has no conversion manifest: {dataset_dir}\n"
                "Run once with --overwrite to rebuild the dataset and create the manifest, then use --append-new "
                "for future incremental updates."
            )
        converted_raw_ids = set(str(raw_id) for raw_id in _read_manifest(manifest_path)["converted_raw_episodes"])
        before = len(episodes)
        episodes = [episode for episode in episodes if episode.raw_id not in converted_raw_ids]
        skipped = before - len(episodes)
        print(f"append-new manifest: {manifest_path}")
        print(f"skip already converted raw episode(s): {skipped}")
        if not episodes:
            print("no new raw episodes to convert")
            return

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_episode_count = len(converted_raw_ids) if append_new and not overwrite else 0
    records: list[dict[str, Any]] = []
    records_by_episode: list[EpisodeRecord] = []
    missing_images: list[str] = []

    for episode_index, episode in enumerate(episodes):
        global_episode_index = existing_episode_count + episode_index
        image_dir = output_dir / "images" / f"episode_{global_episode_index:06d}"
        episode_records: list[dict[str, Any]] = []
        canonical_frames: list[FrameRecord] = []
        for frame_index, row in enumerate(episode.rows[:-1]):
            next_row = episode.rows[frame_index + 1]
            instruction = _instruction_from_task(episode.task)
            base = {
                "episode_index": global_episode_index,
                "frame_index": frame_index,
                "timestamp": float(row["ts"]),
                "raw_id": episode.raw_id,
                "raw_task": episode.task,
                "task": instruction,
                "prompt": instruction,
                "source": "real",
                "state": _state_from_row(row),
                "actions": _state_from_row(next_row),
            }

            try:
                raw_image = episode.raw_dir / row["realsense_0_file"]
                raw_wrist_image = episode.raw_dir / row["realsense_1_file"]
                if skip_light_image_copy:
                    if not raw_image.exists():
                        raise FileNotFoundError(raw_image)
                    if not raw_wrist_image.exists():
                        raise FileNotFoundError(raw_wrist_image)
                    image = raw_image.resolve().as_posix()
                    wrist_image = raw_wrist_image.resolve().as_posix()
                else:
                    image = _copy_image(
                        raw_image,
                        image_dir / "image" / f"{frame_index:06d}.png",
                        overwrite=overwrite,
                    )
                    wrist_image = _copy_image(
                        raw_wrist_image,
                        image_dir / "wrist_image" / f"{frame_index:06d}.png",
                        overwrite=overwrite,
                    )
            except FileNotFoundError as exc:
                missing_images.append(str(exc))
                continue

            record = {**base, "image": image, "wrist_image": wrist_image}
            canonical_frames.append(FrameRecord.from_mapping(record))
            records.append(record)
            episode_records.append(record)
        if episode_records:
            canonical_episode = EpisodeRecord(
                episode_index=global_episode_index,
                source=SourceBackend.REAL,
                frames=tuple(canonical_frames),
                metadata={"raw_id": episode.raw_id, "raw_task": episode.task},
            )
            records_by_episode.append(canonical_episode)

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise SystemExit(f"Missing {len(missing_images)} image files. First missing files:\n{preview}")

    jsonl_path = output_dir / "data" / "train.jsonl"
    append_light_dataset = append_new and not overwrite
    _write_jsonl(jsonl_path, records, append=append_light_dataset)
    metadata_records = _read_jsonl(jsonl_path) if append_light_dataset else records
    _write_metadata(output_dir, episodes, metadata_records)
    hf_written = False if skip_hf_dataset else _try_write_hf_dataset(output_dir, records)
    lerobot_written = _try_write_lerobot_dataset(
        records_by_episode,
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        push_to_hub=push_to_hub,
        hub_private=hub_private,
        overwrite=overwrite,
        append_new=append_new,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    if lerobot_written or not records_by_episode:
        converted_raw_ids.update(episode.raw_id for episode in episodes)
        _write_manifest(manifest_path, converted_raw_ids=converted_raw_ids)
        print(f"conversion manifest: {manifest_path}")

    print(f"converted episodes: {len(episodes)}")
    print(f"converted frames: {len(records)}")
    print(f"jsonl: {output_dir / 'data' / 'train.jsonl'}")
    print(f"metadata: {output_dir / 'meta' / 'info.json'}")
    if hf_written:
        print(f"huggingface dataset: {output_dir / 'hf_dataset'}")
    elif skip_hf_dataset:
        print("skipped Hugging Face dataset export")
    if not lerobot_written:
        print("install lerobot in the OpenPI environment to create the real LeRobotDataset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Override raw data root from xarm_data_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/real/lerobot",
        help="Converted dataset directory.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/xarm_pi05_data",
        help="LeRobot repo id, also used as the local dataset name under HF_LEROBOT_HOME.",
    )
    parser.add_argument("--robot-type", default="xarm6")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-private", action="store_true", help="Create/update a private Hugging Face dataset repo.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append-new",
        action="store_true",
        help="Append only raw episodes not listed in the persistent conversion manifest.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Override the append-new manifest path. Defaults under HF_LEROBOT_HOME/repo-id/meta.",
    )
    parser.add_argument(
        "--skip-light-image-copy",
        action="store_true",
        help="Do not copy raw images into output-dir/images; JSONL records point to raw images directly.",
    )
    parser.add_argument(
        "--skip-hf-dataset",
        action="store_true",
        help="Skip the extra datasets.Dataset save_to_disk export; LeRobotDataset export still runs.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="LeRobot image writer thread count.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=5,
        help="LeRobot image writer process count.",
    )
    args = parser.parse_args()

    if args.overwrite and args.append_new:
        raise SystemExit("--overwrite and --append-new cannot be used together")

    raw_root = _path_arg(args.raw_root, arg_name="--raw-root")
    output_dir = _path_arg(
        args.output_dir,
        arg_name="--output-dir",
        default=Path("datasets/real/lerobot"),
    )
    manifest_path = _path_arg(args.manifest_path, arg_name="--manifest-path")
    if output_dir is None:
        raise SystemExit("--output-dir could not be resolved")
    output_dir = resolve_from_repo(output_dir)
    if manifest_path is not None:
        manifest_path = resolve_from_repo(manifest_path)

    convert(
        get_raw_data_root(raw_root),
        output_dir,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        hub_private=args.hub_private,
        overwrite=args.overwrite,
        append_new=args.append_new,
        manifest_path=manifest_path,
        skip_light_image_copy=args.skip_light_image_copy,
        skip_hf_dataset=args.skip_hf_dataset,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


if __name__ == "__main__":
    main()
