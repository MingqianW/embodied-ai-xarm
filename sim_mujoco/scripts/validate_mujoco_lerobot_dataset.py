"""Strict schema, content, loader, and temporal validation for MuJoCo LeRobot data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sim.generation.legacy.episode_recorder import REAL_TRAINING_PROMPT
from data.sim.generation.legacy.lerobot_adapter import (
    RawOracleEpisode,
    discover_successful_episodes,
    load_episode_records,
    read_json,
    validate_temporal_alignment,
)
from sim_mujoco.paths import mujoco_dataset_root, mujoco_output_root


DEFAULT_DATASET = mujoco_dataset_root() / "xarm_mujoco_red_block_lerobot"
DEFAULT_RAW_INPUT = mujoco_dataset_root() / "xarm_mujoco_red_block_raw"
DEFAULT_OUTPUT = mujoco_output_root() / "dataset_validation"
DEFAULT_OPENPI_ASSETS = mujoco_output_root() / "openpi_smoke_assets"
DEFAULT_REAL_SCHEMA = mujoco_output_root() / "sim_data_pipeline_audit" / "current_real_schema.json"
DEFAULT_COMPARISON_CSV = mujoco_output_root() / "real_sim_comparison" / "distribution_comparison.csv"
EXPECTED_FEATURES = {
    "image": {
        "dtype": "image",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
    },
    "state": {"dtype": "float32", "shape": [7], "names": ["state"]},
    "actions": {
        "dtype": "float32",
        "shape": [7],
        "names": ["actions"],
    },
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


class Validation:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def run(
        self,
        name: str,
        function: Callable[[], Any],
        *,
        warning: bool = False,
    ) -> Any | None:
        try:
            result = function()
        except Exception as exc:
            self.checks.append(
                Check(name, "WARN" if warning else "FAIL", f"{type(exc).__name__}: {exc}")
            )
            return None
        detail = "passed" if result is None else str(result)
        self.checks.append(Check(name, "PASS", detail))
        return result

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def passed(self) -> bool:
        return not any(check.status == "FAIL" for check in self.checks)


def _jsonlines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object at {path}:{line_number}")
            rows.append(value)
    return rows


def _normalize_features(features: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, feature in features.items():
        normalized[key] = {
            "dtype": feature.get("dtype"),
            "shape": list(feature.get("shape") or []),
            "names": feature.get("names"),
        }
    return normalized


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value: Any) -> int | float:
    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar, got {array.shape}")
    return array.reshape(-1)[0].item()


def _decode_loaded_rgb(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.shape != (3, 480, 640):
        raise ValueError(f"Loaded image must be CHW (3,480,640), got {array.shape}")
    if array.dtype != np.float32:
        raise ValueError(f"Loaded image must be float32, got {array.dtype}")
    if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("Loaded image has invalid values")
    return np.rint(np.transpose(array, (1, 2, 0)) * 255.0).astype(np.uint8)


def _load_lerobot(repo_id: str, dataset_dir: Path, *, action_horizon: int | None = None):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "root": dataset_dir,
        "download_videos": False,
    }
    if action_horizon is not None:
        kwargs["delta_timestamps"] = {
            "actions": [index / 10.0 for index in range(action_horizon)]
        }
    return LeRobotDataset(**kwargs)


def _validate_required_files(dataset_dir: Path, info: dict[str, Any]) -> str:
    required = (
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/mujoco_conversion_manifest.json",
    )
    missing = [relative for relative in required if not (dataset_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    episode_count = int(info["total_episodes"])
    parquet = sorted((dataset_dir / "data").rglob("episode_*.parquet"))
    if len(parquet) != episode_count:
        raise ValueError(
            f"Expected {episode_count} parquet files, found {len(parquet)}"
        )
    return f"{len(required)} metadata files, {len(parquet)} parquet files"


def _validate_info(info: dict[str, Any]) -> str:
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"codebase_version={info.get('codebase_version')!r}")
    if info.get("robot_type") != "xarm6":
        raise ValueError(f"robot_type={info.get('robot_type')!r}")
    if int(info.get("fps", -1)) != 10:
        raise ValueError(f"fps={info.get('fps')!r}")
    if info.get("data_path") != (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    ):
        raise ValueError(f"Unexpected data_path: {info.get('data_path')!r}")
    expected_split = {"train": f"0:{int(info['total_episodes'])}"}
    if info.get("splits") != expected_split:
        raise ValueError(
            f"splits={info.get('splits')!r}, expected={expected_split!r}"
        )
    if int(info.get("chunks_size", -1)) != 1000:
        raise ValueError(f"chunks_size={info.get('chunks_size')!r}")
    if int(info.get("total_tasks", -1)) != 1:
        raise ValueError(f"total_tasks={info.get('total_tasks')!r}")
    return (
        f"v2.1, fps=10, episodes={info['total_episodes']}, "
        f"frames={info['total_frames']}"
    )


def _validate_metadata_indices(dataset_dir: Path, info: dict[str, Any]) -> str:
    tasks = _jsonlines(dataset_dir / "meta" / "tasks.jsonl")
    if tasks != [{"task_index": 0, "task": REAL_TRAINING_PROMPT}]:
        raise ValueError(f"Unexpected task mapping: {tasks}")
    episodes = _jsonlines(dataset_dir / "meta" / "episodes.jsonl")
    expected_indices = list(range(int(info["total_episodes"])))
    actual_indices = [int(row["episode_index"]) for row in episodes]
    if actual_indices != expected_indices:
        raise ValueError(f"Non-contiguous episode metadata: {actual_indices}")
    if any(row.get("tasks") != [REAL_TRAINING_PROMPT] for row in episodes):
        raise ValueError("Episode prompt list mismatch")
    if sum(int(row["length"]) for row in episodes) != int(info["total_frames"]):
        raise ValueError("Episode lengths do not sum to total_frames")
    stats = _jsonlines(dataset_dir / "meta" / "episodes_stats.jsonl")
    if [int(row["episode_index"]) for row in stats] != expected_indices:
        raise ValueError("Episode stats indices are not contiguous")
    return f"{len(episodes)} contiguous episodes and one exact task"


def _validate_success_manifest(
    dataset_dir: Path,
    raw_input_dir: Path,
    info: dict[str, Any],
) -> str:
    conversion = read_json(
        dataset_dir / "meta" / "mujoco_conversion_manifest.json"
    )
    rows = conversion.get("episodes")
    if conversion.get("success_only") is not True or not isinstance(rows, list):
        raise ValueError("Conversion manifest is not success-only")
    if len(rows) != int(info["total_episodes"]):
        raise ValueError("Conversion manifest episode count mismatch")
    if any(row.get("success") is not True for row in rows):
        raise ValueError("Failed source episode entered conversion manifest")
    raw_episodes = _selected_raw_episodes(
        dataset_dir,
        raw_input_dir,
    )
    raw_ids = {episode.source_id for episode in raw_episodes}
    converted_ids = {str(row["source_id"]) for row in rows}
    if converted_ids != raw_ids:
        raise ValueError(
            "Converted source IDs differ from successful raw manifest IDs"
        )
    return (
        f"{len(rows)} successful source episodes; "
        "failed_attempts excluded"
    )


def _selected_raw_episodes(
    dataset_dir: Path,
    raw_input_dir: Path,
) -> list[RawOracleEpisode]:
    conversion = read_json(
        dataset_dir / "meta" / "mujoco_conversion_manifest.json"
    )
    selection = conversion.get("source_selection")
    available = discover_successful_episodes(raw_input_dir)
    if selection is None:
        return available
    if not isinstance(selection, dict):
        raise ValueError("Conversion source_selection must be an object")
    strategy = selection.get("strategy")
    episode_limit = selection.get("episode_limit")
    if strategy == "all_successful_by_episode_index":
        if episode_limit is not None:
            raise ValueError("All-episode selection must have a null limit")
        return available
    if strategy != "first_successful_by_episode_index":
        raise ValueError(f"Unsupported source selection strategy: {strategy!r}")
    if not isinstance(episode_limit, int) or episode_limit <= 0:
        raise ValueError(f"Invalid source episode limit: {episode_limit!r}")
    if len(available) < episode_limit:
        raise ValueError(
            f"Selection requires {episode_limit} episodes, "
            f"but only {len(available)} are available"
        )
    return available[:episode_limit]


def _validate_loaded_content(
    dataset: Any,
    *,
    dataset_dir: Path,
    raw_input_dir: Path,
    real_schema: dict[str, Any],
) -> dict[str, Any]:
    raw_episodes = _selected_raw_episodes(
        dataset_dir,
        raw_input_dir,
    )
    raw_records = [
        load_episode_records(episode, validate_images=False)
        for episode in raw_episodes
    ]
    expected_total = sum(len(records) for records in raw_records)
    if len(dataset) != expected_total:
        raise ValueError(
            f"Loader length {len(dataset)} != raw frames {expected_total}"
        )
    flat_states: list[np.ndarray] = []
    flat_actions: list[np.ndarray] = []
    global_index = 0
    for episode_index, records in enumerate(raw_records):
        loaded_records: list[dict[str, Any]] = []
        for frame_index, raw in enumerate(records):
            item = dataset[global_index]
            if int(_scalar(item["index"])) != global_index:
                raise ValueError(f"Global index mismatch at {global_index}")
            if int(_scalar(item["episode_index"])) != episode_index:
                raise ValueError(f"Episode index mismatch at {global_index}")
            if int(_scalar(item["frame_index"])) != frame_index:
                raise ValueError(f"Frame index mismatch at {global_index}")
            if int(_scalar(item["task_index"])) != 0:
                raise ValueError(f"Task index mismatch at {global_index}")
            if str(item["task"]) != REAL_TRAINING_PROMPT:
                raise ValueError(f"Prompt mismatch at {global_index}")
            timestamp = float(_scalar(item["timestamp"]))
            if abs(timestamp - frame_index / 10.0) > 1e-4:
                raise ValueError(f"Timestamp mismatch at {global_index}")
            state = _to_numpy(item["state"]).astype(np.float32, copy=False)
            actions = _to_numpy(item["actions"]).astype(np.float32, copy=False)
            if state.shape != (7,) or actions.shape != (7,):
                raise ValueError(
                    f"State/action shape mismatch at {global_index}"
                )
            if not np.isfinite(state).all() or not np.isfinite(actions).all():
                raise ValueError(f"NaN/Inf at {global_index}")
            np.testing.assert_array_equal(state, raw["state"])
            np.testing.assert_array_equal(actions, raw["actions"])
            # Compare first/middle/last images of each episode pixel-for-pixel.
            if frame_index in {0, len(records) // 2, len(records) - 1}:
                loaded_base = _decode_loaded_rgb(item["image"])
                loaded_wrist = _decode_loaded_rgb(item["wrist_image"])
                with Image.open(raw["image"]) as image:
                    raw_base = np.asarray(image.convert("RGB"))
                with Image.open(raw["wrist_image"]) as image:
                    raw_wrist = np.asarray(image.convert("RGB"))
                np.testing.assert_array_equal(loaded_base, raw_base)
                np.testing.assert_array_equal(loaded_wrist, raw_wrist)
            loaded_records.append({"state": state, "actions": actions})
            flat_states.append(state)
            flat_actions.append(actions)
            global_index += 1
        validate_temporal_alignment(loaded_records)
    states = np.asarray(flat_states, dtype=np.float64)
    actions = np.asarray(flat_actions, dtype=np.float64)
    observed = real_schema["state_action_contract"]["gripper_observed_real_range"]
    safety = real_schema["state_action_contract"]["runtime_safety_range"]
    if float(states[:, 6].min()) < float(safety[0]) or float(states[:, 6].max()) > float(safety[1]):
        raise ValueError("State gripper outside runtime safety range")
    if float(actions[:, 6].min()) < float(safety[0]) or float(actions[:, 6].max()) > float(safety[1]):
        raise ValueError("Action gripper outside runtime safety range")
    real_min = np.asarray(
        real_schema["observed_real_state_distribution"]["minimum"],
        dtype=np.float64,
    )
    real_max = np.asarray(
        real_schema["observed_real_state_distribution"]["maximum"],
        dtype=np.float64,
    )
    plausible_min = real_min - np.asarray([0.1] * 6 + [50.0])
    plausible_max = real_max + np.asarray([0.1] * 6 + [50.0])
    if np.any(states.min(axis=0) < plausible_min) or np.any(states.max(axis=0) > plausible_max):
        raise ValueError(
            "Simulation state exceeds real observed range plus explicit "
            "0.1 rad / 50 raw plausibility margin"
        )
    if np.any(actions.min(axis=0) < plausible_min) or np.any(actions.max(axis=0) > plausible_max):
        raise ValueError(
            "Simulation action exceeds real observed range plus explicit "
            "0.1 rad / 50 raw plausibility margin"
        )
    return {
        "frames": global_index,
        "episodes": len(raw_records),
        "state_min": states.min(axis=0).tolist(),
        "state_max": states.max(axis=0).tolist(),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "real_observed_gripper_range": observed,
        "pixel_exact_rgb_samples": len(raw_records) * 3 * 2,
    }


def _validate_action_chunk_loader(dataset: Any) -> str:
    if len(dataset) < 1:
        raise ValueError("Empty action-chunk dataset")
    item = dataset[0]
    actions = _to_numpy(item["actions"])
    padding = _to_numpy(item["actions_is_pad"])
    if actions.shape != (10, 7):
        raise ValueError(f"Action chunk shape is {actions.shape}, not (10,7)")
    if padding.shape != (10,):
        raise ValueError(f"Action padding shape is {padding.shape}, not (10,)")
    return "same LeRobot loader produced actions=(10,7), padding=(10,)"


def _run_openpi_batch_smoke(
    *,
    python: Path,
    dataset_dir: Path,
    repo_id: str,
    assets_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        str(python),
        str(PROJECT_ROOT / "fine_tune" / "smoke_test_openpi_xarm_dataset.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--repo-id",
        repo_id,
        "--assets-dir",
        str(assets_dir),
        "--output-json",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"OpenPI smoke failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    result = read_json(output_path)
    if result.get("passed") is not True:
        raise ValueError(f"OpenPI smoke did not pass: {result}")
    return result


def validate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    raw_input_dir = args.raw_input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = Validation()

    info = validation.run(
        "required info.json parses",
        lambda: read_json(dataset_dir / "meta" / "info.json"),
    )
    if info is None:
        raise SystemExit("Dataset info.json is unavailable; see report")
    validation.run(
        "required files and one parquet per episode",
        lambda: _validate_required_files(dataset_dir, info),
    )
    validation.run("canonical v2.1 info and split", lambda: _validate_info(info))
    actual_features = _normalize_features(info.get("features") or {})
    validation.run(
        "feature names, shapes, and dtypes equal real writer contract",
        lambda: (
            "exact schema equality"
            if actual_features == EXPECTED_FEATURES
            else (_ for _ in ()).throw(
                ValueError(
                    f"actual={actual_features!r}, expected={EXPECTED_FEATURES!r}"
                )
            )
        ),
    )
    validation.run(
        "task and episode metadata indices",
        lambda: _validate_metadata_indices(dataset_dir, info),
    )
    validation.run(
        "success-only source filtering",
        lambda: _validate_success_manifest(dataset_dir, raw_input_dir, info),
    )

    dataset = validation.run(
        "same LeRobot loader used by OpenPI can load dataset",
        lambda: _load_lerobot(args.repo_id, dataset_dir),
    )
    real_schema = read_json(args.real_schema.resolve())
    content_report = None
    if dataset is not None:
        content_report = validation.run(
            "all frames, RGB pixels, finite values, ranges, and alignment",
            lambda: _validate_loaded_content(
                dataset,
                dataset_dir=dataset_dir,
                raw_input_dir=raw_input_dir,
                real_schema=real_schema,
            ),
        )
        chunk_dataset = validation.run(
            "LeRobot action-horizon loader initializes",
            lambda: _load_lerobot(
                args.repo_id,
                dataset_dir,
                action_horizon=10,
            ),
        )
        if chunk_dataset is not None:
            validation.run(
                "LeRobot action horizon and boundary padding",
                lambda: _validate_action_chunk_loader(chunk_dataset),
            )

    openpi_result = None
    if args.skip_openpi_batch:
        validation.add(
            "existing OpenPI pipeline produces one transformed batch",
            "WARN",
            "explicitly skipped by --skip-openpi-batch",
        )
    else:
        openpi_result = validation.run(
            "existing OpenPI pipeline produces one transformed batch",
            lambda: _run_openpi_batch_smoke(
                # Keep the venv entry point itself. Resolving this symlink
                # selects the base interpreter and loses venv site-packages.
                python=args.python.absolute(),
                dataset_dir=dataset_dir,
                repo_id=args.repo_id,
                assets_dir=args.openpi_assets_dir.resolve(),
                output_path=output_dir / "openpi_batch_smoke.json",
            ),
        )

    comparison_copy = output_dir / "distribution_comparison.csv"
    if args.comparison_csv.is_file():
        shutil.copy2(args.comparison_csv, comparison_copy)
        validation.add(
            "real/sim distribution comparison attached",
            "PASS",
            str(comparison_copy),
        )
    else:
        validation.add(
            "real/sim distribution comparison attached",
            "WARN",
            f"not found: {args.comparison_csv}",
        )

    schema_comparison = {
        "real_schema_path": str(args.real_schema.resolve()),
        "real_hub_schema_directly_verified": bool(
            real_schema["dataset_identity"]["hub_metadata_verified"]
        ),
        "expected_features": EXPECTED_FEATURES,
        "actual_features": actual_features,
        "local_schema_equal": actual_features == EXPECTED_FEATURES,
        "checks": [asdict(check) for check in validation.checks],
    }
    (output_dir / "schema_comparison.json").write_text(
        json.dumps(schema_comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "passed": validation.passed,
        "dataset_dir": str(dataset_dir),
        "repo_id": args.repo_id,
        "content": content_report,
        "openpi_batch": openpi_result,
        "checks": [asdict(check) for check in validation.checks],
    }
    report_lines = [
        "# MuJoCo LeRobot dataset validation",
        "",
        f"Overall result: **{'PASS' if validation.passed else 'FAIL'}**",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in validation.checks:
        detail = check.detail.replace("|", "\\|").replace("\n", " ")
        report_lines.append(f"| {check.name} | {check.status} | {detail} |")
    report_lines.extend(
        [
            "",
            "Remote named real-dataset metadata equality remains separate from "
            "the locally reconstructed writer contract unless authenticated Hub "
            "metadata is available.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_result.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not validation.passed:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-input-dir", type=Path, default=DEFAULT_RAW_INPUT)
    parser.add_argument(
        "--repo-id",
        default="MingqianW/xarm_mujoco_red_block_v1",
    )
    parser.add_argument("--real-schema", type=Path, default=DEFAULT_REAL_SCHEMA)
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=DEFAULT_COMPARISON_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--openpi-assets-dir",
        type=Path,
        default=DEFAULT_OPENPI_ASSETS,
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter containing OpenPI and pinned LeRobot.",
    )
    parser.add_argument("--skip-openpi-batch", action="store_true")
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
