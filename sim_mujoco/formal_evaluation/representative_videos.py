"""Category-aware retention of formal-evaluation video bundles.

The runner records into an episode-local temporary directory. Only after the
automated result is safely written does this module retain a bundle according
to its observed category or remove the temporary bundle.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

from sim_mujoco.formal_evaluation.outputs import read_json
from sim_mujoco.formal_evaluation.outputs import write_json

REPRESENTATIVE_INDEX_VERSION = "xarm-representative-video-index-v1"
SELECTION_POLICY = "lowest_seed_per_model_task_category"
VIDEO_POLICIES = ("category_representative", "all", "periodic")


def outcome_category(result: dict[str, Any]) -> str:
    episode = result["episode"]
    if bool(episode["success"]):
        return "SUCCESS"
    if not bool(episode["valid"]):
        return "INVALID"
    return str(episode.get("failure_category") or "UNCLASSIFIED_LEGACY")


def representative_root(model_root: Path) -> Path:
    return Path(model_root) / "representative_videos"


def index_json_path(model_root: Path) -> Path:
    return Path(model_root) / "representative_video_index.json"


def index_csv_path(model_root: Path) -> Path:
    return Path(model_root) / "representative_video_index.csv"


def _empty_index(*, model_id: str) -> dict[str, Any]:
    return {
        "schema_version": REPRESENTATIVE_INDEX_VERSION,
        "model": model_id,
        "selection_policy": SELECTION_POLICY,
        "records": [],
    }


def load_representative_index(*, model_root: Path, model_id: str) -> dict[str, Any]:
    path = index_json_path(model_root)
    if not path.is_file():
        return _empty_index(model_id=model_id)
    index = read_json(path)
    if index.get("schema_version") != REPRESENTATIVE_INDEX_VERSION:
        raise ValueError(f"Unsupported representative video index: {path}")
    if index.get("model") != model_id or index.get("selection_policy") != SELECTION_POLICY:
        raise ValueError(f"Representative video index identity differs from requested model: {path}")
    if not isinstance(index.get("records"), list):
        raise ValueError(f"Representative video index records are invalid: {path}")
    keys = [(str(record.get("task")), str(record.get("category"))) for record in index["records"]]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Representative video index has duplicate task/category records: {path}")
    return index


def _write_index(*, model_root: Path, index: dict[str, Any]) -> None:
    records = sorted(index["records"], key=lambda row: (row["task"], row["category"], row["seed"]))
    document = {**index, "records": records}
    write_json(index_json_path(model_root), document)
    fields = (
        "model",
        "task",
        "category",
        "seed",
        "video_path",
        "video_bundle_path",
        "result_json_path",
        "success",
        "valid",
        "failure_category",
        "selection_policy",
    )
    target = index_csv_path(model_root)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
    temporary.replace(target)


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return str(record["model"]), str(record["task"]), str(record["category"])


def _bundle_exists(record: dict[str, Any]) -> bool:
    bundle = Path(str(record["video_bundle_path"]))
    video = Path(str(record["video_path"]))
    return bundle.is_dir() and video.exists()


def _move_bundle(*, temporary_dir: Path, destination: Path) -> None:
    source = Path(temporary_dir).resolve()
    target = Path(destination).resolve()
    if not source.is_dir() or not any(source.iterdir()):
        raise FileNotFoundError(f"Temporary video bundle is absent or empty: {source}")
    if target.exists():
        raise FileExistsError(f"Representative destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.replace(target)
    except OSError:
        shutil.move(str(source), str(target))
    if not target.is_dir() or not any(target.iterdir()):
        raise RuntimeError(f"Video bundle move did not finalize: {target}")


def _remove_temporary(temporary_dir: Path) -> None:
    path = Path(temporary_dir).resolve()
    if path.name != "temporary_video":
        raise ValueError(f"Refusing to remove a non-temporary path: {path}")
    if path.exists():
        shutil.rmtree(path)


def _remove_representative_bundle(path: Path) -> None:
    bundle = Path(path).resolve()
    if bundle.name.startswith("seed_") and "representative_videos" in bundle.parts and bundle.is_dir():
        shutil.rmtree(bundle)
        return
    raise ValueError(f"Refusing to remove a non-representative video bundle: {bundle}")


def _rebase_video_metadata(*, metadata: dict[str, Any], source_root: Path, destination_root: Path) -> dict[str, Any]:
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()

    def rebase(value: object) -> object:
        if not isinstance(value, str):
            return value
        try:
            return str(destination / Path(value).resolve().relative_to(source))
        except ValueError:
            return value

    updated = json.loads(json.dumps(metadata))
    paths = updated.get("video_paths")
    if isinstance(paths, dict):
        updated["video_paths"] = {key: rebase(value) for key, value in paths.items()}
    if "combined_video_path" in updated:
        updated["combined_video_path"] = rebase(updated["combined_video_path"])
    return updated


def _record_from_result(
    *,
    result: dict[str, Any],
    result_json_path: Path,
    bundle_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    episode, model = result["episode"], result["model"]
    combined = Path(str(metadata["combined_video_path"]))
    return {
        "model": str(model["model_id"]),
        "task": str(episode["task"]),
        "category": outcome_category(result),
        "seed": int(episode["seed"]),
        "video_path": str(combined),
        "video_bundle_path": str(bundle_path),
        "result_json_path": str(Path(result_json_path).resolve()),
        "success": bool(episode["success"]),
        "valid": bool(episode["valid"]),
        "failure_category": episode.get("failure_category"),
        "selection_policy": SELECTION_POLICY,
    }


def _mark_replaced_result(previous: dict[str, Any], replacement: dict[str, Any]) -> bool:
    path = Path(str(previous["result_json_path"]))
    if not path.is_file():
        return False
    try:
        result = read_json(path)
    except ValueError:
        return False
    artifacts = dict(result.get("artifacts") or {})
    artifacts["video_retention"] = {
        "status": "superseded_by_lower_seed_representative",
        "category": previous["category"],
        "replacement_seed": replacement["seed"],
        "selection_policy": SELECTION_POLICY,
    }
    artifacts.pop("representative_video", None)
    result["artifacts"] = artifacts
    write_json(path, result)
    return True


def retain_video_bundle(
    *,
    model_root: Path,
    result_json_path: Path,
    result: dict[str, Any],
    temporary_video_dir: Path,
    temporary_metadata: dict[str, Any],
    video_policy: str,
) -> dict[str, Any]:
    """Retain or remove a temporary video bundle after result.json is written."""

    if video_policy not in VIDEO_POLICIES:
        raise ValueError(f"Unsupported formal video policy: {video_policy}")
    model_root = Path(model_root).resolve()
    temporary_video_dir = Path(temporary_video_dir).resolve()
    category = outcome_category(result)
    if video_policy == "all":
        destination = Path(result_json_path).parent / "videos"
        _move_bundle(temporary_dir=temporary_video_dir, destination=destination)
        metadata = _rebase_video_metadata(
            metadata=temporary_metadata, source_root=temporary_video_dir, destination_root=destination
        )
        return {
            **metadata,
            "video_retention": {
                "status": "retained_all_episode_video",
                "category": category,
                "video_policy": video_policy,
            },
        }

    if video_policy == "periodic":
        destination = Path(result_json_path).parent / "videos"
        _move_bundle(temporary_dir=temporary_video_dir, destination=destination)
        metadata = _rebase_video_metadata(
            metadata=temporary_metadata, source_root=temporary_video_dir, destination_root=destination
        )
        return {
            **metadata,
            "video_retention": {
                "status": "retained_periodic_episode_video",
                "category": category,
                "video_policy": video_policy,
            },
        }

    model_id = str(result["model"]["model_id"])
    candidate_destination = (
        representative_root(model_root)
        / str(result["episode"]["task"])
        / category
        / f"seed_{int(result['episode']['seed'])}"
    )
    candidate_metadata = _rebase_video_metadata(
        metadata=temporary_metadata,
        source_root=temporary_video_dir,
        destination_root=candidate_destination,
    )
    candidate = _record_from_result(
        result=result,
        result_json_path=result_json_path,
        bundle_path=candidate_destination,
        metadata=candidate_metadata,
    )
    index = load_representative_index(model_root=model_root, model_id=model_id)
    key = _record_key(candidate)
    records = list(index["records"])
    previous = next((record for record in records if _record_key(record) == key), None)
    previous_valid = previous is not None and _bundle_exists(previous)
    if previous_valid and int(previous["seed"]) <= candidate["seed"]:
        _remove_temporary(temporary_video_dir)
        return {
            "video_retention": {
                "status": "discarded_after_classification",
                "category": category,
                "video_policy": video_policy,
                "selection_policy": SELECTION_POLICY,
                "representative_seed": int(previous["seed"]),
            }
        }

    _move_bundle(temporary_dir=temporary_video_dir, destination=candidate_destination)
    records = [record for record in records if _record_key(record) != key]
    records.append(candidate)
    index["records"] = records
    _write_index(model_root=model_root, index=index)

    if (
        previous_valid
        and Path(str(previous["video_bundle_path"])) != candidate_destination
        and _mark_replaced_result(previous, candidate)
    ):
        _remove_representative_bundle(Path(str(previous["video_bundle_path"])))
    return {
        **candidate_metadata,
        "representative_video": candidate,
        "video_retention": {
            "status": "preserved_as_representative",
            "category": category,
            "video_policy": video_policy,
            "selection_policy": SELECTION_POLICY,
        },
    }


def unrecorded_video_artifacts(*, result: dict[str, Any], video_policy: str) -> dict[str, Any]:
    return {
        "video_retention": {
            "status": "not_recorded_periodic_policy",
            "category": outcome_category(result),
            "video_policy": video_policy,
        }
    }


def validate_category_video_coverage(evaluation_root: Path) -> dict[str, Any]:
    """Compare observed result categories against retained representative videos."""

    root = Path(evaluation_root).expanduser().resolve()
    observed: dict[tuple[str, str], set[str]] = {}
    for result_path in sorted(root.glob("models/*/tasks/*/seed_*/result.json")):
        result = read_json(result_path)
        key = (str(result["model"]["model_id"]), str(result["episode"]["task"]))
        observed.setdefault(key, set()).add(outcome_category(result))
    reports = []
    for (model_id, task), categories in sorted(observed.items()):
        model_root = root / "models" / model_id
        index = load_representative_index(model_root=model_root, model_id=model_id)
        video_categories = {
            str(record["category"])
            for record in index["records"]
            if str(record["task"]) == task and _bundle_exists(record)
        }
        missing = sorted(categories.difference(video_categories))
        reports.append(
            {
                "model": model_id,
                "task": task,
                "observed_categories": sorted(categories),
                "categories_with_videos": sorted(video_categories),
                "missing_categories": missing,
                "complete": not missing,
            }
        )
    return {
        "coverage_complete": all(report["complete"] for report in reports),
        "selection_policy": SELECTION_POLICY,
        "model_task_reports": reports,
    }
