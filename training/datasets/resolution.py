"""Independent local-path resolution for named training datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from training.datasets.spec import DatasetSet


class DatasetResolutionError(ValueError):
    pass


def resolve_dataset_paths(
    datasets: DatasetSet,
    overrides: Mapping[str, Path] | None = None,
    *,
    require_exists: bool = True,
) -> dict[str, Path]:
    """Resolve every dataset independently without mutating ``HF_LEROBOT_HOME``.

    Explicit ``dataset_id=path`` overrides win. A DatasetSpec local path is the
    next choice. Finally, a conventional ``HF_LEROBOT_HOME/repo_id`` location
    is used only for that one source; different sources never need to share a
    parent directory.
    """

    provided = {str(dataset_id): Path(path).expanduser() for dataset_id, path in (overrides or {}).items()}
    known_ids = {dataset.dataset_id for dataset in datasets.datasets}
    unknown = sorted(set(provided) - known_ids)
    if unknown:
        raise DatasetResolutionError(f"Unknown dataset path override(s): {unknown}")
    home_text = os.environ.get("HF_LEROBOT_HOME")
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for dataset in datasets.datasets:
        path = provided.get(dataset.dataset_id)
        if path is None and dataset.local_path is not None:
            path = dataset.local_path
        if path is None and home_text:
            path = Path(home_text) / dataset.repo_id
        if path is None:
            missing.append(
                f"{dataset.dataset_id} ({dataset.repo_id}): pass --dataset-path {dataset.dataset_id}=PATH"
            )
            continue
        path = path.resolve()
        if require_exists and not path.is_dir():
            missing.append(f"{dataset.dataset_id} ({dataset.repo_id}): directory does not exist: {path}")
            continue
        resolved[dataset.dataset_id] = path
    if missing:
        raise DatasetResolutionError("Cannot resolve training dataset paths:\n" + "\n".join(missing))
    return resolved
