"""Explicit model/checkpoint specifications and preflight validation."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    training_config: str
    checkpoint_root: Path
    manager_step: int
    norm_asset_id: str
    description: str = ""

    @property
    def manager_directory(self) -> Path:
        return self.checkpoint_root / str(self.manager_step)

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["checkpoint_root"] = str(self.checkpoint_root)
        value["resolved_manager_directory"] = str(self.manager_directory)
        return value


def load_model_spec(path: Path) -> ModelSpec:
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    root_value = str(raw["checkpoint_root"]).replace(
        "${XARM_WORK_ROOT}",
        os.environ.get("XARM_WORK_ROOT", "/work/nvme/bfmk/mw89"),
    )
    root_value = os.path.expandvars(root_value)
    if "$" in root_value or "%" in root_value:
        raise ValueError("Model checkpoint_root contains an unresolved environment variable")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    spec = ModelSpec(
        model_id=str(raw["model_id"]),
        training_config=str(raw["training_config"]),
        checkpoint_root=root.resolve(),
        manager_step=int(raw["manager_step"]),
        norm_asset_id=str(raw["norm_asset_id"]),
        description=str(raw.get("description", "")),
    )
    validate_model_spec(spec)
    return spec


def validate_model_spec(spec: ModelSpec) -> None:
    if not spec.model_id or not spec.training_config or not spec.norm_asset_id:
        raise ValueError("Model spec requires model_id, training_config, and norm_asset_id")
    if spec.manager_step < 0:
        raise ValueError("manager_step must be non-negative")
    manager = spec.manager_directory
    required = (
        manager,
        manager / "params",
        manager / "params" / "manifest.ocdbt",
        manager / "assets",
        manager / "assets" / spec.norm_asset_id / "norm_stats.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Model checkpoint is incomplete: " + ", ".join(missing))
    norm_path = required[-1]
    if not norm_path.is_file() or norm_path.stat().st_size == 0:
        raise ValueError(f"Model normalization asset is empty: {norm_path}")


def validate_training_config_asset(spec: ModelSpec, *, openpi_root: Path) -> str:
    """Confirm the selected OpenPI config expects the spec's embedded asset."""

    source = Path(openpi_root).expanduser().resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"OpenPI source tree not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from openpi.training import checkpoints  # pylint: disable=import-outside-toplevel
    from openpi.training import config as training_config  # pylint: disable=import-outside-toplevel

    config = training_config.get_config(spec.training_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    asset_id = data_config.asset_id
    if asset_id != spec.norm_asset_id:
        raise ValueError(
            f"Model spec norm asset {spec.norm_asset_id!r} differs from training config "
            f"asset {asset_id!r}"
        )
    # Use OpenPI's normal loader now, before a Slurm job starts a GPU policy
    # server. This catches malformed or misplaced embedded assets early.
    loaded = checkpoints.load_norm_stats(spec.manager_directory / "assets", str(asset_id))
    if not loaded:
        raise ValueError("Checkpoint normalization asset loaded as an empty mapping")
    return str(asset_id)


def validate_abc_comparison_specs(specs: tuple[ModelSpec, ...]) -> dict[str, Any]:
    """Check comparison invariants a single-model evaluation cannot observe."""

    by_id = {spec.model_id: spec for spec in specs}
    if set(by_id) != {"A", "B", "C"} or len(by_id) != len(specs):
        raise ValueError("Expected exactly one explicit A, B, and C model specification")
    if len({spec.manager_step for spec in specs}) != 1:
        raise ValueError("A/B/C manager steps must match")
    if len({spec.checkpoint_root for spec in specs}) != 3:
        raise ValueError("A/B/C checkpoint roots must be distinct")
    if by_id["A"].norm_asset_id == by_id["B"].norm_asset_id:
        raise ValueError("A must not use the B/C normalization asset")
    if by_id["B"].norm_asset_id != by_id["C"].norm_asset_id:
        raise ValueError("B/C must use the same normalization asset ID")
    b_norm = (
        by_id["B"].manager_directory
        / "assets"
        / by_id["B"].norm_asset_id
        / "norm_stats.json"
    )
    c_norm = (
        by_id["C"].manager_directory
        / "assets"
        / by_id["C"].norm_asset_id
        / "norm_stats.json"
    )
    b_hash = hashlib.sha256(b_norm.read_bytes()).hexdigest()
    c_hash = hashlib.sha256(c_norm.read_bytes()).hexdigest()
    if b_hash != c_hash:
        raise ValueError("B/C embedded normalization assets differ byte-for-byte")
    return {
        "manager_step": by_id["A"].manager_step,
        "b_c_norm_asset_id": by_id["B"].norm_asset_id,
        "b_c_norm_stats_sha256": b_hash,
        "checkpoint_roots": {
            key: str(value.checkpoint_root) for key, value in by_id.items()
        },
    }
