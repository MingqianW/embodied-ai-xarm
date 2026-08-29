"""Training-owned selection of normalization assets; statistics stay in OpenPI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NormalizationMode(str, Enum):
    COMPUTE_FROM_DATASETS = "compute_from_datasets"
    PRECOMPUTED_ASSET = "precomputed_asset"
    PRESERVE_CHECKPOINT = "preserve_checkpoint"


@dataclass(frozen=True)
class NormalizationSpec:
    mode: NormalizationMode
    asset_id: str
    assets_dir: str | None = None
    sha256: str | None = None
    use_quantiles: bool = True
    state_semantics: str = "7D absolute state; six joint radians plus raw controller gripper value"
    action_semantics: str = "six joint deltas relative to state; absolute raw gripper value"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", NormalizationMode(self.mode))
        if not self.asset_id.strip():
            raise ValueError("normalization asset_id must be non-empty")
        if self.mode is NormalizationMode.PRESERVE_CHECKPOINT and not self.assets_dir:
            raise ValueError("preserve_checkpoint requires an immutable checkpoint assets directory")

    @property
    def requires_existing_asset(self) -> bool:
        return self.mode is not NormalizationMode.COMPUTE_FROM_DATASETS
