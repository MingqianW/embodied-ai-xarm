"""Portable runtime paths owned by the simulation-data subsystem."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from simulation.resources import repository_root


def dataset_root(environ: Mapping[str, str] | None = None) -> Path:
    """Return the external or ignored root for generated simulation datasets."""

    values = os.environ if environ is None else environ
    configured = values.get("MUJOCO_DATASET_ROOT")
    default = repository_root(values) / "datasets" / "simulation"
    path = Path(configured).expanduser() if configured else default
    return path.resolve()
