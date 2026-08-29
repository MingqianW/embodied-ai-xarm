"""Central cluster settings and environment construction."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePath, PurePosixPath


DEFAULT_WORK_ROOT = Path("/work/nvme/bfmk/mw89")
DEFAULT_ACCOUNT = "bfmk-dtai-gh"
DEFAULT_PARTITION = "ghx4"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _deployment_path(value: object, *, resolve_local: bool = False) -> PurePath:
    text = str(value)
    if os.name == "nt" and text.startswith("/"):
        return PurePosixPath(text)
    path = Path(text).expanduser()
    return path.resolve() if resolve_local else path


@dataclass(frozen=True)
class ClusterSettings:
    repository: PurePath
    work_root: PurePath
    openpi_root: PurePath
    python: PurePath
    account: str
    partition: str
    log_root: PurePath

    @classmethod
    def from_environment(cls) -> "ClusterSettings":
        repository = _deployment_path(
            os.environ.get("XARM_REPOSITORY", repository_root()), resolve_local=True
        )
        work_root: PurePath = (
            _deployment_path(os.environ["XARM_WORK_ROOT"])
            if "XARM_WORK_ROOT" in os.environ
            else _deployment_path(DEFAULT_WORK_ROOT.as_posix())
        )
        openpi_root: PurePath = (
            _deployment_path(os.environ["OPENPI_ROOT"])
            if "OPENPI_ROOT" in os.environ
            else _deployment_path("/u/mw89/repos/openpi")
        )
        python: PurePath = (
            _deployment_path(os.environ["XARM_PYTHON"])
            if "XARM_PYTHON" in os.environ
            else openpi_root / ".venv" / "bin" / "python"
        )
        return cls(
            repository=repository,
            work_root=work_root,
            openpi_root=openpi_root,
            python=python,
            account=os.environ.get("XARM_SLURM_ACCOUNT", DEFAULT_ACCOUNT),
            partition=os.environ.get("XARM_SLURM_PARTITION", DEFAULT_PARTITION),
            log_root=(
                _deployment_path(os.environ["XARM_CLUSTER_LOG_ROOT"])
                if "XARM_CLUSTER_LOG_ROOT" in os.environ
                else work_root / "logs" / "cluster"
            ),
        )

    def runtime_environment(self) -> dict[str, str]:
        cache = self.work_root / "caches"
        return {
            "XARM_REPOSITORY": str(self.repository),
            "XARM_WORK_ROOT": str(self.work_root),
            "XARM_CLUSTER_LOG_ROOT": str(self.log_root),
            "XARM_PYTHON": str(self.python),
            "OPENPI_ROOT": str(self.openpi_root),
            "PYTHONUNBUFFERED": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_OUTPUT_ROOT": str(self.work_root / "mujoco_outputs"),
            "MUJOCO_DATASET_ROOT": str(self.work_root / "mujoco_datasets"),
            "HF_LEROBOT_HOME": str(self.work_root / "mujoco_datasets"),
            "HF_HOME": str(cache / "huggingface"),
            "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
            "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
            "UV_CACHE_DIR": str(cache / "uv"),
        }
