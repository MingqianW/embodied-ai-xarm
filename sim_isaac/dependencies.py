from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any


ISAAC_MODULE_CANDIDATES = ("isaacsim", "omni", "pxr")


class IsaacDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class IsaacModuleStatus:
    available: dict[str, bool]

    @property
    def ready(self) -> bool:
        # Recent Isaac distributions expose isaacsim; older distributions may
        # expose omni and pxr only from their bundled launcher.
        return self.available.get("isaacsim", False) or (
            self.available.get("omni", False) and self.available.get("pxr", False)
        )


def isaac_module_status() -> IsaacModuleStatus:
    return IsaacModuleStatus(
        {name: importlib.util.find_spec(name) is not None for name in ISAAC_MODULE_CANDIDATES}
    )


def require_isaac_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise IsaacDependencyError(
            f"Isaac Sim module {name!r} is unavailable in this Python interpreter. "
            "Run sim_isaac/scripts/check_isaac_installation.py and then use the "
            "Python launcher supplied by the installed Isaac Sim version."
        ) from exc


def require_isaac_runtime() -> None:
    status = isaac_module_status()
    if not status.ready:
        missing = [name for name, available in status.available.items() if not available]
        raise IsaacDependencyError(
            "Isaac Sim runtime is unavailable; missing module candidates: "
            + ", ".join(missing)
        )

