from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from policy_runtime.schemas import POLICY_SCHEMA_VERSION


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass
class EpisodeLogger:
    output_dir: Path
    simulator: str
    schema_version: str = POLICY_SCHEMA_VERSION
    _event_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._event_path = self.output_dir / "events.jsonl"

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "schema_version": self.schema_version,
            "timestamp_s": time.time(),
            "simulator": self.simulator,
            "event": event,
            **payload,
        }
        with self._event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=json_default, sort_keys=True) + "\n")

    def save_array(self, name: str, value: np.ndarray) -> Path:
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(value)
        np.save(path, array)
        write_json(
            path.with_name(f"{path.name}.meta.json"),
            {
                "schema_version": self.schema_version,
                "simulator": self.simulator,
                "array_path": str(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            },
        )
        return path

    def save_image(self, name: str, value: np.ndarray) -> Path:
        from PIL import Image

        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.asarray(value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Logged image must be RGB uint8 HxWx3, "
                f"got {image.shape}/{image.dtype}"
            )
        Image.fromarray(image).save(path)
        return path

    def write_metadata(self, payload: dict[str, Any]) -> Path:
        path = self.output_dir / "episode.json"
        write_json(
            path,
            {
                "schema_version": self.schema_version,
                "simulator": self.simulator,
                **payload,
            },
        )
        return path
