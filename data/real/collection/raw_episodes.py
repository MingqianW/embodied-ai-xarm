"""Discover and parse the tracked output contract of the real xArm collector.

No hardware is accessed here.  The physical collector is maintained outside
this repository; this module is the explicit offline acquisition boundary for
its ``meta.json``/``robot_log.csv``/camera-file output.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.common.schema import XARM_STATE_COLUMNS


@dataclass(frozen=True)
class RawEpisode:
    raw_id: str
    task: str
    raw_dir: Path
    meta: dict[str, Any]
    rows: list[dict[str, str]]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [
            {
                str(key).strip(): value.strip()
                for key, value in row.items()
                if key is not None
            }
            for row in csv.DictReader(stream)
        ]


def discover_raw_episodes(raw_root: Path) -> list[RawEpisode]:
    """Return valid real-collector episodes in the historical sorted order."""

    episodes: list[RawEpisode] = []
    for meta_path in sorted(Path(raw_root).glob("*/*/meta.json")):
        episode_dir = meta_path.parent
        robot_log = episode_dir / "robot_log.csv"
        if not robot_log.exists():
            print(f"skip {episode_dir}: missing robot_log.csv")
            continue

        meta = read_json(meta_path)
        task = str(meta.get("task") or meta_path.parent.parent.name)
        rows = _read_csv(robot_log)
        if len(rows) < 2:
            print(f"skip {episode_dir}: need at least 2 robot rows")
            continue
        required_columns = {
            "ts",
            *XARM_STATE_COLUMNS,
            "realsense_0_file",
            "realsense_1_file",
        }
        missing_columns = sorted(required_columns - set(rows[0]))
        if missing_columns:
            print(f"skip {episode_dir}: missing columns {missing_columns}")
            continue

        raw_id = meta_path.relative_to(raw_root).parent.as_posix()
        episodes.append(
            RawEpisode(
                raw_id=raw_id,
                task=task,
                raw_dir=episode_dir,
                meta=meta,
                rows=rows,
            )
        )
    return episodes


def state_from_row(row: dict[str, str]) -> list[float]:
    """Return six joint radians and hardware gripper raw, in canonical order."""

    return [float(row[name]) for name in XARM_STATE_COLUMNS]


def instruction_from_task(task: str) -> str:
    """Preserve the real converter's historical underscore normalization."""

    return task.replace("_", " ")

