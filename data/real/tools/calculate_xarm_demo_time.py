#!/usr/bin/env python3
"""Summarize raw xArm demonstration time by task and overall.

Expected raw layout:

    raw/<task>/episode_xxx/
      meta.json
      robot_log.csv  # must contain a ts column

The script is intentionally dependency-free so it can run locally or in Colab.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from data.real.config import get_raw_data_root


@dataclass
class EpisodeDuration:
    task: str
    episode_name: str
    path: Path
    rows: int = 0
    duration_s: float = 0.0
    first_ts: float | None = None
    last_ts: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((seconds - whole_seconds) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
        if secs == 60:
            minutes += 1
            secs = 0
        if minutes == 60:
            hours += 1
            minutes = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def read_timestamps(csv_path: Path) -> tuple[list[float], list[str]]:
    warnings: list[str] = []
    timestamps: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("empty CSV")
        fieldnames = [name.strip() for name in reader.fieldnames]
        if "ts" not in fieldnames:
            raise ValueError("missing required ts column")

        for row_idx, row in enumerate(reader, start=2):
            normalized = {
                str(key).strip(): (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
                if key is not None
            }
            raw_ts = normalized.get("ts", "")
            try:
                timestamps.append(float(raw_ts))
            except Exception:
                warnings.append(f"row {row_idx}: invalid timestamp {raw_ts!r}")

    return timestamps, warnings


def measure_episode(episode_dir: Path, *, include_last_frame: bool) -> EpisodeDuration:
    report = EpisodeDuration(task=episode_dir.parent.name, episode_name=episode_dir.name, path=episode_dir)
    csv_path = episode_dir / "robot_log.csv"

    if not csv_path.exists():
        report.errors.append("missing robot_log.csv")
        return report

    try:
        timestamps, warnings = read_timestamps(csv_path)
    except Exception as exc:
        report.errors.append(f"invalid robot_log.csv: {exc}")
        return report

    report.warnings.extend(warnings)
    report.rows = len(timestamps)
    if len(timestamps) < 2:
        report.errors.append("need at least 2 valid timestamps")
        return report

    dts = [b - a for a, b in zip(timestamps, timestamps[1:])]
    non_positive = [dt for dt in dts if dt <= 0]
    if non_positive:
        report.errors.append("timestamps are not strictly increasing")
        return report

    report.first_ts = timestamps[0]
    report.last_ts = timestamps[-1]
    report.duration_s = report.last_ts - report.first_ts
    if include_last_frame:
        report.duration_s += median(dts)

    return report


def print_report(reports: list[EpisodeDuration], *, show_episodes: bool, max_messages: int) -> None:
    by_task: dict[str, list[EpisodeDuration]] = defaultdict(list)
    for report in reports:
        by_task[report.task].append(report)

    print("\n=== Task Durations ===")
    print(f"{'task':<34} {'episodes':>8} {'rows':>10} {'seconds':>12} {'duration':>15}")
    print("-" * 84)
    for task in sorted(by_task):
        task_reports = by_task[task]
        rows = sum(r.rows for r in task_reports if r.ok)
        duration_s = sum(r.duration_s for r in task_reports if r.ok)
        ok_count = sum(1 for r in task_reports if r.ok)
        print(f"{task:<34} {ok_count:>8} {rows:>10} {duration_s:>12.3f} {format_duration(duration_s):>15}")

    total_duration_s = sum(r.duration_s for r in reports if r.ok)
    total_rows = sum(r.rows for r in reports if r.ok)
    ok_episodes = sum(1 for r in reports if r.ok)
    print("-" * 84)
    print(f"{'TOTAL':<34} {ok_episodes:>8} {total_rows:>10} {total_duration_s:>12.3f} {format_duration(total_duration_s):>15}")

    if show_episodes:
        print("\n=== Episode Durations ===")
        print(f"{'task/episode':<52} {'rows':>10} {'seconds':>12} {'duration':>15}")
        print("-" * 93)
        for report in reports:
            name = f"{report.task}/{report.episode_name}"
            if report.ok:
                print(f"{name:<52} {report.rows:>10} {report.duration_s:>12.3f} {format_duration(report.duration_s):>15}")
            else:
                print(f"{name:<52} {'ERROR':>10} {'':>12} {'':>15}")

    messages_printed = 0
    print("\n=== Issues ===")
    for report in reports:
        messages = [("ERROR", msg) for msg in report.errors] + [("WARN", msg) for msg in report.warnings]
        for level, msg in messages:
            if messages_printed >= max_messages:
                print(f"... stopped after {max_messages} messages")
                return
            print(f"{level} {report.task}/{report.episode_name}: {msg}")
            messages_printed += 1
    if messages_printed == 0:
        print("No issues found.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None, help="Override raw data root from xarm_data_config.json.")
    parser.add_argument(
        "--include-last-frame",
        action="store_true",
        help="Add the median frame interval to each episode duration.",
    )
    parser.add_argument("--show-episodes", action="store_true", help="Print one line per episode.")
    parser.add_argument("--max-messages", type=int, default=80)
    args = parser.parse_args()

    raw_root = get_raw_data_root(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"raw root does not exist: {raw_root}")

    episode_dirs = sorted(p.parent for p in raw_root.glob("*/*/meta.json"))
    if not episode_dirs:
        raise SystemExit(f"no episodes found under {raw_root}")

    reports = [measure_episode(episode_dir, include_last_frame=args.include_last_frame) for episode_dir in episode_dirs]
    print_report(reports, show_episodes=args.show_episodes, max_messages=args.max_messages)

    if any(r.errors for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
