#!/usr/bin/env python3
"""Quality-check raw xArm demonstrations before LeRobot conversion.

Checks the raw layout used by data.real.conversion.convert_xarm_raw_to_lerobot:

    raw/<task>/episode_xxx/
      meta.json
      robot_log.csv
      gripper_events.csv
      realsense_0/*.png
      realsense_1/*.png
      realsense_2/*.png  # optional / currently unused

The script is intentionally dependency-free so it can run locally or in Colab.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from data.common.schema import XARM_STATE_COLUMNS
from data.real.config import get_raw_data_root


STATE_COLUMNS = XARM_STATE_COLUMNS
REQUIRED_IMAGE_COLUMNS = ("realsense_0_file", "realsense_1_file")
OPTIONAL_IMAGE_COLUMNS = ("realsense_2_file",)
REQUIRED_COLUMNS = ("ts", *STATE_COLUMNS, *REQUIRED_IMAGE_COLUMNS)

STAGE_RANGES = (
    (0, 9, "stage_0_pipeline_validation"),
    (10, 49, "stage_1_single_block_basic"),
    (50, 89, "stage_2_single_block_randomized"),
    (90, 134, "stage_3_three_block_selection"),
    (135, 154, "stage_4_hard_pick_up"),
    (155, 167, "stage_5_recovery_pick_up"),
)


@dataclass
class EpisodeReport:
    task: str
    episode_name: str
    path: Path
    rows: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dt_mean: float | None = None
    dt_min: float | None = None
    dt_max: float | None = None
    max_joint_delta: float | None = None
    max_gripper_delta: float | None = None
    gripper_min: float | None = None
    gripper_max: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows: list[dict[str, str]] = []
        for row in csv.DictReader(f):
            rows.append({str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None})
        return rows


def png_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) for PNG without importing PIL/cv2."""
    try:
        with path.open("rb") as f:
            header = f.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            return None
        width, height = struct.unpack(">II", header[16:24])
        return int(width), int(height)
    except OSError:
        return None


def parse_episode_index(name: str) -> int | None:
    if not name.startswith("episode_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def stage_for_episode(name: str) -> str:
    idx = parse_episode_index(name)
    if idx is None:
        return "unknown"
    for start, end, stage in STAGE_RANGES:
        if start <= idx <= end:
            return stage
    return "outside_plan"


def as_float(row: dict[str, str], key: str, report: EpisodeReport, row_idx: int) -> float | None:
    try:
        value = float(row[key])
    except Exception:
        report.errors.append(f"row {row_idx}: invalid float in {key!r}: {row.get(key)!r}")
        return None
    if not math.isfinite(value):
        report.errors.append(f"row {row_idx}: non-finite value in {key!r}: {value}")
        return None
    return value


def check_episode(
    episode_dir: Path,
    *,
    expected_fps: float,
    fps_tolerance: float,
    expected_width: int | None,
    expected_height: int | None,
    max_joint_abs: float,
    max_joint_delta_warn: float,
    max_gripper_delta_warn: float,
    gripper_min_warn: float,
    gripper_max_warn: float,
    check_optional_camera: bool,
    stationary_duration: float,
    stationary_joint_threshold: float,
    stationary_gripper_threshold: float,
) -> EpisodeReport:
    task = episode_dir.parent.name
    report = EpisodeReport(task=task, episode_name=episode_dir.name, path=episode_dir)

    meta_path = episode_dir / "meta.json"
    csv_path = episode_dir / "robot_log.csv"
    gripper_events_path = episode_dir / "gripper_events.csv"

    if not meta_path.exists():
        report.errors.append("missing meta.json")
        return report
    if not csv_path.exists():
        report.errors.append("missing robot_log.csv")
        return report
    if not gripper_events_path.exists():
        report.warnings.append("missing gripper_events.csv")

    try:
        meta = read_json(meta_path)
    except Exception as exc:
        report.errors.append(f"invalid meta.json: {exc}")
        return report

    meta_task = str(meta.get("task") or "")
    if meta_task and meta_task != task:
        report.warnings.append(f"meta task {meta_task!r} differs from folder task {task!r}")

    try:
        rows = read_csv_rows(csv_path)
    except Exception as exc:
        report.errors.append(f"invalid robot_log.csv: {exc}")
        return report

    report.rows = len(rows)
    if len(rows) < 2:
        report.errors.append("need at least 2 robot_log rows")
        return report

    columns = set(rows[0])
    missing = sorted(set(REQUIRED_COLUMNS) - columns)
    if missing:
        report.errors.append(f"missing required columns: {missing}")
        return report

    if check_optional_camera:
        for col in OPTIONAL_IMAGE_COLUMNS:
            if col not in columns:
                report.warnings.append(f"optional camera column missing: {col}")

    timestamps: list[float] = []
    states: list[list[float]] = []
    missing_images = 0
    bad_image_sizes: list[str] = []

    for row_idx, row in enumerate(rows):
        ts = as_float(row, "ts", report, row_idx)
        state = [as_float(row, col, report, row_idx) for col in STATE_COLUMNS]
        if ts is not None and all(v is not None for v in state):
            timestamps.append(ts)
            states.append([float(v) for v in state if v is not None])

        image_cols = list(REQUIRED_IMAGE_COLUMNS)
        if check_optional_camera:
            image_cols += [col for col in OPTIONAL_IMAGE_COLUMNS if col in row]

        for col in image_cols:
            rel = row.get(col, "")
            if not rel:
                missing_images += 1
                continue
            img_path = episode_dir / rel
            if not img_path.exists():
                missing_images += 1
                if missing_images <= 5:
                    report.errors.append(f"row {row_idx}: missing image {col}={rel}")
                continue
            if expected_width is not None and expected_height is not None and img_path.suffix.lower() == ".png":
                size = png_size(img_path)
                if size is None:
                    bad_image_sizes.append(f"{rel}: not a valid PNG")
                elif size != (expected_width, expected_height):
                    bad_image_sizes.append(f"{rel}: {size[0]}x{size[1]}")

    if missing_images > 5:
        report.errors.append(f"missing {missing_images} image references total")

    if bad_image_sizes:
        preview = "; ".join(bad_image_sizes[:5])
        report.warnings.append(f"unexpected image sizes: {preview}")
        if len(bad_image_sizes) > 5:
            report.warnings.append(f"unexpected image size count: {len(bad_image_sizes)}")

    if len(timestamps) != len(rows) or len(states) != len(rows):
        return report

    dts = [b - a for a, b in zip(timestamps, timestamps[1:])]
    if not all(dt > 0 for dt in dts):
        report.errors.append("timestamps are not strictly increasing")
    if dts:
        report.dt_mean = mean(dts)
        report.dt_min = min(dts)
        report.dt_max = max(dts)
        expected_dt = 1.0 / expected_fps
        if abs(report.dt_mean - expected_dt) > fps_tolerance:
            report.warnings.append(f"mean dt {report.dt_mean:.4f}s differs from expected {expected_dt:.4f}s")
        long_dt = [dt for dt in dts if dt > 0.5]
        if long_dt:
            report.warnings.append(f"long timestamp gaps >0.5s: {len(long_dt)} max={max(long_dt):.3f}s")

    joint_values = [abs(v) for state in states for v in state[:6]]
    if joint_values and max(joint_values) > max_joint_abs:
        report.warnings.append(
            f"joint abs max {max(joint_values):.3f} exceeds {max_joint_abs}; data may be degrees, expected radians"
        )

    grippers = [state[6] for state in states]
    report.gripper_min = min(grippers)
    report.gripper_max = max(grippers)
    if report.gripper_min < gripper_min_warn or report.gripper_max > gripper_max_warn:
        report.warnings.append(
            f"gripper range {report.gripper_min:.1f}..{report.gripper_max:.1f} outside expected "
            f"{gripper_min_warn:.1f}..{gripper_max_warn:.1f}"
        )

    joint_deltas = []
    gripper_deltas = []
    for a, b in zip(states, states[1:]):
        joint_deltas.extend(abs(b[i] - a[i]) for i in range(6))
        gripper_deltas.append(abs(b[6] - a[6]))
    if joint_deltas:
        report.max_joint_delta = max(joint_deltas)
        if report.max_joint_delta > max_joint_delta_warn:
            report.warnings.append(f"large joint delta max={report.max_joint_delta:.3f} rad")
    if gripper_deltas:
        report.max_gripper_delta = max(gripper_deltas)
        if report.max_gripper_delta > max_gripper_delta_warn:
            report.warnings.append(f"large gripper delta max={report.max_gripper_delta:.1f} mm")

    # Detect long stationary segments: consecutive frames where movement is below thresholds.
    if stationary_duration > 0 and len(states) >= 2:
        # compute per-pair max joint delta and gripper delta
        pair_joint_max: list[float] = []
        pair_gripper_delta: list[float] = []
        for a, b in zip(states, states[1:]):
            pair_joint_max.append(max(abs(b[i] - a[i]) for i in range(6)))
            pair_gripper_delta.append(abs(b[6] - a[6]))

        # accumulate consecutive stationary pairs' durations
        cur_duration = 0.0
        cur_start = None
        stationary_segments: list[tuple[int, int, float]] = []  # (start_idx, end_idx, duration)
        for i, dt in enumerate(dts):
            is_stationary = (
                pair_joint_max[i] <= stationary_joint_threshold and pair_gripper_delta[i] <= stationary_gripper_threshold
            )
            if is_stationary:
                if cur_start is None:
                    cur_start = i
                    cur_duration = dt
                else:
                    cur_duration += dt
            else:
                if cur_start is not None:
                    if cur_duration >= stationary_duration:
                        stationary_segments.append((cur_start, i, cur_duration))
                    cur_start = None
                    cur_duration = 0.0
        # tail
        if cur_start is not None and cur_duration >= stationary_duration:
            stationary_segments.append((cur_start, len(dts), cur_duration))

        if stationary_segments:
            for start_i, end_i, dur in stationary_segments[:5]:
                # report rows as inclusive range of row indices (0-based)->present as counts
                report.warnings.append(
                    f"stationary segment rows {start_i}..{end_i} duration={dur:.3f}s "
                    f"(joint<{stationary_joint_threshold}, gripper<{stationary_gripper_threshold})"
                )
            if len(stationary_segments) > 5:
                report.warnings.append(f"{len(stationary_segments)} stationary segments total")

    return report


def print_report(reports: list[EpisodeReport], *, show_ok: bool, max_messages: int) -> None:
    task_counts = Counter(r.task for r in reports)
    stage_counts = Counter((r.task, stage_for_episode(r.episode_name)) for r in reports)
    row_counts = defaultdict(int)
    for report in reports:
        row_counts[report.task] += report.rows

    errors = sum(len(r.errors) for r in reports)
    warnings = sum(len(r.warnings) for r in reports)
    ok_episodes = sum(1 for r in reports if r.ok)

    print("\n=== Summary ===")
    print(f"episodes: {len(reports)} total, {ok_episodes} without errors")
    print(f"errors: {errors}")
    print(f"warnings: {warnings}")

    print("\n=== Tasks ===")
    for task in sorted(task_counts):
        print(f"{task}: {task_counts[task]} episodes, {row_counts[task]} raw rows")

    print("\n=== Stage Counts ===")
    for task in sorted(task_counts):
        print(f"{task}:")
        for _, _, stage in (*STAGE_RANGES,):
            print(f"  {stage}: {stage_counts[(task, stage)]}")
        outside = stage_counts[(task, "outside_plan")]
        unknown = stage_counts[(task, "unknown")]
        if outside:
            print(f"  outside_plan: {outside}")
        if unknown:
            print(f"  unknown: {unknown}")

    print("\n=== Episode Issues ===")
    printed = 0
    for report in reports:
        messages = [("ERROR", msg) for msg in report.errors] + [("WARN", msg) for msg in report.warnings]
        if not messages and show_ok:
            print(f"OK {report.task}/{report.episode_name}: rows={report.rows}")
        for level, msg in messages:
            if printed >= max_messages:
                print(f"... stopped after {max_messages} messages")
                return
            print(f"{level} {report.task}/{report.episode_name}: {msg}")
            printed += 1
    if printed == 0:
        print("No issues found.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None, help="Override raw data root from xarm_data_config.json.")
    parser.add_argument("--expected-fps", type=float, default=10.0)
    parser.add_argument("--fps-tolerance", type=float, default=0.03, help="Allowed mean dt error in seconds.")
    parser.add_argument("--expected-width", type=int, default=640)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--max-joint-abs", type=float, default=6.5)
    parser.add_argument("--max-joint-delta-warn", type=float, default=0.25)
    parser.add_argument("--max-gripper-delta-warn", type=float, default=250.0)
    parser.add_argument("--gripper-min-warn", type=float, default=100.0)
    parser.add_argument("--gripper-max-warn", type=float, default=900.0)
    parser.add_argument("--check-optional-camera", action="store_true", help="Also check realsense_2_file if present.")
    parser.add_argument(
        "--short-length-fraction",
        type=float,
        default=0.5,
        help="Flag episodes shorter than (median rows * FRACTION). Set <=0 to disable.",
    )
    parser.add_argument(
        "--stationary-duration",
        type=float,
        default=2.0,
        help="Duration (s) to consider a segment stationary (<=0 to disable).",
    )
    parser.add_argument(
        "--stationary-joint-threshold",
        type=float,
        default=1e-3,
        help="Max joint delta (rad) per frame to count as stationary.",
    )
    parser.add_argument(
        "--stationary-gripper-threshold",
        type=float,
        default=1.0,
        help="Max gripper delta (mm) per frame to count as stationary.",
    )
    parser.add_argument("--show-ok", action="store_true")
    parser.add_argument("--max-messages", type=int, default=80)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on warnings as well as errors.")
    args = parser.parse_args()

    raw_root = get_raw_data_root(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"raw root does not exist: {raw_root}")

    episode_dirs = sorted(p.parent for p in raw_root.glob("*/*/meta.json"))
    if not episode_dirs:
        raise SystemExit(f"no episodes found under {raw_root}")

    reports = [
        check_episode(
            episode_dir,
            expected_fps=args.expected_fps,
            fps_tolerance=args.fps_tolerance,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            max_joint_abs=args.max_joint_abs,
            max_joint_delta_warn=args.max_joint_delta_warn,
            max_gripper_delta_warn=args.max_gripper_delta_warn,
            gripper_min_warn=args.gripper_min_warn,
            gripper_max_warn=args.gripper_max_warn,
            check_optional_camera=args.check_optional_camera,
            stationary_duration=args.stationary_duration,
            stationary_joint_threshold=args.stationary_joint_threshold,
            stationary_gripper_threshold=args.stationary_gripper_threshold,
        )
        for episode_dir in episode_dirs
    ]

    # Detect unusually short episodes per task: compare rows to task median.
    if args.short_length_fraction > 0:
        task_rows: dict[str, list[int]] = defaultdict(list)
        for r in reports:
            task_rows[r.task].append(r.rows)
        task_medians: dict[str, float] = {t: median(v) if v else 0.0 for t, v in task_rows.items()}
        for r in reports:
            med = task_medians.get(r.task, 0.0)
            if med > 0 and r.rows < med * args.short_length_fraction:
                r.warnings.append(
                    f"short episode length rows={r.rows} << task median {int(med)} (fraction={args.short_length_fraction})"
                )

    print_report(reports, show_ok=args.show_ok, max_messages=args.max_messages)

    error_count = sum(len(r.errors) for r in reports)
    warning_count = sum(len(r.warnings) for r in reports)
    if error_count or (args.strict and warning_count):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
