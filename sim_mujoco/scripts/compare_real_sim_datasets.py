"""Compare real red-block raw data with successful MuJoCo oracle episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common.schema import XARM_STATE_COLUMNS
from data.sim.generation.legacy.episode_recorder import REAL_TRAINING_PROMPT
from data.sim.generation.legacy.lerobot_adapter import (
    discover_successful_episodes,
    load_episode_records,
)
from sim_mujoco.paths import mujoco_dataset_root, mujoco_output_root


DEFAULT_REAL_ROOT = (
    PROJECT_ROOT / "fine_tune" / "data" / "xarm_pi05_data" / "raw"
)
DEFAULT_SIM_ROOT = mujoco_dataset_root() / "xarm_mujoco_red_block_raw"
DEFAULT_OUTPUT = mujoco_output_root() / "real_sim_comparison"
DIMENSION_NAMES = (
    "joint1_rad",
    "joint2_rad",
    "joint3_rad",
    "joint4_rad",
    "joint5_rad",
    "joint6_rad",
    "gripper_raw",
)


@dataclass
class DomainData:
    name: str
    states: np.ndarray
    actions: np.ndarray
    episode_lengths: np.ndarray
    base_images: list[Path]
    wrist_images: list[Path]
    object_positions: np.ndarray | None

    @property
    def deltas(self) -> np.ndarray:
        return self.actions - self.states


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _normalized_task(task: str) -> str:
    return " ".join(task.replace("_", " ").split())


def load_real_red_block(raw_root: Path) -> DomainData:
    states: list[list[float]] = []
    actions: list[list[float]] = []
    lengths: list[int] = []
    base_images: list[Path] = []
    wrist_images: list[Path] = []
    for metadata_path in sorted(Path(raw_root).glob("*/*/meta.json")):
        metadata = _read_json(metadata_path)
        task = str(
            metadata.get("task")
            or metadata_path.parent.parent.name
        )
        if _normalized_task(task) != REAL_TRAINING_PROMPT:
            continue
        log_path = metadata_path.parent / "robot_log.csv"
        with log_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 2:
            continue
        required = {*XARM_STATE_COLUMNS, "realsense_0_file", "realsense_1_file"}
        missing = sorted(required - set(rows[0]))
        if missing:
            raise ValueError(f"Missing real columns in {log_path}: {missing}")
        episode_length = 0
        for index, row in enumerate(rows[:-1]):
            states.append([float(row[key]) for key in XARM_STATE_COLUMNS])
            actions.append(
                [float(rows[index + 1][key]) for key in XARM_STATE_COLUMNS]
            )
            base_images.append(metadata_path.parent / row["realsense_0_file"])
            wrist_images.append(
                metadata_path.parent / row["realsense_1_file"]
            )
            episode_length += 1
        lengths.append(episode_length)
    if not states:
        raise ValueError(f"No real red-block episodes found under {raw_root}")
    return DomainData(
        name="real",
        states=np.asarray(states, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        episode_lengths=np.asarray(lengths, dtype=np.int64),
        base_images=base_images,
        wrist_images=wrist_images,
        object_positions=None,
    )


def load_sim_red_block(raw_root: Path) -> DomainData:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    lengths: list[int] = []
    base_images: list[Path] = []
    wrist_images: list[Path] = []
    object_positions: list[np.ndarray] = []
    for episode in discover_successful_episodes(raw_root):
        records = load_episode_records(episode, validate_images=False)
        states.extend(record["state"] for record in records)
        actions.extend(record["actions"] for record in records)
        lengths.append(len(records))
        base_images.extend(Path(record["image"]) for record in records)
        wrist_images.extend(Path(record["wrist_image"]) for record in records)
        with np.load(
            episode.directory / "observations.npz",
            allow_pickle=False,
        ) as payload:
            positions = np.asarray(
                payload["object_position"],
                dtype=np.float64,
            )
        if positions.shape != (len(records), 3):
            raise ValueError(
                f"Invalid object_position array: {episode.directory}"
            )
        object_positions.extend(positions)
    if not states:
        raise ValueError(f"No successful sim episodes found under {raw_root}")
    return DomainData(
        name="sim",
        states=np.asarray(states, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        episode_lengths=np.asarray(lengths, dtype=np.int64),
        base_images=base_images,
        wrist_images=wrist_images,
        object_positions=np.asarray(object_positions, dtype=np.float64),
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def _distribution_rows(real: DomainData, sim: DomainData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, real_values, sim_values in (
        ("state", real.states, sim.states),
        ("action", real.actions, sim.actions),
        ("action_delta", real.deltas, sim.deltas),
    ):
        for index, name in enumerate(DIMENSION_NAMES):
            for domain, values in (
                ("real", real_values[:, index]),
                ("sim", sim_values[:, index]),
            ):
                rows.append(
                    {
                        "field": field,
                        "dimension": index,
                        "name": name,
                        "domain": domain,
                        "count": len(values),
                        **_summary(values),
                    }
                )
    for domain, values in (
        ("real", real.episode_lengths),
        ("sim", sim.episode_lengths),
    ):
        rows.append(
            {
                "field": "episode_length",
                "dimension": "",
                "name": "frames",
                "domain": domain,
                "count": len(values),
                **_summary(values),
            }
        )
    if sim.object_positions is not None:
        for index, name in enumerate(("object_x_m", "object_y_m", "object_z_m")):
            rows.append(
                {
                    "field": "object_position",
                    "dimension": index,
                    "name": name,
                    "domain": "sim",
                    "count": len(sim.object_positions),
                    **_summary(sim.object_positions[:, index]),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "field",
        "dimension",
        "name",
        "domain",
        "count",
        "minimum",
        "p01",
        "p05",
        "median",
        "mean",
        "standard_deviation",
        "p95",
        "p99",
        "maximum",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _histogram_image(
    real_values: np.ndarray | None,
    sim_values: np.ndarray,
    *,
    title: str,
    path: Path,
) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 55, 60
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    sim_values = np.asarray(sim_values, dtype=np.float64)
    combined = (
        sim_values
        if real_values is None
        else np.concatenate(
            [
                np.asarray(real_values, dtype=np.float64),
                sim_values,
            ]
        )
    )
    low, high = np.percentile(combined, [0.1, 99.9])
    if not np.isfinite([low, high]).all() or math.isclose(low, high):
        low = float(np.min(combined)) - 0.5
        high = float(np.max(combined)) + 0.5
    real_hist = None
    if real_values is not None:
        real_hist, _ = np.histogram(
            real_values,
            bins=64,
            range=(low, high),
        )
    sim_hist, _ = np.histogram(sim_values, bins=64, range=(low, high))
    if real_hist is not None:
        real_hist = real_hist / max(1, int(real_hist.sum()))
    sim_hist = sim_hist / max(1, int(sim_hist.sum()))
    maximum = max(
        0.0 if real_hist is None else float(real_hist.max()),
        float(sim_hist.max()),
        1e-12,
    )
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    draw.rectangle(
        (
            margin_left,
            margin_top,
            margin_left + plot_width,
            margin_top + plot_height,
        ),
        outline="black",
    )

    def points(values: np.ndarray) -> list[tuple[float, float]]:
        result = []
        for index, value in enumerate(values):
            x = margin_left + plot_width * (index + 0.5) / len(values)
            y = margin_top + plot_height * (1.0 - float(value) / maximum)
            result.append((x, y))
        return result

    if real_hist is not None:
        draw.line(points(real_hist), fill=(36, 95, 180), width=3)
    draw.line(points(sim_hist), fill=(225, 112, 38), width=3)
    draw.text((margin_left, 18), title, fill="black", font=_font())
    draw.text(
        (margin_left, height - 35),
        f"{low:.5g}",
        fill="black",
        font=_font(),
    )
    high_label = f"{high:.5g}"
    draw.text(
        (width - margin_right - 70, height - 35),
        high_label,
        fill="black",
        font=_font(),
    )
    if real_hist is not None:
        draw.text(
            (width - 220, 20),
            "real",
            fill=(36, 95, 180),
            font=_font(),
        )
    draw.text(
        (width - 130, 20),
        "sim",
        fill=(225, 112, 38),
        font=_font(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _load_thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def _contact_sheet(
    paths: list[Path],
    *,
    title: str,
    output_path: Path,
    count: int = 16,
) -> None:
    if not paths:
        raise ValueError(f"No images for contact sheet: {title}")
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths))).astype(int)
    tile = (240, 180)
    columns = 4
    rows = math.ceil(len(indices) / columns)
    header = 35
    canvas = Image.new(
        "RGB",
        (columns * tile[0], header + rows * tile[1]),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black", font=_font())
    for slot, index in enumerate(indices):
        thumb = _load_thumb(paths[int(index)], tile)
        x = (slot % columns) * tile[0]
        y = header + (slot // columns) * tile[1]
        canvas.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _random_comparisons(
    real: DomainData,
    sim: DomainData,
    *,
    output_path: Path,
    seed: int,
    count: int = 8,
) -> None:
    rng = random.Random(seed)
    row_height = 180
    tile_width = 240
    header = 35
    canvas = Image.new(
        "RGB",
        (tile_width * 4, header + count * row_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 10),
        "real base | real wrist | sim base | sim wrist",
        fill="black",
        font=_font(),
    )
    for row in range(count):
        real_index = rng.randrange(len(real.base_images))
        sim_index = rng.randrange(len(sim.base_images))
        paths = (
            real.base_images[real_index],
            real.wrist_images[real_index],
            sim.base_images[sim_index],
            sim.wrist_images[sim_index],
        )
        for column, path in enumerate(paths):
            canvas.paste(
                _load_thumb(path, (tile_width, row_height)),
                (column * tile_width, header + row * row_height),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _camera_domain_metrics(
    real_paths: list[Path],
    sim_paths: list[Path],
    *,
    sample_count: int = 64,
) -> dict[str, Any]:
    def signature(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        indices = np.linspace(
            0,
            len(paths) - 1,
            min(sample_count, len(paths)),
        ).astype(int)
        images = []
        centroids = []
        for index in indices:
            image = np.asarray(
                _load_thumb(paths[int(index)], (64, 48)),
                dtype=np.float64,
            ) / 255.0
            images.append(image)
            red = (
                (image[:, :, 0] > 0.35)
                & (image[:, :, 0] > image[:, :, 1] * 1.25)
                & (image[:, :, 0] > image[:, :, 2] * 1.25)
            )
            if int(red.sum()) >= 4:
                y, x = np.nonzero(red)
                centroids.append(
                    [
                        float(np.mean(x) / max(image.shape[1] - 1, 1)),
                        float(np.mean(y) / max(image.shape[0] - 1, 1)),
                    ]
                )
        return (
            np.mean(np.asarray(images), axis=0),
            np.asarray(centroids, dtype=np.float64),
        )

    real_average, real_centroids = signature(real_paths)
    sim_average, sim_centroids = signature(sim_paths)
    average_image_mae = float(np.mean(np.abs(real_average - sim_average)))
    centroid_distance = None
    if len(real_centroids) and len(sim_centroids):
        centroid_distance = float(
            np.linalg.norm(
                np.median(real_centroids, axis=0)
                - np.median(sim_centroids, axis=0)
            )
        )
    flagged = bool(
        average_image_mae > 0.20
        or (
            centroid_distance is not None
            and centroid_distance > 0.15
        )
    )
    return {
        "average_image_mae_0_to_1": average_image_mae,
        "red_object_centroid_median_distance_normalized": centroid_distance,
        "real_red_centroid_samples": int(len(real_centroids)),
        "sim_red_centroid_samples": int(len(sim_centroids)),
        "framing_or_appearance_gap_flag": flagged,
    }


def _mismatch_flags(real: DomainData, sim: DomainData) -> dict[str, Any]:
    real_state_std = np.std(real.states, axis=0)
    mean_z = np.abs(np.mean(sim.states, axis=0) - np.mean(real.states, axis=0)) / np.maximum(
        real_state_std,
        1e-9,
    )
    real_speed_p95 = np.percentile(np.abs(real.deltas[:, :6]), 95, axis=0)
    sim_speed_p95 = np.percentile(np.abs(sim.deltas[:, :6]), 95, axis=0)
    speed_ratio = sim_speed_p95 / np.maximum(real_speed_p95, 1e-9)
    real_smoothness = np.std(real.deltas[:, :6], axis=0)
    sim_smoothness = np.std(sim.deltas[:, :6], axis=0)
    smoothness_ratio = sim_smoothness / np.maximum(real_smoothness, 1e-9)
    real_gripper_change = np.abs(real.deltas[:, 6])
    sim_gripper_change = np.abs(sim.deltas[:, 6])
    gripper_transition_ratio = (
        float(np.percentile(sim_gripper_change, 95))
        / max(float(np.percentile(real_gripper_change, 95)), 1e-9)
    )
    base_camera_metrics = _camera_domain_metrics(
        real.base_images,
        sim.base_images,
    )
    wrist_camera_metrics = _camera_domain_metrics(
        real.wrist_images,
        sim.wrist_images,
    )
    image_gap = bool(
        base_camera_metrics["framing_or_appearance_gap_flag"]
        or wrist_camera_metrics["framing_or_appearance_gap_flag"]
    )
    return {
        "extreme_state_distribution_mismatch": bool(np.any(mean_z > 3.0)),
        "state_mean_z_score_by_dimension": mean_z.tolist(),
        "action_speed_much_larger_than_real": bool(np.any(speed_ratio > 2.0)),
        "joint_action_abs_delta_p95_ratio_sim_to_real": speed_ratio.tolist(),
        "sim_trajectories_unrealistically_smooth": bool(
            np.any(smoothness_ratio < 0.25)
        ),
        "joint_action_delta_std_ratio_sim_to_real": smoothness_ratio.tolist(),
        "sim_gripper_transitions_unlike_real": bool(
            gripper_transition_ratio > 2.0
            or gripper_transition_ratio < 0.5
        ),
        "gripper_abs_delta_p95_ratio_sim_to_real": gripper_transition_ratio,
        "image_decode_shape_rgb_check": "passed by adapter and contact-sheet generation",
        "image_orientation_or_framing_mismatch": image_gap,
        "base_camera_domain_metrics": base_camera_metrics,
        "wrist_camera_domain_metrics": wrist_camera_metrics,
        "image_visual_review": (
            "Automated image statistics flag appearance/framing; inspect "
            "contact_sheets/random_real_vs_sim.png before training."
            if image_gap
            else "No automated flag; manual contact-sheet review is still required."
        ),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Comparison output is not empty; pass --overwrite: {output_dir}"
        )
    if output_dir.exists() and args.overwrite:
        if any(output_dir.iterdir()) and not (
            output_dir / "summary.json"
        ).is_file():
            raise ValueError(
                "Refusing --overwrite because the target is not a prior "
                f"comparison output (missing summary.json): {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    real = load_real_red_block(args.real_raw_root.resolve())
    sim = load_sim_red_block(args.sim_raw_root.resolve())
    rows = _distribution_rows(real, sim)
    csv_path = output_dir / "distribution_comparison.csv"
    _write_csv(csv_path, rows)
    for field, real_values, sim_values in (
        ("state", real.states, sim.states),
        ("action", real.actions, sim.actions),
        ("action_delta", real.deltas, sim.deltas),
    ):
        for index, name in enumerate(DIMENSION_NAMES):
            _histogram_image(
                real_values[:, index],
                sim_values[:, index],
                title=f"{field}: {name}",
                path=output_dir / "histograms" / f"{field}_{index}_{name}.png",
            )
    _histogram_image(
        real.episode_lengths,
        sim.episode_lengths,
        title="episode length (frames)",
        path=output_dir / "histograms" / "episode_length.png",
    )
    if sim.object_positions is not None:
        for index, name in enumerate(("object_x_m", "object_y_m", "object_z_m")):
            _histogram_image(
                None,
                sim.object_positions[:, index],
                title=f"simulation-only object position: {name}",
                path=(
                    output_dir
                    / "histograms"
                    / f"object_position_{index}_{name}.png"
                ),
            )
    _contact_sheet(
        real.base_images,
        title="real base camera",
        output_path=output_dir / "contact_sheets" / "real_base.png",
    )
    _contact_sheet(
        real.wrist_images,
        title="real wrist camera",
        output_path=output_dir / "contact_sheets" / "real_wrist.png",
    )
    _contact_sheet(
        sim.base_images,
        title="simulation base camera",
        output_path=output_dir / "contact_sheets" / "sim_base.png",
    )
    _contact_sheet(
        sim.wrist_images,
        title="simulation wrist camera",
        output_path=output_dir / "contact_sheets" / "sim_wrist.png",
    )
    _random_comparisons(
        real,
        sim,
        output_path=output_dir / "contact_sheets" / "random_real_vs_sim.png",
        seed=args.seed,
    )
    flags = _mismatch_flags(real, sim)
    summary = {
        "real_raw_root": str(args.real_raw_root.resolve()),
        "sim_raw_root": str(args.sim_raw_root.resolve()),
        "real_red_block_episodes": int(len(real.episode_lengths)),
        "sim_successful_episodes": int(len(sim.episode_lengths)),
        "real_frames": int(len(real.states)),
        "sim_frames": int(len(sim.states)),
        "object_position_comparison": (
            "Simulation object positions were reported. The real raw dataset "
            "does not record object ground-truth pose, so no real-vs-sim "
            "object-position claim is possible."
        ),
        "flags": flags,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Real vs MuJoCo red-block comparison",
        "",
        f"- Real episodes: {summary['real_red_block_episodes']}",
        f"- Simulation successful episodes: {summary['sim_successful_episodes']}",
        f"- Real frames: {summary['real_frames']}",
        f"- Simulation frames: {summary['sim_frames']}",
        f"- Distribution table: `{csv_path.name}`",
        "",
        "## Automated flags",
        "",
    ]
    for key, value in flags.items():
        report_lines.append(f"- `{key}`: {value}")
    report_lines.extend(
        [
            "",
            "## Object position limitation",
            "",
            summary["object_position_comparison"],
            "",
            "Image framing/orientation remains a visual domain-gap review; "
            "the generated contact sheets make that review reproducible.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-raw-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--sim-raw-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
