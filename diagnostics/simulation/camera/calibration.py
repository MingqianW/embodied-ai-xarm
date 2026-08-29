from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import mujoco
import numpy as np
import yaml
from openpi_client import image_tools
from PIL import Image, ImageDraw
from data.real.config import get_raw_data_root
from simulation.robot.gripper import set_raw_gripper_configuration
from simulation.resources import DEFAULT_CAMERA_CONFIG_PATH
from simulation.resources import DEFAULT_MODEL_PATH
from simulation.resources import repository_root


PROJECT_ROOT = repository_root()
RAW_ROOT = get_raw_data_root()
MODEL_PATH = DEFAULT_MODEL_PATH
CONFIG_PATH = DEFAULT_CAMERA_CONFIG_PATH
CALIBRATION_ROOT = Path(__file__).resolve().parent
BASELINE_CONFIG_PATH = CALIBRATION_ROOT / "baseline_camera_calibration.yaml"
MANIFEST_PATH = CALIBRATION_ROOT / "selected_frames.json"
METRICS_PATH = CALIBRATION_ROOT / "calibration_metrics.json"
DISCOVERY_PATH = CALIBRATION_ROOT / "dataset_discovery.json"

JOINT_COLUMNS = tuple(f"j{i}_rad" for i in range(1, 7))
CAMERA_KEYS = {"base_camera": "realsense_0_file", "wrist_camera": "realsense_1_file"}


@dataclass(frozen=True)
class Episode:
    task: str
    episode_name: str
    directory: Path
    meta: dict[str, Any]
    rows: list[dict[str, str]]

    @property
    def identifier(self) -> str:
        return self.directory.relative_to(RAW_ROOT).as_posix()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / Path(value)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


def write_config(config: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, default_flow_style=False)


def discover_episodes(raw_root: Path = RAW_ROOT) -> list[Episode]:
    episodes: list[Episode] = []
    for meta_path in sorted(raw_root.glob("*/*/meta.json")):
        episode_dir = meta_path.parent
        log_path = episode_dir / "robot_log.csv"
        if not log_path.is_file():
            continue
        meta = read_json(meta_path)
        with log_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
        if not rows:
            continue
        episodes.append(
            Episode(
                task=str(meta.get("task") or episode_dir.parent.name),
                episode_name=episode_dir.name,
                directory=episode_dir,
                meta=meta,
                rows=rows,
            )
        )
    return episodes


def dataset_summary(episodes: list[Episode]) -> dict[str, Any]:
    tasks: dict[str, int] = {}
    resolutions: set[tuple[int, int]] = set()
    total_rows = 0
    valid_image_rows = 0
    for episode in episodes:
        tasks[episode.task] = tasks.get(episode.task, 0) + 1
        total_rows += len(episode.rows)
        for camera in episode.meta.get("cameras", []):
            resolutions.add((int(camera["width"]), int(camera["height"])))
        for row in episode.rows:
            if all((episode.directory / row.get(key, "")).is_file() for key in CAMERA_KEYS.values()):
                valid_image_rows += 1
    first = episodes[0].rows[0] if episodes else {}
    return {
        "raw_root": relative_path(RAW_ROOT),
        "episode_count": len(episodes),
        "task_episode_counts": tasks,
        "total_robot_rows": total_rows,
        "rows_with_base_and_wrist_images": valid_image_rows,
        "image_resolutions": [list(item) for item in sorted(resolutions)],
        "camera_mapping": {
            "base_camera": "realsense_0 (RGB PNG)",
            "wrist_camera": "realsense_1 (RGB PNG)",
            "unused_camera": "realsense_2",
        },
        "joint_state_order": list(JOINT_COLUMNS),
        "joint_units": "radians",
        "gripper_column": "gripper_mm",
        "csv_columns": list(first),
        "opencv_note": "cv2.imread returns BGR; calibration loader converts BGR to RGB",
        "tcp_note": "tcp_x_m/tcp_y_m/tcp_z_m contain millimeters despite their suffix",
    }


def load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def frame_quality(base_rgb: np.ndarray, wrist_rgb: np.ndarray) -> float:
    values = []
    for image in (base_rgb, wrist_rgb):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sharpness = math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        edges = cv2.Canny(gray, 60, 160)
        density = float(np.mean(edges > 0))
        values.append(sharpness + 5.0 * min(density, 0.15))
    wrist_gray = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2GRAY)
    wrist_top_contrast = float(np.std(wrist_gray[: wrist_gray.shape[0] // 2])) / 32.0
    return float(sum(values) + wrist_top_contrast)


def sample_candidates(episodes: list[Episode], max_episodes: int = 36) -> list[dict[str, Any]]:
    if len(episodes) > max_episodes:
        indices = np.linspace(0, len(episodes) - 1, max_episodes, dtype=int)
        episodes = [episodes[index] for index in sorted(set(indices.tolist()))]
    candidates: list[dict[str, Any]] = []
    for episode in episodes:
        row_indices = np.linspace(0, len(episode.rows) - 1, 7, dtype=int)
        for row_index in sorted(set(row_indices.tolist())):
            row = episode.rows[row_index]
            try:
                joints = [float(row[column]) for column in JOINT_COLUMNS]
                gripper = float(row["gripper_mm"])
                base_path = episode.directory / row[CAMERA_KEYS["base_camera"]]
                wrist_path = episode.directory / row[CAMERA_KEYS["wrist_camera"]]
                base_rgb = load_rgb(base_path)
                wrist_rgb = load_rgb(wrist_path)
            except (KeyError, ValueError, FileNotFoundError):
                continue
            if base_rgb.shape[:2] != wrist_rgb.shape[:2]:
                continue
            candidates.append(
                {
                    "episode": episode.identifier,
                    "episode_name": episode.episode_name,
                    "frame_index": int(row_index),
                    "timestamp": float(row["ts"]),
                    "task": episode.task,
                    "base_image": relative_path(base_path),
                    "wrist_image": relative_path(wrist_path),
                    "joint_positions_rad": joints,
                    "gripper_mm": gripper,
                    "source_resolution": [int(base_rgb.shape[1]), int(base_rgb.shape[0])],
                    "quality": frame_quality(base_rgb, wrist_rgb),
                }
            )
    return candidates


def select_diverse_frames(
    candidates: list[dict[str, Any]], calibration_count: int = 12, validation_count: int = 4
) -> list[dict[str, Any]]:
    total = calibration_count + validation_count
    if len(candidates) < total:
        raise RuntimeError(f"Only {len(candidates)} valid candidates for {total} requested frames")
    qualities = np.asarray([item["quality"] for item in candidates], dtype=np.float64)
    cutoff = float(np.quantile(qualities, 0.25))
    pool = [item for item in candidates if item["quality"] >= cutoff]
    poses = np.asarray([item["joint_positions_rad"] for item in pool], dtype=np.float64)
    scale = np.std(poses, axis=0)
    scale[scale < 0.05] = 0.05
    normalized = (poses - np.mean(poses, axis=0)) / scale
    selected: list[int] = [int(np.argmax([item["quality"] for item in pool]))]
    selected_episodes = {pool[selected[0]]["episode"]}
    while len(selected) < total:
        distances = np.min(
            np.linalg.norm(normalized[:, None, :] - normalized[selected][None, :, :], axis=2), axis=1
        )
        for index in selected:
            distances[index] = -1.0
        for index, item in enumerate(pool):
            if item["episode"] not in selected_episodes:
                distances[index] += 0.75
        choice = int(np.argmax(distances))
        selected.append(choice)
        selected_episodes.add(pool[choice]["episode"])
    output = []
    for order, index in enumerate(selected):
        item = dict(pool[index])
        item["sample_id"] = f"sample_{order:03d}"
        item["split"] = "calibration" if order < calibration_count else "validation"
        output.append(item)
    return output


def camera_axes(position: Iterable[float], target: Iterable[float], roll_deg: float = 0.0) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    forward = target_array - position_array
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        raise ValueError("Camera position and target cannot be identical")
    forward /= norm
    camera_z = -forward
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, camera_z))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    camera_x = np.cross(up, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    roll = math.radians(float(roll_deg))
    rolled_x = math.cos(roll) * camera_x + math.sin(roll) * camera_y
    rolled_y = -math.sin(roll) * camera_x + math.cos(roll) * camera_y
    return np.column_stack((rolled_x, rolled_y, camera_z))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(-1))
    return quat


def set_camera_parameters(model: mujoco.MjModel, camera_name: str, parameters: dict[str, Any]) -> None:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise RuntimeError(f"Camera not found: {camera_name}")
    model.cam_pos[camera_id] = np.asarray(parameters["position"], dtype=np.float64)
    rotation = camera_axes(parameters["position"], parameters["target"], parameters.get("roll_deg", 0.0))
    model.cam_quat[camera_id] = matrix_to_quaternion(rotation)
    model.cam_fovy[camera_id] = float(parameters["fovy_deg"])


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Joint not found: {joint_name}")
    data.qpos[model.jnt_qposadr[joint_id]] = value


class CalibrationRenderer:
    def __init__(self, width: int, height: int, model_path: Path = MODEL_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.config = load_config()

    def close(self) -> None:
        self.renderer.close()

    def render(
        self,
        sample: dict[str, Any],
        camera_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        for index, value in enumerate(sample["joint_positions_rad"], start=1):
            set_joint_qpos(self.model, self.data, f"joint{index}", float(value))
        set_raw_gripper_configuration(
            self.model,
            self.data,
            float(sample["gripper_mm"]),
            self.config,
        )
        if parameters is not None:
            set_camera_parameters(self.model, camera_name, parameters)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera_name)
        return np.asarray(self.renderer.render(), dtype=np.uint8).copy()


def edge_map(image: np.ndarray, width: int = 160, height: int = 120) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lower = int(max(25, 0.55 * median))
    upper = int(min(220, max(lower + 30, 1.35 * median)))
    return cv2.Canny(gray, lower, upper) > 0


def geometric_loss(real_rgb: np.ndarray, simulated_rgb: np.ndarray, camera_name: str) -> float:
    real_edges = edge_map(real_rgb)
    sim_edges = edge_map(simulated_rgb)
    if np.count_nonzero(sim_edges) < 20:
        return 10.0
    distance_to_real = cv2.distanceTransform((~real_edges).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_sim = cv2.distanceTransform((~sim_edges).astype(np.uint8), cv2.DIST_L2, 3)
    sim_to_real = float(np.mean(distance_to_real[sim_edges])) / 20.0
    real_to_sim = float(np.mean(distance_to_sim[real_edges])) / 20.0
    density_penalty = abs(float(np.mean(sim_edges)) - float(np.mean(real_edges)))
    loss = 0.78 * sim_to_real + 0.22 * real_to_sim + 0.8 * density_penalty
    real_gray = cv2.cvtColor(cv2.resize(real_rgb, (160, 120)), cv2.COLOR_RGB2GRAY)
    sim_gray = cv2.cvtColor(cv2.resize(simulated_rgb, (160, 120)), cv2.COLOR_RGB2GRAY)
    dark_area_mismatch = abs(float(np.mean(real_gray < 55)) - float(np.mean(sim_gray < 55)))
    loss += (0.65 if camera_name == "wrist_camera" else 0.20) * dark_area_mismatch
    if camera_name == "wrist_camera":
        columns = slice(12, 148)
        real_top = np.argmax(real_edges[:75, columns], axis=0)
        sim_top = np.argmax(sim_edges[:75, columns], axis=0)
        valid = (np.any(real_edges[:75, columns], axis=0) & np.any(sim_edges[:75, columns], axis=0))
        if np.any(valid):
            loss += 0.18 * float(np.mean(np.abs(real_top[valid] - sim_top[valid]))) / 75.0
    return float(loss)


def average_loss(
    renderer: CalibrationRenderer,
    samples: list[dict[str, Any]],
    camera_name: str,
    parameters: dict[str, Any],
    real_images: dict[str, np.ndarray],
) -> float:
    losses = []
    for sample in samples:
        simulated = renderer.render(sample, camera_name, parameters)
        losses.append(geometric_loss(real_images[sample["sample_id"]], simulated, camera_name))
    distance = float(np.linalg.norm(np.asarray(parameters["position"]) - np.asarray(parameters["target"])))
    preferred_distance = 0.78 if camera_name == "base_camera" else 0.20
    distance_penalty = 0.12 * abs(distance - preferred_distance)
    return float(np.mean(losses) + distance_penalty)


def parameter_vector(parameters: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [*parameters["position"], *parameters["target"], parameters.get("roll_deg", 0.0), parameters["fovy_deg"]],
        dtype=np.float64,
    )


def vector_parameters(vector: np.ndarray, template: dict[str, Any]) -> dict[str, Any]:
    result = dict(template)
    result["position"] = [float(value) for value in vector[:3]]
    result["target"] = [float(value) for value in vector[3:6]]
    result["roll_deg"] = float(vector[6])
    result["fovy_deg"] = float(vector[7])
    return result


def optimization_bounds(camera_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera_name == "base_camera":
        lower = np.array([0.80, -0.40, 0.40, 0.05, -0.20, 0.00, -15.0, 35.0])
        upper = np.array([1.40, 0.40, 1.00, 0.45, 0.20, 0.30, 15.0, 70.0])
        heuristic = np.array([1.00, 0.0, 0.60, 0.20, 0.0, 0.18, 0.0, 50.0])
    else:
        lower = np.array([0.06, -0.03, 0.075, -0.03, -0.03, 0.30, -110.0, 60.0])
        upper = np.array([0.10, 0.03, 0.11, 0.03, 0.03, 0.40, -70.0, 95.0])
        heuristic = np.array([0.08, 0.0, 0.09, 0.0, 0.0, 0.34, -90.0, 80.0])
    return lower, upper, heuristic


def optimize_camera(
    renderer: CalibrationRenderer,
    samples: list[dict[str, Any]],
    camera_name: str,
    initial: dict[str, Any],
    trials: int = 120,
    seed: int = 7,
    bounds_override: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    lower, upper, heuristic = bounds_override or optimization_bounds(camera_name)
    real_key = "base_image" if camera_name == "base_camera" else "wrist_image"
    real_images = {sample["sample_id"]: load_rgb(project_path(sample[real_key])) for sample in samples}
    initial_vector = np.clip(parameter_vector(initial), lower, upper)
    candidates = [initial_vector, heuristic]
    for _ in range(trials):
        candidates.append(rng.uniform(lower, upper))
    best_vector = candidates[0]
    best_loss = float("inf")
    history = []
    for index, vector in enumerate(candidates):
        parameters = vector_parameters(vector, initial)
        loss = average_loss(renderer, samples, camera_name, parameters, real_images)
        history.append({"stage": "coarse", "trial": index, "loss": loss})
        if loss < best_loss:
            best_loss = loss
            best_vector = vector.copy()
    steps = 0.12 * (upper - lower)
    for pass_index in range(4):
        improved = True
        while improved:
            improved = False
            for dimension in range(len(best_vector)):
                for direction in (-1.0, 1.0):
                    candidate = best_vector.copy()
                    candidate[dimension] = np.clip(
                        candidate[dimension] + direction * steps[dimension], lower[dimension], upper[dimension]
                    )
                    loss = average_loss(
                        renderer, samples, camera_name, vector_parameters(candidate, initial), real_images
                    )
                    history.append(
                        {"stage": "local", "pass": pass_index, "dimension": dimension, "loss": loss}
                    )
                    if loss + 1e-7 < best_loss:
                        best_loss = loss
                        best_vector = candidate
                        improved = True
            if len(history) > trials + 180:
                improved = False
        steps *= 0.45
    result = vector_parameters(best_vector, initial)
    diagnostics = {
        "initial_loss": average_loss(renderer, samples, camera_name, initial, real_images),
        "optimized_loss": best_loss,
        "trials": len(history),
        "seed": seed,
        "objective": "weighted symmetric Canny-edge distance with density and wrist orientation-profile penalties",
        "history": history,
    }
    return result, diagnostics


def blend_overlay(real: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    if simulated.shape[:2] != real.shape[:2]:
        simulated = cv2.resize(simulated, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(real, 0.55, simulated, 0.45, 0.0)


def edge_overlay(real: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    real_edges = edge_map(real, real.shape[1], real.shape[0])
    sim_edges = edge_map(simulated, real.shape[1], real.shape[0])
    output = np.zeros_like(real)
    output[real_edges] = (0, 255, 0)
    output[sim_edges] = (255, 0, 255)
    output[real_edges & sim_edges] = (255, 255, 255)
    return output


def policy_image(native_rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    render_config = config["render"]
    return np.asarray(
        image_tools.resize_with_pad(
            native_rgb, int(render_config["policy_height"]), int(render_config["policy_width"])
        ),
        dtype=np.uint8,
    )


def render_comparisons(
    samples: list[dict[str, Any]],
    before_config: dict[str, Any],
    after_config: dict[str, Any],
) -> dict[str, Any]:
    render_config = after_config["render"]
    width = int(render_config["native_width"])
    height = int(render_config["native_height"])
    renderer = CalibrationRenderer(width, height)
    records = []
    try:
        for sample in samples:
            sample_record = {"sample_id": sample["sample_id"], "split": sample["split"], "cameras": {}}
            for camera_name, real_key in (("base_camera", "base_image"), ("wrist_camera", "wrist_image")):
                short = camera_name.replace("_camera", "")
                real = load_rgb(project_path(sample[real_key]))
                before = renderer.render(sample, camera_name, before_config[camera_name])
                after = renderer.render(sample, camera_name, after_config[camera_name])
                before_real_path = CALIBRATION_ROOT / "before" / f"{sample['sample_id']}_real_{short}.png"
                before_sim_path = CALIBRATION_ROOT / "before" / f"{sample['sample_id']}_sim_{short}.png"
                after_real_path = CALIBRATION_ROOT / "after" / f"{sample['sample_id']}_real_{short}.png"
                after_sim_path = CALIBRATION_ROOT / "after" / f"{sample['sample_id']}_sim_{short}.png"
                save_rgb(before_real_path, real)
                save_rgb(before_sim_path, before)
                save_rgb(after_real_path, real)
                save_rgb(after_sim_path, after)
                save_rgb(CALIBRATION_ROOT / "after" / f"{sample['sample_id']}_policy_{short}.png", policy_image(after, after_config))
                save_rgb(CALIBRATION_ROOT / "overlays" / f"{sample['sample_id']}_{short}.png", blend_overlay(real, after))
                save_rgb(CALIBRATION_ROOT / "overlays" / f"{sample['sample_id']}_{short}_edges.png", edge_overlay(real, after))
                sample_record["cameras"][camera_name] = {
                    "before_loss": geometric_loss(real, before, camera_name),
                    "after_loss": geometric_loss(real, after, camera_name),
                }
            records.append(sample_record)
    finally:
        renderer.close()
    create_contact_sheets(samples)
    return {"samples": records}


def render_current_comparisons(
    samples: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    render_config = config["render"]
    renderer = CalibrationRenderer(
        int(render_config["native_width"]),
        int(render_config["native_height"]),
    )
    try:
        for sample in samples:
            for camera_name, real_key in (("base_camera", "base_image"), ("wrist_camera", "wrist_image")):
                short = camera_name.replace("_camera", "")
                real = load_rgb(project_path(sample[real_key]))
                simulated = renderer.render(sample, camera_name, config[camera_name])
                save_rgb(CALIBRATION_ROOT / "after" / f"{sample['sample_id']}_real_{short}.png", real)
                save_rgb(CALIBRATION_ROOT / "after" / f"{sample['sample_id']}_sim_{short}.png", simulated)
                save_rgb(
                    CALIBRATION_ROOT / "after" / f"{sample['sample_id']}_policy_{short}.png",
                    policy_image(simulated, config),
                )
                save_rgb(
                    CALIBRATION_ROOT / "overlays" / f"{sample['sample_id']}_{short}.png",
                    blend_overlay(real, simulated),
                )
                save_rgb(
                    CALIBRATION_ROOT / "overlays" / f"{sample['sample_id']}_{short}_edges.png",
                    edge_overlay(real, simulated),
                )
    finally:
        renderer.close()
    create_contact_sheets(samples)


def create_contact_sheets(samples: list[dict[str, Any]]) -> None:
    output_dir = CALIBRATION_ROOT / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera in ("base", "wrist"):
        rows = []
        for sample in samples:
            real = Image.open(CALIBRATION_ROOT / "after" / f"{sample['sample_id']}_real_{camera}.png").convert("RGB")
            sim = Image.open(CALIBRATION_ROOT / "after" / f"{sample['sample_id']}_sim_{camera}.png").convert("RGB")
            overlay = Image.open(CALIBRATION_ROOT / "overlays" / f"{sample['sample_id']}_{camera}.png").convert("RGB")
            triplet = Image.new("RGB", (480, 120), "white")
            for index, image in enumerate((real, sim, overlay)):
                image.thumbnail((160, 120))
                triplet.paste(image, (index * 160, 0))
            draw = ImageDraw.Draw(triplet)
            draw.rectangle((0, 0, 145, 16), fill="black")
            draw.text((3, 2), f"{sample['sample_id']} {sample['split']}", fill="white")
            rows.append(triplet)
        sheet = Image.new("RGB", (480, 120 * len(rows)), "white")
        for index, row in enumerate(rows):
            sheet.paste(row, (0, 120 * index))
        sheet.save(output_dir / f"{camera}_real_sim_overlay.png")


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("calibration", "validation"):
        output[split] = {}
        split_records = [record for record in records if record["split"] == split]
        for camera_name in CAMERA_KEYS:
            before = [record["cameras"][camera_name]["before_loss"] for record in split_records]
            after = [record["cameras"][camera_name]["after_loss"] for record in split_records]
            output[split][camera_name] = {
                "sample_count": len(split_records),
                "before_geometric_loss": float(np.mean(before)),
                "after_geometric_loss": float(np.mean(after)),
                "relative_improvement_percent": float(100.0 * (np.mean(before) - np.mean(after)) / np.mean(before)),
            }
    return output


def verify_raw_snapshot(episodes: list[Episode]) -> dict[str, Any]:
    files = list(RAW_ROOT.rglob("*"))
    regular = [path for path in files if path.is_file()]
    return {
        "episode_count": len(episodes),
        "file_count": len(regular),
        "png_count": sum(path.suffix.lower() == ".png" for path in regular),
        "csv_count": sum(path.suffix.lower() == ".csv" for path in regular),
        "json_count": sum(path.suffix.lower() == ".json" for path in regular),
        "total_bytes": sum(path.stat().st_size for path in regular),
    }
