from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_runtime.image_preprocessing import image_diagnostics
from policy_runtime.image_preprocessing import preprocess_policy_image
from policy_runtime.schemas import POLICY_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare policy-facing MuJoCo and Isaac base/wrist camera images."
    )
    parser.add_argument("--mujoco-base", type=Path)
    parser.add_argument("--mujoco-wrist", type=Path)
    parser.add_argument("--isaac-base", type=Path)
    parser.add_argument("--isaac-wrist", type=Path)
    parser.add_argument("--capture-mujoco", action="store_true")
    parser.add_argument("--capture-isaac", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    parser.add_argument(
        "--preprocess-inputs",
        action="store_true",
        help="Apply the shared OpenPI resize-with-pad path to all supplied images.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _capture_mujoco() -> dict[str, np.ndarray]:
    from sim_mujoco.environment import MuJoCoEnvironment

    with MuJoCoEnvironment() as environment:
        observation = environment.reset(seed=0)
        return {
            "base": observation.base_image,
            "wrist": observation.wrist_image,
        }


def _capture_isaac(headless: bool) -> dict[str, np.ndarray]:
    from sim_isaac.environment import IsaacEnvironment

    with IsaacEnvironment(headless=headless, seed=0) as environment:
        environment.require_safe("camera comparison")
        observation = environment.reset(seed=0)
        environment.require_safe("camera comparison reset")
        return {
            "base": observation.base_image,
            "wrist": observation.wrist_image,
        }


def _annotate(image: np.ndarray, points: list[list[float]]) -> np.ndarray:
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    for index, point in enumerate(points):
        x, y = float(point[0]), float(point[1])
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="yellow", width=2)
        draw.text((x + radius + 1, y - radius), str(index), fill="yellow")
    return np.asarray(canvas, dtype=np.uint8)


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape == reference.shape:
        return image
    return np.asarray(
        Image.fromarray(image).resize(
            (reference.shape[1], reference.shape[0]),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )


def _comparison(
    mujoco: np.ndarray,
    isaac: np.ndarray,
    *,
    alpha: float,
    landmarks: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mujoco_marked = _annotate(mujoco, landmarks.get("mujoco", []))
    isaac_marked = _annotate(isaac, landmarks.get("isaac", []))
    isaac_aligned = _resize_like(isaac_marked, mujoco_marked)
    height = max(mujoco_marked.shape[0], isaac_marked.shape[0])
    side = Image.new(
        "RGB",
        (mujoco_marked.shape[1] + isaac_marked.shape[1], height),
    )
    side.paste(Image.fromarray(mujoco_marked), (0, 0))
    side.paste(Image.fromarray(isaac_marked), (mujoco_marked.shape[1], 0))
    overlay = np.clip(
        (1.0 - alpha) * mujoco_marked.astype(np.float32)
        + alpha * isaac_aligned.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    absolute_difference = np.abs(
        mujoco_marked.astype(np.int16) - isaac_aligned.astype(np.int16)
    )
    metadata = {
        "mujoco": image_diagnostics(mujoco),
        "isaac": image_diagnostics(isaac),
        "isaac_resized_for_overlay": list(isaac_aligned.shape),
        "mean_absolute_pixel_difference": float(absolute_difference.mean()),
        "landmarks": landmarks,
    }
    return np.asarray(side, dtype=np.uint8), overlay, metadata


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path(
        os.environ.get("ISAAC_OUTPUT_DIR", "sim_isaac/output")
    ) / "camera_comparison"
    if not 0.0 <= args.overlay_alpha <= 1.0:
        print("ERROR: --overlay-alpha must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        mujoco = _capture_mujoco() if args.capture_mujoco else {
            "base": _load(args.mujoco_base) if args.mujoco_base else None,
            "wrist": _load(args.mujoco_wrist) if args.mujoco_wrist else None,
        }
        isaac = _capture_isaac(args.headless) if args.capture_isaac else {
            "base": _load(args.isaac_base) if args.isaac_base else None,
            "wrist": _load(args.isaac_wrist) if args.isaac_wrist else None,
        }
        if any(mujoco[name] is None or isaac[name] is None for name in ("base", "wrist")):
            raise ValueError(
                "Provide all four image paths or use --capture-mujoco/--capture-isaac"
            )
        if args.preprocess_inputs:
            mujoco = {
                name: preprocess_policy_image(image)
                for name, image in mujoco.items()
            }
            isaac = {
                name: preprocess_policy_image(image)
                for name, image in isaac.items()
            }
        landmark_data = (
            json.loads(args.landmarks.read_text(encoding="utf-8"))
            if args.landmarks
            else {}
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "overlay_alpha": args.overlay_alpha,
            "inputs_preprocessed": args.preprocess_inputs,
            "cameras": {},
        }
        for camera_name in ("base", "wrist"):
            side, overlay, metadata = _comparison(
                mujoco[camera_name],
                isaac[camera_name],
                alpha=args.overlay_alpha,
                landmarks=landmark_data.get(camera_name, {}),
            )
            Image.fromarray(side).save(output_dir / f"{camera_name}_side_by_side.png")
            Image.fromarray(overlay).save(output_dir / f"{camera_name}_overlay.png")
            report["cameras"][camera_name] = metadata
        (output_dir / "comparison.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
