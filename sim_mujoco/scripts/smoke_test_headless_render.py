"""Policy-free smoke test for the active MuJoCo scene and both policy cameras."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.environment import MuJoCoEnvironment
from sim_mujoco.paths import active_model_path, mujoco_output_root
from sim_mujoco.remote_policy_observation import BASE_CAMERA, WRIST_CAMERA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="red_block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=mujoco_output_root() / "headless_render_smoke",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "active_xml": str(active_model_path()),
        "task": args.task,
        "seed": args.seed,
        "cameras": [BASE_CAMERA, WRIST_CAMERA],
        "images": {},
    }
    with MuJoCoEnvironment(task=args.task) as environment:
        environment.reset(seed=args.seed)
        frames = {
            BASE_CAMERA: environment.context.renderer,
            WRIST_CAMERA: environment.context.renderer,
        }
        for camera_name in frames:
            from sim_mujoco.remote_policy_observation import render_native_rgb

            image = render_native_rgb(
                environment.context.renderer,
                environment.context.data,
                camera_name,
            )
            if image.shape != (480, 640, 3):
                raise RuntimeError(f"{camera_name} shape is {image.shape}, expected (480, 640, 3)")
            if image.dtype != np.uint8:
                raise RuntimeError(f"{camera_name} dtype is {image.dtype}, expected uint8")
            if not np.isfinite(image).all():
                raise RuntimeError(f"{camera_name} contains non-finite values")
            output = args.output_dir / f"{camera_name}.png"
            Image.fromarray(image).save(output)
            report["images"][camera_name] = {
                "path": str(output),
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "min": int(image.min()),
                "max": int(image.max()),
            }

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
