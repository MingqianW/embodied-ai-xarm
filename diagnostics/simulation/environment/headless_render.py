"""Policy-free smoke test for the active MuJoCo scene and both policy cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


from simulation.environment import MuJoCoEnvironment
from simulation.resources import output_root
from simulation.observation.cameras import render_rgb
from simulation.resources import model_path
from simulation.robot.model import BASE_CAMERA_NAME
from simulation.robot.model import WRIST_CAMERA_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="red_block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root() / "headless_render_smoke",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "active_xml": str(model_path()),
        "task": args.task,
        "seed": args.seed,
        "cameras": [BASE_CAMERA_NAME, WRIST_CAMERA_NAME],
        "images": {},
    }
    with MuJoCoEnvironment(task=args.task) as environment:
        environment.reset(seed=args.seed)
        frames = {
            BASE_CAMERA_NAME: environment.context.renderer,
            WRIST_CAMERA_NAME: environment.context.renderer,
        }
        for camera_name in frames:
            image = render_rgb(
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
