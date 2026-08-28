from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.remote_policy_observation import (
    BASE_CAMERA,
    WRIST_CAMERA,
    initialize_scene,
    load_simulation,
    render_native_rgb,
)
from sim_mujoco.task_scenes import configure_task_scene, task_names
from sim_mujoco.paths import mujoco_output_root


DEFAULT_OUTPUT_DIR = mujoco_output_root() / "task_scene_preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*task_names(), "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-xy-range", type=float, default=0.0)
    parser.add_argument("--object-yaw-range-deg", type=float, default=0.0)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument("--show-collisions", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = task_names() if args.task == "all" else (args.task,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for task in selected:
        context = load_simulation()
        try:
            initialize_scene(context.model, context.data, settle_steps=0)
            runtime, initial = configure_task_scene(
                context.model,
                context.data,
                task=task,
                seed=args.seed,
                object_xy_range=args.object_xy_range,
                object_yaw_range_deg=args.object_yaw_range_deg,
                joint_noise=args.joint_noise,
            )
            if args.show_collisions:
                for geom_id in range(context.model.ngeom):
                    geom_name = mujoco.mj_id2name(
                        context.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    ) or ""
                    if geom_name.endswith("_visual"):
                        context.model.geom_rgba[geom_id, 3] = 0.25
                    if (
                        geom_name.endswith("_collision")
                        or "finger_pad_" in geom_name
                    ):
                        context.model.geom_group[geom_id] = 0
                        context.model.geom_rgba[geom_id] = [0.1, 0.9, 0.2, 0.55]
            task_dir = args.output_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                render_native_rgb(context.renderer, context.data, BASE_CAMERA)
            ).save(task_dir / "base.png")
            Image.fromarray(
                render_native_rgb(context.renderer, context.data, WRIST_CAMERA)
            ).save(task_dir / "wrist.png")
            if args.show_collisions:
                for geom_id in range(context.model.ngeom):
                    geom_name = mujoco.mj_id2name(
                        context.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    ) or ""
                    if geom_name.endswith("_visual"):
                        context.model.geom_rgba[geom_id, 3] = 0.0
                Image.fromarray(
                    render_native_rgb(
                        context.renderer,
                        context.data,
                        "overview_camera",
                    )
                ).save(task_dir / "overview_collisions.png")
            entry = {
                "task": task,
                "prompt": runtime.prompt,
                "initial_conditions": initial,
                "task_metrics": runtime.metrics(),
                "base_image": str(task_dir / "base.png"),
                "wrist_image": str(task_dir / "wrist.png"),
            }
            if args.show_collisions:
                entry["overview_collision_image"] = str(
                    task_dir / "overview_collisions.png"
                )
            (task_dir / "scene.json").write_text(
                json.dumps(entry, indent=2),
                encoding="utf-8",
            )
            manifest.append(entry)
            print(f"rendered {task}: {task_dir}")
        finally:
            context.close()

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
