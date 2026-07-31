from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim_isaac.articulation import IsaacArticulation, RobotMapping
from sim_isaac.object_spawning import ObjectSpawner, TaskConfig
from sim_isaac.version_compat import (
    add_neutral_dome_light,
    add_usd_reference,
    bind_preview_surface_material,
    create_articulation,
    create_dynamic_cuboid,
    create_fixed_cuboid,
    create_world,
    set_rigid_body_gravity,
)


@dataclass
class IsaacScene:
    world: Any
    robot: IsaacArticulation
    objects: ObjectSpawner
    table: Any
    placement_surface: Any | None = None


def build_scene(
    *,
    robot_asset_path: Path,
    mapping: RobotMapping,
    task: TaskConfig,
    physics_hz: float,
    rendering_hz: float,
) -> IsaacScene:
    if not robot_asset_path.is_file():
        raise FileNotFoundError(
            f"Generated xArm USD is missing: {robot_asset_path}. "
            "Run sim_isaac/scripts/prepare_xarm_asset.py from Isaac Sim's python.bat."
        )
    world = create_world(
        physics_dt=1.0 / float(physics_hz),
        rendering_dt=1.0 / float(rendering_hz),
    )
    add_neutral_dome_light(world.stage)
    # Isaac's convenience ground plane references an online NVIDIA asset.
    # Author a local fixed cuboid so headless validation remains fully offline.
    world.scene.add(
        create_fixed_cuboid(
            "/World/ground",
            name="ground",
            position=np.asarray([0.0, 0.0, -0.01], dtype=np.float32),
            scale=np.asarray([4.0, 4.0, 0.02], dtype=np.float32),
            color=np.asarray([0.55, 0.57, 0.60], dtype=np.float32),
        )
    )
    add_usd_reference(robot_asset_path, mapping.articulation_prim_path)
    set_rigid_body_gravity(
        world.stage,
        mapping.articulation_prim_path,
        enabled=mapping.gravity_enabled,
    )
    bind_preview_surface_material(
        world.stage,
        f"{mapping.articulation_prim_path}/{mapping.gripper_visual_frame}",
        material_path="/World/Looks/xarm_gripper_black",
        diffuse_color=mapping.gripper_color_rgb,
    )
    robot_prim = create_articulation(
        mapping.articulation_prim_path,
        initialize=False,
    )
    world.scene.add(robot_prim)
    table = world.scene.add(
        create_fixed_cuboid(
            "/World/table",
            name="table",
            position=task.table_position_m,
            scale=task.table_size_m,
            color=np.asarray([0.45, 0.32, 0.20], dtype=np.float32),
        )
    )
    target = world.scene.add(
        create_dynamic_cuboid(
            f"/World/{task.object_name}",
            name=task.object_name,
            position=task.object_position_m,
            orientation=task.object_orientation_wxyz,
            scale=task.object_size_m,
            color=task.target_color_rgb,
            mass=task.object_mass_kg,
        )
    )
    distractor_prims = []
    for spec in task.distractors:
        distractor_prims.append(
            world.scene.add(
                create_dynamic_cuboid(
                    f"/World/{spec.name}",
                    name=spec.name,
                    position=np.asarray([0.45, 0.0, -1.0], dtype=np.float32),
                    orientation=spec.orientation_wxyz,
                    scale=spec.size_m,
                    color=spec.color_rgb,
                    mass=spec.mass_kg,
                )
            )
        )
    placement_surface = None
    if task.place_in_ring:
        placement_surface = world.scene.add(
            create_fixed_cuboid(
                "/World/color_ring",
                name="color_ring",
                position=np.asarray([0.48, 0.08, 0.057], dtype=np.float32),
                scale=np.asarray([0.18, 0.18, 0.01], dtype=np.float32),
                color=np.asarray([0.95, 0.95, 0.95], dtype=np.float32),
            )
        )
    world.reset()
    robot = IsaacArticulation(mapping, robot_prim)
    return IsaacScene(
        world=world,
        robot=robot,
        objects=ObjectSpawner(target, task, tuple(distractor_prims)),
        table=table,
        placement_surface=placement_surface,
    )
