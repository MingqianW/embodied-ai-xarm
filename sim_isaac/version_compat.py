from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


def create_simulation_app(
    *,
    headless: bool,
    width: int = 1280,
    height: int = 720,
    renderer: str = "RayTracedLighting",
    anti_aliasing: int = 1,
) -> Any:
    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": bool(headless),
            "width": int(width),
            "height": int(height),
            "renderer": str(renderer),
            "anti_aliasing": int(anti_aliasing),
        }
    )


def create_world(*, physics_dt: float, rendering_dt: float) -> Any:
    """Create the stable Core World facade shipped with Isaac Sim 4.5-6.x."""

    try:
        from isaacsim.core.api.world import World
    except ImportError:
        from isaacsim.core.api import World

    if World.instance() is not None:
        World.instance().clear_instance()
    return World(
        physics_dt=float(physics_dt),
        rendering_dt=float(rendering_dt),
        stage_units_in_meters=1.0,
    )


def add_neutral_dome_light(
    stage: Any,
    *,
    prim_path: str = "/World/ambient_light",
    intensity: float = 500.0,
) -> Any:
    """Author uniform fill light so wrist observations are not fully shadowed."""

    from pxr import Gf, UsdLux

    light = UsdLux.DomeLight.Define(stage, prim_path)
    light.GetIntensityAttr().Set(float(intensity))
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    return light


def add_usd_reference(usd_path: Path, prim_path: str) -> Any:
    try:
        from isaacsim.core.experimental.utils.stage import add_reference_to_stage

        return add_reference_to_stage(usd_path=str(usd_path), path=prim_path)
    except (ImportError, TypeError):
        from isaacsim.core.utils.stage import add_reference_to_stage

        return add_reference_to_stage(usd_path=str(usd_path), prim_path=prim_path)


def set_rigid_body_gravity(
    stage: Any,
    root_prim_path: str,
    *,
    enabled: bool,
) -> tuple[str, ...]:
    """Set gravity consistently on every rigid body below a referenced asset."""

    from pxr import PhysxSchema, Usd, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(
            f"Cannot configure rigid-body gravity; prim does not exist: {root_prim_path}"
        )

    configured_paths = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid_body_api.CreateDisableGravityAttr(not bool(enabled))
        configured_paths.append(str(prim.GetPath()))

    if not configured_paths:
        raise RuntimeError(
            "Cannot configure rigid-body gravity; no rigid bodies were found below "
            f"{root_prim_path}"
        )
    return tuple(configured_paths)


def bind_preview_surface_material(
    stage: Any,
    prim_path: str,
    *,
    material_path: str,
    diffuse_color: tuple[float, float, float],
    roughness: float = 0.65,
) -> Any:
    """Bind one strong visual material to an asset subtree."""

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot bind visual material; prim does not exist: {prim_path}")
    if len(diffuse_color) != 3:
        raise ValueError("Preview-surface diffuse color must contain three values")

    material_parent = material_path.rsplit("/", 1)[0]
    UsdGeom.Scope.Define(stage, material_parent)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(float(value) for value in diffuse_color))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material, UsdShade.Tokens.strongerThanDescendants)
    return material


def create_fixed_cuboid(
    prim_path: str,
    *,
    name: str,
    position: np.ndarray,
    scale: np.ndarray,
    color: np.ndarray,
) -> Any:
    from isaacsim.core.api.objects import FixedCuboid

    return FixedCuboid(
        prim_path=prim_path,
        name=name,
        position=np.asarray(position, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        color=np.asarray(color, dtype=np.float32),
    )


def create_dynamic_cuboid(
    prim_path: str,
    *,
    name: str,
    position: np.ndarray,
    orientation: np.ndarray,
    scale: np.ndarray,
    color: np.ndarray,
    mass: float,
) -> Any:
    from isaacsim.core.api.objects import DynamicCuboid

    return DynamicCuboid(
        prim_path=prim_path,
        name=name,
        position=np.asarray(position, dtype=np.float32),
        orientation=np.asarray(orientation, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        color=np.asarray(color, dtype=np.float32),
        mass=float(mass),
    )


def import_urdf_to_usd(
    urdf_path: Path,
    usd_path: Path,
    options: dict[str, Any],
) -> Path:
    """Import after SimulationApp has started, supporting current and legacy APIs."""

    urdf_path = urdf_path.resolve()
    usd_path = usd_path.resolve()
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        robot_name = urdf_path.stem
        expected_package_dir = usd_path.parent
        if (
            expected_package_dir.name != robot_name
            or usd_path.stem != robot_name
            or usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}
        ):
            raise ValueError(
                "Isaac 6 URDF output must be configured as "
                f"<output>/{robot_name}/{robot_name}.usda, got {usd_path}"
            )
        output_root = expected_package_dir.parent
        with tempfile.TemporaryDirectory(
            prefix=f".{robot_name}_import_",
            dir=output_root,
        ) as staging_root:
            importer = URDFImporter(
                URDFImporterConfig(
                    urdf_path=str(urdf_path),
                    usd_path=str(staging_root),
                    merge_mesh=bool(options.get("merge_meshes", True)),
                    merge_fixed_joints=bool(options.get("merge_fixed_joints", False)),
                    allow_self_collision=bool(options.get("allow_self_collision", False)),
                    collision_from_visuals=bool(
                        options.get("collision_from_visuals", False)
                    ),
                    fix_base=bool(options.get("fix_base", True)),
                    robot_type=str(options.get("robot_type", "Manipulator")),
                    joint_drive_type=options.get("joint_drive_type"),
                    joint_target_type=options.get("joint_target_type"),
                    override_joint_stiffness=options.get(
                        "override_joint_stiffness"
                    ),
                    override_joint_damping=options.get("override_joint_damping"),
                    ros_package_paths=list(options.get("ros_package_paths", [])),
                )
            )
            staged_result = Path(importer.import_urdf()).resolve()
            if not staged_result.is_file() or staged_result.stat().st_size == 0:
                raise RuntimeError(
                    "Isaac URDF importer did not create a nonempty USD root layer: "
                    f"{staged_result}"
                )
            if expected_package_dir.exists():
                shutil.rmtree(expected_package_dir)
            # Copy into a normally-created target so Windows inherits the
            # repository ACL instead of retaining TemporaryDirectory's
            # intentionally restrictive ACL after a same-volume move.
            expected_package_dir.mkdir(parents=True)
            shutil.copytree(
                staged_result.parent,
                expected_package_dir,
                dirs_exist_ok=True,
            )
        result = usd_path
    except (ImportError, TypeError):
        import omni.kit.commands

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("Isaac URDFCreateImportConfig command failed")
        import_config.merge_fixed_joints = bool(options.get("merge_fixed_joints", False))
        import_config.import_inertia_tensor = bool(options.get("import_inertia_tensor", True))
        import_config.fix_base = bool(options.get("fix_base", True))
        import_config.collision_from_visuals = bool(
            options.get("collision_from_visuals", False)
        )
        status, result_value = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
            dest_path=str(usd_path),
        )
        if not status:
            raise RuntimeError("Isaac URDFParseAndImportFile command failed")
        result = Path(str(result_value or usd_path))
    if not result.exists() and usd_path.exists():
        result = usd_path
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError(f"Isaac reported URDF import success but no USD exists: {result}")
    return result


def create_articulation(prim_path: str, *, initialize: bool = True) -> Any:
    try:
        from isaacsim.core.prims import SingleArticulation
    except ImportError:
        from isaacsim.core.api.articulations import Articulation as SingleArticulation

    articulation = SingleArticulation(prim_path=prim_path, name="xarm6")
    if initialize and hasattr(articulation, "initialize"):
        articulation.initialize()
    return articulation


def apply_joint_position_targets(
    articulation: Any,
    targets: np.ndarray,
    indices: np.ndarray,
) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    articulation.apply_action(
        ArticulationAction(
            joint_positions=np.asarray(targets, dtype=np.float32),
            joint_indices=np.asarray(indices, dtype=np.int64),
        )
    )


def set_joint_positions(
    articulation: Any,
    positions: np.ndarray,
    indices: np.ndarray,
) -> None:
    articulation.set_joint_positions(
        np.asarray(positions, dtype=np.float32),
        joint_indices=np.asarray(indices, dtype=np.int64),
    )


def set_joint_velocities(
    articulation: Any,
    velocities: np.ndarray,
    indices: np.ndarray,
) -> None:
    articulation.set_joint_velocities(
        np.asarray(velocities, dtype=np.float32),
        joint_indices=np.asarray(indices, dtype=np.int64),
    )
