from __future__ import annotations

import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

XARM_DESCRIPTION = (
    PROJECT_ROOT
    / "third_party"
    / "xarm_ros2"
    / "xarm_description"
)

MESH_DIR = (
    XARM_DESCRIPTION
    / "meshes"
    / "xarm6"
    / "visual"
)

KINEMATICS_PATH = (
    XARM_DESCRIPTION
    / "config"
    / "kinematics"
    / "default"
    / "xarm6_default_kinematics.yaml"
)

INERTIA_PATH = (
    XARM_DESCRIPTION
    / "config"
    / "link_inertial"
    / "xarm6_type6_HT_BR2.yaml"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "assets"
    / "xarm6"
    / "xarm6_arm.xml"
)


JOINT_LIMITS = {
    "joint1": (-2.0 * math.pi, 2.0 * math.pi),
    "joint2": (-2.059, 2.0944),
    "joint3": (-3.927, 0.19198),
    "joint4": (-2.0 * math.pi, 2.0 * math.pi),
    "joint5": (-1.69297, math.pi),
    "joint6": (-2.0 * math.pi, 2.0 * math.pi),
}

JOINT_EFFORTS = {
    "joint1": 50.0,
    "joint2": 50.0,
    "joint3": 32.0,
    "joint4": 32.0,
    "joint5": 32.0,
    "joint6": 20.0,
}

ACTUATOR_KP = {
    "joint1": 120.0,
    "joint2": 120.0,
    "joint3": 100.0,
    "joint4": 70.0,
    "joint5": 50.0,
    "joint6": 30.0,
}

HOME_QPOS = [
    0.0,
    -0.6,
    -1.2,
    0.0,
    1.8,
    0.0,
]


def format_values(*values: float) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    require_file(KINEMATICS_PATH)
    require_file(INERTIA_PATH)

    mesh_names = [
        "link_base",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
    ]

    for mesh_name in mesh_names:
        require_file(MESH_DIR / f"{mesh_name}.stl")

    with KINEMATICS_PATH.open("r", encoding="utf-8") as file:
        kinematics = yaml.safe_load(file)["kinematics"]

    with INERTIA_PATH.open("r", encoding="utf-8") as file:
        inertial_parameters = yaml.safe_load(file)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    relative_mesh_dir = Path(
        os.path.relpath(MESH_DIR, OUTPUT_PATH.parent)
    ).as_posix()

    root = ET.Element("mujoco", model="xarm6_arm_test")

    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        autolimits="true",
        meshdir=relative_mesh_dir,
    )

    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        azimuth="135",
        elevation="-20",
    )
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.35 0.35 0.35",
        diffuse="0.6 0.6 0.6",
        specular="0.2 0.2 0.2",
    )

    defaults = ET.SubElement(root, "default")

    ET.SubElement(
        defaults,
        "joint",
        damping="1.0",
        armature="0.01",
    )

    # First-stage model: mesh geoms are visual only.
    # Simplified collision geoms will be added later.
    ET.SubElement(
        defaults,
        "geom",
        contype="0",
        conaffinity="0",
        group="2",
    )

    assets = ET.SubElement(root, "asset")

    ET.SubElement(
        assets,
        "material",
        name="robot_white",
        rgba="0.86 0.86 0.88 1",
    )

    ET.SubElement(
        assets,
        "material",
        name="robot_silver",
        rgba="0.55 0.57 0.6 1",
    )

    for mesh_name in mesh_names:
        ET.SubElement(
            assets,
            "mesh",
            name=f"{mesh_name}_mesh",
            file=f"{mesh_name}.stl",
        )

    worldbody = ET.SubElement(root, "worldbody")

    ET.SubElement(
        worldbody,
        "light",
        name="main_light",
        pos="1 -1 2.5",
        dir="-0.3 0.3 -1",
    )

    ET.SubElement(
        worldbody,
        "camera",
        name="overview_camera",
        pos="1.4 -1.4 1.1",
        xyaxes="0.707 0.707 0 -0.36 0.36 0.86",
        fovy="50",
    )

    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.1",
        rgba="0.75 0.75 0.75 1",
        contype="1",
        conaffinity="1",
        friction="1 0.01 0.001",
    )

    base_body = ET.SubElement(
        worldbody,
        "body",
        name="link_base",
        pos="0 0 0",
    )

    ET.SubElement(
        base_body,
        "geom",
        name="link_base_visual",
        type="mesh",
        mesh="link_base_mesh",
        material="robot_white",
    )

    parent_body = base_body

    for index in range(1, 7):
        joint_name = f"joint{index}"
        link_name = f"link{index}"

        transform = kinematics[joint_name]

        body = ET.SubElement(
            parent_body,
            "body",
            name=link_name,
            pos=format_values(
                transform["x"],
                transform["y"],
                transform["z"],
            ),
            euler=format_values(
                transform["roll"],
                transform["pitch"],
                transform["yaw"],
            ),
        )

        lower, upper = JOINT_LIMITS[joint_name]

        ET.SubElement(
            body,
            "joint",
            name=joint_name,
            type="hinge",
            axis="0 0 1",
            limited="true",
            range=format_values(lower, upper),
        )

        inertial = inertial_parameters[link_name]
        origin = inertial["origin"]
        inertia = inertial["inertia"]

        ET.SubElement(
            body,
            "inertial",
            pos=format_values(
                origin["x"],
                origin["y"],
                origin["z"],
            ),
            mass=format_values(inertial["mass"]),
            fullinertia=format_values(
                inertia["ixx"],
                inertia["iyy"],
                inertia["izz"],
                inertia["ixy"],
                inertia["ixz"],
                inertia["iyz"],
            ),
        )

        material = (
            "robot_silver"
            if index == 6
            else "robot_white"
        )

        ET.SubElement(
            body,
            "geom",
            name=f"{link_name}_visual",
            type="mesh",
            mesh=f"{link_name}_mesh",
            material=material,
        )

        parent_body = body

    ET.SubElement(
        parent_body,
        "site",
        name="end_effector_site",
        pos="0 0 0",
        size="0.01",
        rgba="1 0 0 1",
    )

    actuators = ET.SubElement(root, "actuator")

    for index in range(1, 7):
        joint_name = f"joint{index}"
        lower, upper = JOINT_LIMITS[joint_name]
        effort = JOINT_EFFORTS[joint_name]

        ET.SubElement(
            actuators,
            "position",
            name=f"{joint_name}_actuator",
            joint=joint_name,
            kp=format_values(ACTUATOR_KP[joint_name]),
            ctrlrange=format_values(lower, upper),
            forcelimited="true",
            forcerange=format_values(-effort, effort),
        )

    keyframes = ET.SubElement(root, "keyframe")

    ET.SubElement(
        keyframes,
        "key",
        name="home",
        qpos=format_values(*HOME_QPOS),
        ctrl=format_values(*HOME_QPOS),
    )

    ET.indent(root, space="  ")

    ET.ElementTree(root).write(
        OUTPUT_PATH,
        encoding="utf-8",
        xml_declaration=True,
    )

    print("Generated:", OUTPUT_PATH)
    print("Mesh directory:", MESH_DIR)
    print("Kinematics:", KINEMATICS_PATH)
    print("Inertial parameters:", INERTIA_PATH)


if __name__ == "__main__":
    main()