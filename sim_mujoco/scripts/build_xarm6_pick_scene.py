from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_MODEL = PROJECT_ROOT / "sim_mujoco" / "assets" / "xarm6" / "xarm6_arm.xml"

OUTPUT_MODEL = PROJECT_ROOT / "sim_mujoco" / "assets" / "xarm6" / "xarm6_pick_scene.xml"

CAMERA_CONFIG = PROJECT_ROOT / "sim_mujoco" / "config" / "camera_calibration.yaml"


HOME_ARM_QPOS = [
    0.0,
    -0.6,
    -1.2,
    0.0,
    1.8,
    0.0,
]

MENAGERIE_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"
MENAGERIE_GRIPPER_MESH_DIR = "../../gripper/xarm"
MENAGERIE_DRIVER_RANGE = "0 0.85"
PROJECT_OPEN_DRIVER_ANGLE = 0.005
PROJECT_OPEN_CTRL = 100.0 * PROJECT_OPEN_DRIVER_ANGLE / 0.333

OBJECT_INITIAL_QPOS = [
    0.458,  # x
    -0.205,  # y
    0.063,  # z
    1.0,  # quaternion w
    0.0,  # quaternion x
    0.0,  # quaternion y
    0.0,  # quaternion z
]


def format_values(values) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(f"{value:.10g}" for value in array)


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body

    raise RuntimeError(f"Body not found: {name}")


def camera_xyaxes(
    position,
    target,
    up=(0.0, 0.0, 1.0),
    roll_deg=0.0,
) -> str:
    """Return MuJoCo xyaxes for a camera looking from position to target."""

    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - position
    norm = np.linalg.norm(forward)

    if norm < 1e-8:
        raise ValueError("Camera position and target cannot be identical.")

    forward /= norm

    # MuJoCo camera looks along its local negative Z axis.
    camera_z = -forward

    if abs(np.dot(up, camera_z)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    camera_x = np.cross(up, camera_z)
    camera_x /= np.linalg.norm(camera_x)

    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)

    roll = np.deg2rad(float(roll_deg))
    rolled_x = np.cos(roll) * camera_x + np.sin(roll) * camera_y
    rolled_y = -np.sin(roll) * camera_x + np.cos(roll) * camera_y

    return format_values(np.concatenate([rolled_x, rolled_y]))


def load_camera_config() -> dict:
    if not CAMERA_CONFIG.is_file():
        raise FileNotFoundError(f"Camera calibration config not found: {CAMERA_CONFIG}")

    with CAMERA_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    for camera_name in ("base_camera", "wrist_camera"):
        section = config.get(camera_name, {})
        for key in ("position", "target", "roll_deg", "fovy_deg"):
            if key not in section:
                raise ValueError(f"Missing {camera_name}.{key} in {CAMERA_CONFIG}")

    return config


def add_material(
    asset: ET.Element,
    name: str,
    rgba: str,
) -> None:
    if asset.find(f"./material[@name='{name}']") is None:
        ET.SubElement(
            asset,
            "material",
            name=name,
            rgba=rgba,
        )


def add_free_block(
    worldbody: ET.Element,
    *,
    body_name: str,
    position: tuple[float, float, float],
    half_size: float,
    material: str,
    mass: float,
    enabled: bool = False,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        name=body_name,
        pos=format_values(position),
    )
    ET.SubElement(body, "freejoint", name=f"{body_name}_freejoint")
    attributes = {
        "name": f"{body_name}_geom",
        "type": "box",
        "size": format_values([half_size] * 3),
        "mass": format_values(mass),
        "material": material,
        "contype": "1",
        "conaffinity": "1",
        "friction": "1.2 0.01 0.001",
    }
    if not enabled:
        attributes["rgba"] = "1 1 1 0"
    ET.SubElement(body, "geom", **attributes)


def add_pepper_geoms(
    body: ET.Element,
    *,
    prefix: str,
    enabled: bool,
) -> None:
    hidden_rgba = None if enabled else "0.82 0.035 0.035 0"
    lobe_positions = (
        (0.011, 0.000, 0.000),
        (-0.011, 0.000, 0.000),
        (0.000, 0.011, 0.000),
        (0.000, -0.011, 0.000),
    )
    for index, position in enumerate(lobe_positions):
        attributes = {
            "name": f"{prefix}_lobe_{index}",
            "type": "ellipsoid",
            "pos": format_values(position),
            "size": "0.016 0.016 0.022",
            "mass": "0.004",
            "material": "pepper_red_material",
            # Compile all free-pepper contacts so the Pick profile can use the
            # lobed acquisition surface. The Place reset profile disables the
            # overlapping contacts before stepping physics.
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.3 0.02 0.002",
        }
        if hidden_rgba is not None:
            attributes["rgba"] = hidden_rgba
        ET.SubElement(body, "geom", **attributes)

    # The visible overlapping lobes reproduce the pepper silhouette but make
    # an unreliable non-convex pinch surface: the fingers can momentarily lift
    # a lobe and then lose every contact as the body rotates.  A single convex
    # torso surface represents the pepper flesh for sustained normal-contact
    # grasping.  It is neither attached to the gripper nor pose-controlled.
    if prefix == "red_pepper":
        torso_attributes = {
            "name": "red_pepper_grasp_collision",
            "type": "ellipsoid",
            "size": "0.022 0.022 0.021",
            "mass": "0",
            "material": "pepper_red_material",
            "contype": "1",
            "conaffinity": "1",
            "condim": "4",
            "friction": "2.0 0.02 0.002",
        }
        if hidden_rgba is not None:
            torso_attributes["rgba"] = hidden_rgba
        ET.SubElement(body, "geom", **torso_attributes)

    stem_attributes = {
        "name": f"{prefix}_stem",
        "type": "capsule",
        "pos": "0 0 0.026",
        "size": "0.005 0.009",
        "mass": "0.001",
        "material": "pepper_green_material",
        "contype": "1",
        "conaffinity": "1",
        "friction": "1.0 0.01 0.001",
    }
    if not enabled:
        stem_attributes["rgba"] = "0.08 0.34 0.10 0"
    ET.SubElement(body, "geom", **stem_attributes)


def add_ring_wall(body: ET.Element, segments: int = 16, enabled: bool = False) -> None:
    radius = 0.058
    for index in range(segments):
        angle_a = 2.0 * np.pi * index / segments
        angle_b = 2.0 * np.pi * (index + 1) / segments
        point_a = (radius * np.cos(angle_a), radius * np.sin(angle_a), 0.012)
        point_b = (radius * np.cos(angle_b), radius * np.sin(angle_b), 0.012)
        attributes = {
            "name": f"ring_wall_{index:02d}",
            "type": "capsule",
            "fromto": format_values([*point_a, *point_b]),
            "size": "0.005",
            "material": "ring_material",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.0 0.01 0.001",
        }
        if not enabled:
            attributes["rgba"] = "0.92 0.92 0.90 0"
        ET.SubElement(body, "geom", **attributes)


def add_collision_geom(
    body: ET.Element,
    *,
    name: str,
    geom_type: str,
    size: str,
    pos: str | None = None,
    fromto: str | None = None,
) -> None:
    attributes = {
        "name": name,
        "type": geom_type,
        "size": size,
        "mass": "0",
        "rgba": "0.2 0.8 0.2 0",
        "group": "3",
        "contype": "1",
        "conaffinity": "1",
        "friction": "0.8 0.01 0.001",
    }
    if pos is not None:
        attributes["pos"] = pos
    if fromto is not None:
        attributes["fromto"] = fromto
    ET.SubElement(body, "geom", **attributes)


def add_robot_collision_model(root: ET.Element) -> None:
    # Primitive dimensions are conservative approximations of the compiled
    # visual-mesh bounds in each link's local frame.
    add_collision_geom(
        find_body(root, "link_base"),
        name="link_base_collision",
        geom_type="cylinder",
        pos="0 0.01 0.08",
        size="0.08 0.08",
    )
    add_collision_geom(
        find_body(root, "link1"),
        name="link1_collision",
        geom_type="ellipsoid",
        pos="0 -0.002 -0.003",
        size="0.050 0.074 0.096",
    )

    link2 = find_body(root, "link2")
    add_collision_geom(
        link2,
        name="link2_shaft_collision",
        geom_type="capsule",
        fromto="0 0 -0.15 0 0 0.11",
        size="0.060",
    )
    add_collision_geom(
        link2,
        name="link2_upper_collision",
        geom_type="ellipsoid",
        pos="0 0.004 0.13",
        size="0.064 0.088 0.070",
    )
    add_collision_geom(
        link2,
        name="link2_lower_collision",
        geom_type="ellipsoid",
        pos="0 0.004 -0.16",
        size="0.064 0.088 0.065",
    )

    add_collision_geom(
        find_body(root, "link3"),
        name="link3_collision",
        geom_type="capsule",
        fromto="-0.004 0.004 -0.06 -0.004 0.004 0.075",
        size="0.068",
    )
    add_collision_geom(
        find_body(root, "link4"),
        name="link4_collision",
        geom_type="capsule",
        fromto="0 -0.004 -0.068 0 -0.004 0.064",
        size="0.052",
    )
    add_collision_geom(
        find_body(root, "link5"),
        name="link5_collision",
        geom_type="capsule",
        fromto="0 0 -0.038 0 0 0.028",
        size="0.056",
    )
    add_collision_geom(
        find_body(root, "link6"),
        name="link6_collision",
        geom_type="ellipsoid",
        pos="0 0 0.006",
        size="0.018 0.040 0.047",
    )

    contact = root.find("contact")
    if contact is None:
        contact = ET.Element("contact")
        actuator = root.find("actuator")
        if actuator is None:
            root.append(contact)
        else:
            root.insert(list(root).index(actuator), contact)

    adjacent_pairs = (
        ("link_base", "link1"),
        ("link1", "link2"),
        ("link2", "link3"),
        ("link3", "link4"),
        ("link4", "link5"),
        ("link5", "link6"),
        ("link4", "link6"),
        ("link6", "xarm_gripper_base_link"),
        ("link4", "xarm_gripper_base_link"),
        ("link5", "xarm_gripper_base_link"),
    )
    for body1, body2 in adjacent_pairs:
        ET.SubElement(
            contact,
            "exclude",
            name=f"exclude_{body1}_{body2}",
            body1=body1,
            body2=body2,
        )


def add_menagerie_gripper_assets(asset: ET.Element) -> None:
    """Add the exact UFACTORY mesh subset used by pinned Menagerie hand.xml."""

    meshes = {
        "xarm_gripper_base_mesh": "base_link.stl",
        "left_outer_knuckle_mesh": "left_outer_knuckle.stl",
        "left_finger_mesh": "left_finger.stl",
        "left_inner_knuckle_mesh": "left_inner_knuckle.stl",
        "right_outer_knuckle_mesh": "right_outer_knuckle.stl",
        "right_finger_mesh": "right_finger.stl",
        "right_inner_knuckle_mesh": "right_inner_knuckle.stl",
    }
    for name, filename in meshes.items():
        ET.SubElement(
            asset,
            "mesh",
            name=name,
            file=f"{MENAGERIE_GRIPPER_MESH_DIR}/{filename}",
        )


def add_menagerie_joint(
    body: ET.Element,
    *,
    name: str,
    axis: str,
    kind: str,
) -> None:
    attributes = {
        "name": name,
        "type": "hinge",
        "axis": axis,
        "range": MENAGERIE_DRIVER_RANGE,
        "armature": "0.1",
        "frictionloss": "1",
    }
    if kind == "driver":
        attributes.update(
            armature="0.005",
            damping="0.1",
            solreflimit="0.005 1",
        )
    elif kind == "follower":
        attributes["solreflimit"] = "0.005 1"
    elif kind == "spring_link":
        attributes.update(
            stiffness="0.05",
            springref="2.62",
            damping="0.00125",
        )
    else:
        raise ValueError(f"Unknown Menagerie gripper joint kind: {kind}")
    ET.SubElement(body, "joint", **attributes)


def add_menagerie_mesh_geom(
    body: ET.Element,
    *,
    name: str,
    mesh: str,
    material: str,
    visual_only: bool = False,
) -> None:
    attributes = {
        "name": name,
        "type": "mesh",
        "mesh": mesh,
        "material": material,
    }
    if visual_only:
        attributes.update(contype="0", conaffinity="0", group="2")
    else:
        attributes.update(contype="1", conaffinity="1", group="0")
    ET.SubElement(body, "geom", **attributes)


def add_menagerie_pad(
    body: ET.Element,
    *,
    name: str,
    position: str,
    friction: str,
    rgba: str,
) -> None:
    ET.SubElement(
        body,
        "geom",
        name=name,
        type="box",
        pos=position,
        size="0.015 0.002 0.0095",
        friction=friction,
        solimp="0.95 0.99 0.001",
        solref="0.004 1",
        mass="0",
        priority="1",
        contype="1",
        conaffinity="1",
        group="3",
        rgba=rgba,
    )


def add_menagerie_gripper(
    root: ET.Element,
    *,
    link6: ET.Element,
    asset: ET.Element,
    actuator: ET.Element,
    camera_config: dict,
) -> None:
    """Port pinned Menagerie hand.xml mechanics onto the existing xArm6 flange.

    Numeric hand parameters are copied from Menagerie revision
    ``da76818e...``.  The direct, identity-frame attachment to xArm6 link6 is
    the sole kinematic integration choice; it preserves this project's
    existing tool/camera frame while matching UFACTORY's xArm6 URDF mount.
    """

    add_menagerie_gripper_assets(asset)
    gripper_base = ET.SubElement(
        link6,
        "body",
        name="xarm_gripper_base_link",
    )
    ET.SubElement(
        gripper_base,
        "inertial",
        pos="-0.00065489 -0.0018497 0.048028",
        quat="0.997403 -0.0717512 -0.0061836 0.000477479",
        mass="0.54156",
        diaginertia="0.000471093 0.000332307 0.000254799",
    )
    add_menagerie_mesh_geom(
        gripper_base,
        name="xarm_gripper_base_collision",
        mesh="xarm_gripper_base_mesh",
        material="menagerie_white",
    )

    ET.SubElement(
        gripper_base,
        "site",
        name="tool_center_point",
        pos="0 0 0.172",
        size="0.012",
        rgba="0 1 0 0",
    )
    wrist_config = camera_config["wrist_camera"]
    wrist_camera_position = np.asarray(wrist_config["position"], dtype=np.float64)
    wrist_camera_target = np.asarray(wrist_config["target"], dtype=np.float64)
    ET.SubElement(
        gripper_base,
        "camera",
        name="wrist_camera",
        pos=format_values(wrist_camera_position),
        xyaxes=camera_xyaxes(
            wrist_camera_position,
            wrist_camera_target,
            roll_deg=wrist_config["roll_deg"],
        ),
        fovy=format_values(wrist_config["fovy_deg"]),
    )
    held_pepper = ET.SubElement(
        gripper_base,
        "body",
        name="held_red_pepper",
        pos="0 0 -1",
    )
    add_pepper_geoms(held_pepper, prefix="held_pepper", enabled=False)

    left_outer = ET.SubElement(
        gripper_base,
        "body",
        name="left_outer_knuckle",
        pos="0 0.035 0.059098",
    )
    ET.SubElement(
        left_outer,
        "inertial",
        pos="0 0.021559 0.015181",
        quat="0.47789 0.87842 0 0",
        mass="0.033618",
        diaginertia="1.9111e-05 1.79089e-05 1.90167e-06",
    )
    add_menagerie_joint(
        left_outer,
        name="left_driver_joint",
        axis="1 0 0",
        kind="driver",
    )
    add_menagerie_mesh_geom(
        left_outer,
        name="left_outer_knuckle_collision",
        mesh="left_outer_knuckle_mesh",
        material="menagerie_black",
    )
    left_finger = ET.SubElement(
        left_outer,
        "body",
        name="left_finger",
        pos="0 0.035465 0.042039",
    )
    ET.SubElement(
        left_finger,
        "inertial",
        pos="0 -0.016413 0.029258",
        quat="0.697634 0.115353 -0.115353 0.697634",
        mass="0.048304",
        diaginertia="1.88037e-05 1.7493e-05 3.56792e-06",
    )
    add_menagerie_joint(
        left_finger,
        name="left_finger_joint",
        axis="-1 0 0",
        kind="follower",
    )
    add_menagerie_mesh_geom(
        left_finger,
        name="left_finger_visual",
        mesh="left_finger_mesh",
        material="menagerie_black",
        visual_only=True,
    )
    add_menagerie_pad(
        left_finger,
        name="left_finger_pad_1",
        position="0 -0.024003 0.032",
        friction="0.7",
        rgba="0.0 0.1 0.7 1",
    )
    add_menagerie_pad(
        left_finger,
        name="left_finger_pad_2",
        position="0 -0.024003 0.050",
        friction="0.6",
        rgba="0.0 0.5 0.5 1",
    )

    left_inner = ET.SubElement(
        gripper_base,
        "body",
        name="left_inner_knuckle",
        pos="0 0.02 0.074098",
    )
    ET.SubElement(
        left_inner,
        "inertial",
        pos="1.86601e-06 0.0220468 0.0261335",
        quat="0.664139 -0.242732 0.242713 0.664146",
        mass="0.0230126",
        diaginertia="8.34216e-06 6.0949e-06 2.75601e-06",
    )
    add_menagerie_joint(
        left_inner,
        name="left_inner_knuckle_joint",
        axis="1 0 0",
        kind="spring_link",
    )
    add_menagerie_mesh_geom(
        left_inner,
        name="left_inner_knuckle_collision",
        mesh="left_inner_knuckle_mesh",
        material="menagerie_black",
    )

    right_outer = ET.SubElement(
        gripper_base,
        "body",
        name="right_outer_knuckle",
        pos="0 -0.035 0.059098",
    )
    ET.SubElement(
        right_outer,
        "inertial",
        pos="0 -0.021559 0.015181",
        quat="0.87842 0.47789 0 0",
        mass="0.033618",
        diaginertia="1.9111e-05 1.79089e-05 1.90167e-06",
    )
    add_menagerie_joint(
        right_outer,
        name="right_driver_joint",
        axis="-1 0 0",
        kind="driver",
    )
    add_menagerie_mesh_geom(
        right_outer,
        name="right_outer_knuckle_collision",
        mesh="right_outer_knuckle_mesh",
        material="menagerie_black",
    )
    right_finger = ET.SubElement(
        right_outer,
        "body",
        name="right_finger",
        pos="0 -0.035465 0.042039",
    )
    ET.SubElement(
        right_finger,
        "inertial",
        pos="0 0.016413 0.029258",
        quat="0.697634 -0.115356 0.115356 0.697634",
        mass="0.048304",
        diaginertia="1.88038e-05 1.7493e-05 3.56779e-06",
    )
    add_menagerie_joint(
        right_finger,
        name="right_finger_joint",
        axis="1 0 0",
        kind="follower",
    )
    add_menagerie_mesh_geom(
        right_finger,
        name="right_finger_visual",
        mesh="right_finger_mesh",
        material="menagerie_black",
        visual_only=True,
    )
    add_menagerie_pad(
        right_finger,
        name="right_finger_pad_1",
        position="0 0.024003 0.032",
        friction="0.7",
        rgba="0.0 0.1 0.7 1",
    )
    add_menagerie_pad(
        right_finger,
        name="right_finger_pad_2",
        position="0 0.024003 0.050",
        friction="0.6",
        rgba="0.0 0.5 0.5 1",
    )

    right_inner = ET.SubElement(
        gripper_base,
        "body",
        name="right_inner_knuckle",
        pos="0 -0.02 0.074098",
    )
    ET.SubElement(
        right_inner,
        "inertial",
        pos="1.866e-06 -0.022047 0.026133",
        quat="0.66415 0.242702 -0.242721 0.664144",
        mass="0.023013",
        diaginertia="8.34209e-06 6.0949e-06 2.75601e-06",
    )
    add_menagerie_joint(
        right_inner,
        name="right_inner_knuckle_joint",
        axis="-1 0 0",
        kind="spring_link",
    )
    add_menagerie_mesh_geom(
        right_inner,
        name="right_inner_knuckle_collision",
        mesh="right_inner_knuckle_mesh",
        material="menagerie_black",
    )

    tendon = root.find("tendon")
    if tendon is None:
        tendon = ET.SubElement(root, "tendon")
    split = ET.SubElement(tendon, "fixed", name="split")
    ET.SubElement(split, "joint", joint="right_driver_joint", coef="0.5")
    ET.SubElement(split, "joint", joint="left_driver_joint", coef="0.5")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "connect",
        name="right_finger_connect",
        anchor="0 0.015 0.015",
        body1="right_finger",
        body2="right_inner_knuckle",
        solref="0.005 1",
    )
    ET.SubElement(
        equality,
        "connect",
        name="left_finger_connect",
        anchor="0 -0.015 0.015",
        body1="left_finger",
        body2="left_inner_knuckle",
        solref="0.005 1",
    )
    ET.SubElement(
        equality,
        "joint",
        name="driver_symmetry",
        joint1="left_driver_joint",
        joint2="right_driver_joint",
        polycoef="0 1 0 0 0",
        solref="0.005 1",
    )
    ET.SubElement(
        actuator,
        "general",
        name="gripper_actuator",
        tendon="split",
        biastype="affine",
        forcerange="-50 50",
        ctrlrange="0 255",
        gainprm="0.333",
        biasprm="0 -100 -10",
    )

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for body1, body2 in (
        ("right_inner_knuckle", "right_outer_knuckle"),
        ("right_inner_knuckle", "right_finger"),
        ("left_inner_knuckle", "left_outer_knuckle"),
        ("left_inner_knuckle", "left_finger"),
        ("left_inner_knuckle", "xarm_gripper_base_link"),
        ("right_inner_knuckle", "xarm_gripper_base_link"),
        ("left_outer_knuckle", "xarm_gripper_base_link"),
        ("right_outer_knuckle", "xarm_gripper_base_link"),
    ):
        ET.SubElement(
            contact,
            "exclude",
            name=f"menagerie_exclude_{body1}_{body2}",
            body1=body1,
            body2=body2,
        )


def main() -> None:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(f"Source arm model not found: {SOURCE_MODEL}")

    camera_config = load_camera_config()
    tree = ET.parse(SOURCE_MODEL)
    root = tree.getroot()

    root.set("model", "xarm6_pick_scene")

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    keyframe = root.find("keyframe")

    if asset is None:
        raise RuntimeError("MJCF has no <asset> element.")
    if worldbody is None:
        raise RuntimeError("MJCF has no <worldbody> element.")
    if actuator is None:
        raise RuntimeError("MJCF has no <actuator> element.")
    if keyframe is None:
        raise RuntimeError("MJCF has no <keyframe> element.")

    for light in worldbody.iter("light"):
        light.set("castshadow", "false")

    add_material(asset, name="menagerie_white", rgba="1 1 1 1")
    add_material(asset, name="menagerie_black", rgba="0.149 0.149 0.149 1")
    add_material(
        asset,
        name="table_material",
        rgba="0.62 0.64 0.64 1",
    )
    add_material(
        asset,
        name="object_material",
        rgba="0.85 0.12 0.10 1",
    )
    add_material(
        asset,
        name="blue_block_material",
        rgba="0.05 0.55 0.78 1",
    )
    add_material(
        asset,
        name="pepper_red_material",
        rgba="0.82 0.035 0.035 1",
    )
    add_material(
        asset,
        name="pepper_green_material",
        rgba="0.08 0.34 0.10 1",
    )
    add_material(
        asset,
        name="ring_material",
        rgba="0.92 0.92 0.90 1",
    )

    link6 = find_body(root, "link6")
    for link_index in range(1, 7):
        find_body(root, f"link{link_index}").set("gravcomp", "1")

    add_menagerie_gripper(
        root,
        link6=link6,
        asset=asset,
        actuator=actuator,
        camera_config=camera_config,
    )

    # ------------------------------------------------------------------
    # Table and object
    # ------------------------------------------------------------------

    ET.SubElement(
        worldbody,
        "geom",
        name="table",
        type="box",
        pos="0.45 0 0.025",
        size="0.35 0.45 0.025",
        material="table_material",
        contype="1",
        conaffinity="1",
        friction="1.0 0.01 0.001",
    )

    object_body = ET.SubElement(
        worldbody,
        "body",
        name="object",
        pos="0.458 -0.205 0.063",
    )

    ET.SubElement(
        object_body,
        "freejoint",
        name="object_freejoint",
    )

    ET.SubElement(
        object_body,
        "geom",
        name="object_geom",
        type="box",
        size="0.013 0.013 0.013",
        mass="0.012",
        material="object_material",
        contype="1",
        conaffinity="1",
        friction="1.2 0.01 0.001",
    )

    add_free_block(
        worldbody,
        body_name="blue_block",
        position=(0.454, -0.204, 0.0685),
        half_size=0.0185,
        material="blue_block_material",
        mass=0.024,
    )
    add_free_block(
        worldbody,
        body_name="small_block",
        position=(0.462, -0.160, 0.063),
        half_size=0.013,
        material="object_material",
        mass=0.012,
    )
    add_free_block(
        worldbody,
        body_name="large_block",
        position=(0.456, -0.194, 0.0685),
        half_size=0.0185,
        material="blue_block_material",
        mass=0.024,
    )

    pepper_body = ET.SubElement(
        worldbody,
        "body",
        name="red_pepper",
        pos="0.492 -0.178 0.072",
    )
    ET.SubElement(pepper_body, "freejoint", name="red_pepper_freejoint")
    add_pepper_geoms(pepper_body, prefix="red_pepper", enabled=False)

    ring_body = ET.SubElement(
        worldbody,
        "body",
        name="ring",
        pos="0 0 -1",
    )
    ET.SubElement(
        ring_body,
        "geom",
        name="ring_bottom",
        type="cylinder",
        size="0.061 0.002",
        mass="0.05",
        material="ring_material",
        rgba="0.92 0.92 0.90 0",
        contype="1",
        conaffinity="1",
        friction="1.0 0.01 0.001",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="camera_backdrop",
        type="box",
        pos="-0.20 0 0.75",
        size="0.02 1.20 0.75",
        rgba="0.82 0.83 0.83 1",
        contype="0",
        conaffinity="0",
    )
    add_ring_wall(ring_body)

    add_robot_collision_model(root)

    # ------------------------------------------------------------------
    # Fixed base camera
    # ------------------------------------------------------------------

    base_config = camera_config["base_camera"]
    base_camera_position = np.asarray(base_config["position"], dtype=np.float64)
    base_camera_target = np.asarray(base_config["target"], dtype=np.float64)

    ET.SubElement(
        worldbody,
        "camera",
        name="base_camera",
        pos=format_values(base_camera_position),
        xyaxes=camera_xyaxes(
            base_camera_position,
            base_camera_target,
            roll_deg=base_config["roll_deg"],
        ),
        fovy=format_values(base_config["fovy_deg"]),
    )

    # ------------------------------------------------------------------
    # Update home keyframe
    #
    # qpos order:
    #   arm 6
    #   6 Menagerie gripper hinge joints
    #   object freejoint: xyz + quaternion
    #
    # ctrl order:
    #   arm actuators 6
    #   gripper actuator 1
    # ------------------------------------------------------------------

    home_key = keyframe.find("./key[@name='home']")

    if home_key is None:
        raise RuntimeError("Home keyframe not found.")

    catalog_object_qpos = [
        *OBJECT_INITIAL_QPOS,
        0.45,
        0.00,
        -1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.43,
        -0.10,
        -1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.48,
        0.10,
        -1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.45,
        0.00,
        -1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    full_home_qpos = (
        HOME_ARM_QPOS + [PROJECT_OPEN_DRIVER_ANGLE] * 6 + catalog_object_qpos
    )

    full_home_ctrl = HOME_ARM_QPOS + [PROJECT_OPEN_CTRL]

    home_key.set(
        "qpos",
        format_values(full_home_qpos),
    )
    home_key.set(
        "ctrl",
        format_values(full_home_ctrl),
    )

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ET.indent(root, space="  ")

    tree.write(
        OUTPUT_MODEL,
        encoding="utf-8",
        xml_declaration=True,
    )

    print("Generated:", OUTPUT_MODEL)
    print("Arm actuators: 6")
    print("Gripper actuators: 1")
    print(f"Menagerie gripper revision: {MENAGERIE_REVISION}")
    print(
        "Gripper architecture: 6 hinges, 1 fixed tendon, 3 equalities, 1 affine actuator"
    )


if __name__ == "__main__":
    main()
