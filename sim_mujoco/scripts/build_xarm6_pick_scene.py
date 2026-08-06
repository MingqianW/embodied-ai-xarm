


from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_MODEL = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "assets"
    / "xarm6"
    / "xarm6_arm.xml"
)

OUTPUT_MODEL = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "assets"
    / "xarm6"
    / "xarm6_pick_scene.xml"
)

CAMERA_CONFIG = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "config"
    / "camera_calibration.yaml"
)


HOME_ARM_QPOS = [
    0.0,
    -0.6,
    -1.2,
    0.0,
    1.8,
    0.0,
]

# The official four-bar gripper reaches an approximately 88.5 mm inner
# aperture at raw position 845. The simplified slider needs this travel to
# reproduce that aperture after accounting for the fingertip pad offset.
GRIPPER_OPEN_HALF_WIDTH = 0.047

OBJECT_INITIAL_QPOS = [
    0.458, # x
    -0.205,# y
    0.063, # z
    1.0,   # quaternion w
    0.0,   # quaternion x
    0.0,   # quaternion y
    0.0,   # quaternion z
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

    return format_values(
        np.concatenate([rolled_x, rolled_y])
    )


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
        ("link6", "gripper_base"),
        ("link4", "gripper_base"),
        ("link5", "gripper_base"),
        ("gripper_base", "left_finger"),
        ("gripper_base", "right_finger"),
        ("left_finger", "right_finger"),
    )
    for body1, body2 in adjacent_pairs:
        ET.SubElement(
            contact,
            "exclude",
            name=f"exclude_{body1}_{body2}",
            body1=body1,
            body2=body2,
        )


def main() -> None:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(
            f"Source arm model not found: {SOURCE_MODEL}"
        )

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

    add_material(
        asset,
        name="gripper_dark",
        rgba="0.18 0.19 0.21 1",
    )
    add_material(
        asset,
        name="finger_pad",
        rgba="0.06 0.06 0.06 1",
    )
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

    # ------------------------------------------------------------------
    # Simplified xArm gripper
    # ------------------------------------------------------------------

    gripper_base = ET.SubElement(
        link6,
        "body",
        name="gripper_base",
        pos="0 0 0",
        gravcomp="1",
    )

    ET.SubElement(
        gripper_base,
        "geom",
        name="gripper_base_collision",
        type="cylinder",
        pos="0 0 0.050",
        size="0.043 0.050",
        mass="0.50",
        material="gripper_dark",
        contype="1",
        conaffinity="1",
        friction="0.8 0.01 0.001",
    )

    # 官方 gripper TCP 在 gripper base 之后约 0.172 m。
    ET.SubElement(
        gripper_base,
        "site",
        name="tool_center_point",
        pos="0 0 0.172",
        size="0.012",
        rgba="0 1 0 0",
    )

    # Wrist camera: located slightly to the side of the gripper,
    # looking forward along the tool direction.
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

    # Left finger
    left_finger = ET.SubElement(
        gripper_base,
        "body",
        name="left_finger",
        pos="0 0.008 0.105",
        gravcomp="1",
    )

    ET.SubElement(
        left_finger,
        "joint",
        name="left_finger_slide",
        type="slide",
        axis="0 1 0",
        limited="true",
        range=f"0 {GRIPPER_OPEN_HALF_WIDTH}",
        damping="4",
        armature="0.002",
    )

    ET.SubElement(
        left_finger,
        "geom",
        name="left_finger_collision",
        type="box",
        pos="0 0 0.017",
        size="0.016 0.006 0.017",
        mass="0.05",
        material="gripper_dark",
        contype="1",
        conaffinity="1",
        friction="1.2 0.01 0.001",
    )

    ET.SubElement(
        left_finger,
        "geom",
        name="left_fingertip_pad",
        type="box",
        pos="0 -0.0075 0.052",
        size="0.016 0.003 0.018",
        mass="0.01",
        material="finger_pad",
        contype="1",
        conaffinity="1",
        friction="2.0 0.02 0.002",
    )

    # Right finger
    right_finger = ET.SubElement(
        gripper_base,
        "body",
        name="right_finger",
        pos="0 -0.008 0.105",
        gravcomp="1",
    )

    ET.SubElement(
        right_finger,
        "joint",
        name="right_finger_slide",
        type="slide",
        axis="0 -1 0",
        limited="true",
        range=f"0 {GRIPPER_OPEN_HALF_WIDTH}",
        damping="4",
        armature="0.002",
    )

    ET.SubElement(
        right_finger,
        "geom",
        name="right_finger_collision",
        type="box",
        pos="0 0 0.017",
        size="0.016 0.006 0.017",
        mass="0.05",
        material="gripper_dark",
        contype="1",
        conaffinity="1",
        friction="1.2 0.01 0.001",
    )

    ET.SubElement(
        right_finger,
        "geom",
        name="right_fingertip_pad",
        type="box",
        pos="0 0.0075 0.052",
        size="0.016 0.003 0.018",
        mass="0.01",
        material="finger_pad",
        contype="1",
        conaffinity="1",
        friction="2.0 0.02 0.002",
    )

    # ------------------------------------------------------------------
    # One gripper actuator
    # ------------------------------------------------------------------

    ET.SubElement(
        actuator,
        "position",
        name="gripper_actuator",
        joint="left_finger_slide",
        kp="500",
        ctrlrange=f"0 {GRIPPER_OPEN_HALF_WIDTH}",
        forcelimited="true",
        forcerange="-40 40",
    )

    equality = root.find("equality")

    if equality is None:
        equality = ET.SubElement(root, "equality")

    # right_finger_slide = left_finger_slide
    ET.SubElement(
        equality,
        "joint",
        name="symmetric_gripper",
        joint1="right_finger_slide",
        joint2="left_finger_slide",
        polycoef="0 1 0 0 0",
        solref="0.01 1",
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
    #   left finger
    #   right finger
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
        0.45, 0.00, -1.0, 1.0, 0.0, 0.0, 0.0,
        0.43, -0.10, -1.0, 1.0, 0.0, 0.0, 0.0,
        0.48, 0.10, -1.0, 1.0, 0.0, 0.0, 0.0,
        0.45, 0.00, -1.0, 1.0, 0.0, 0.0, 0.0,
    ]
    full_home_qpos = (
        HOME_ARM_QPOS
        + [
            GRIPPER_OPEN_HALF_WIDTH,
            GRIPPER_OPEN_HALF_WIDTH,
        ]
        + catalog_object_qpos
    )

    full_home_ctrl = (
        HOME_ARM_QPOS
        + [GRIPPER_OPEN_HALF_WIDTH]
    )

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
    print("Gripper full opening: approximately 88.5 mm")


if __name__ == "__main__":
    main()
