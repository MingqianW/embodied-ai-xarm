from __future__ import annotations

from typing import Any

import mujoco


ROBOT_BODY_NAMES = {
    "link_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_base",
    "left_finger",
    "right_finger",
}
SUPPORT_GEOM_NAMES = {"table", "floor"}
SUPPORT_PENETRATION_TOLERANCE_M = 1e-4


def _name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
    fallback: str,
) -> str:
    value = mujoco.mj_id2name(model, object_type, int(object_id))
    return fallback if value is None else str(value)


def collision_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, Any]:
    contacts: list[dict[str, Any]] = []
    forbidden_contacts: list[dict[str, Any]] = []

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        geom1_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1, f"geom_{geom1}")
        geom2_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2, f"geom_{geom2}")
        body1_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body1, "world")
        body2_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body2, "world")

        robot1 = body1_name in ROBOT_BODY_NAMES
        robot2 = body2_name in ROBOT_BODY_NAMES
        support1 = geom1_name in SUPPORT_GEOM_NAMES
        support2 = geom2_name in SUPPORT_GEOM_NAMES
        kind = "allowed_contact"
        forbidden = False

        if robot1 and robot2 and body1_name != body2_name:
            kind = "self_collision"
            forbidden = True
        elif (support1 and robot2) or (support2 and robot1):
            robot_body = body2_name if support1 else body1_name
            support_geom = geom1_name if support1 else geom2_name
            robot_geom = geom2_name if support1 else geom1_name
            mounted_base_contact = robot_body == "link_base" and support_geom == "floor"
            tabletop_fingertip_contact = (
                support_geom == "table"
                and robot_geom in {"left_fingertip_pad", "right_fingertip_pad"}
            )
            if (
                not mounted_base_contact
                and not tabletop_fingertip_contact
                and float(contact.dist) < -SUPPORT_PENETRATION_TOLERANCE_M
            ):
                kind = "robot_support_collision"
                forbidden = True
            elif not mounted_base_contact:
                kind = "robot_support_contact"
        elif robot1 or robot2:
            kind = "robot_object_contact"

        record = {
            "contact_index": contact_index,
            "kind": kind,
            "forbidden": forbidden,
            "geom1": geom1_name,
            "geom2": geom2_name,
            "body1": body1_name,
            "body2": body2_name,
            "distance_m": float(contact.dist),
        }
        contacts.append(record)
        if forbidden:
            forbidden_contacts.append(record)

    kinds: dict[str, int] = {}
    for contact in contacts:
        kind = str(contact["kind"])
        kinds[kind] = kinds.get(kind, 0) + 1

    termination_reason = None
    if forbidden_contacts:
        termination_reason = (
            "self_collision"
            if forbidden_contacts[0]["kind"] == "self_collision"
            else "robot_support_collision"
        )
    return {
        "contact_count": len(contacts),
        "contact_kind_counts": kinds,
        "forbidden": bool(forbidden_contacts),
        "forbidden_contact_count": len(forbidden_contacts),
        "termination_reason": termination_reason,
        "minimum_distance_m": min(
            (float(contact["distance_m"]) for contact in contacts),
            default=None,
        ),
        "contacts": contacts,
        "forbidden_contacts": forbidden_contacts,
    }
