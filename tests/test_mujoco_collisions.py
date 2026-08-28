from __future__ import annotations

import unittest

import mujoco

from sim_mujoco.collision import collision_diagnostics
from sim_mujoco.remote_policy_observation import initialize_scene, load_simulation
from sim_mujoco.task_scenes import configure_task_scene, task_names


def object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise AssertionError(f"Missing {object_type}: {name}")
    return int(value)


class CollisionModelTests(unittest.TestCase):
    def test_every_arm_link_has_enabled_collision_geometry(self) -> None:
        context = load_simulation()
        try:
            for body_name in ("link_base", *(f"link{i}" for i in range(1, 7))):
                body_id = object_id(
                    context.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_name,
                )
                collision_geoms = [
                    geom_id
                    for geom_id in range(context.model.ngeom)
                    if int(context.model.geom_bodyid[geom_id]) == body_id
                    and (
                        mujoco.mj_id2name(
                            context.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            geom_id,
                        )
                        or ""
                    ).endswith("_collision")
                ]
                self.assertTrue(collision_geoms, body_name)
                for geom_id in collision_geoms:
                    self.assertEqual(int(context.model.geom_contype[geom_id]), 1)
                    self.assertEqual(int(context.model.geom_conaffinity[geom_id]), 1)
        finally:
            context.close()

    def test_visual_meshes_remain_noncolliding(self) -> None:
        context = load_simulation()
        try:
            for link_index in range(1, 7):
                geom_id = object_id(
                    context.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"link{link_index}_visual",
                )
                self.assertEqual(int(context.model.geom_contype[geom_id]), 0)
                self.assertEqual(int(context.model.geom_conaffinity[geom_id]), 0)
        finally:
            context.close()

    def test_four_bar_fingertip_pads_are_collidable(self) -> None:
        context = load_simulation()
        try:
            for geom_name in (
                "left_finger_pad_1",
                "left_finger_pad_2",
                "right_finger_pad_1",
                "right_finger_pad_2",
            ):
                geom_id = object_id(context.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                self.assertEqual(int(context.model.geom_contype[geom_id]), 1, geom_name)
                self.assertEqual(int(context.model.geom_conaffinity[geom_id]), 1, geom_name)
        finally:
            context.close()

    def test_every_task_starts_without_forbidden_collision(self) -> None:
        for task in task_names():
            context = load_simulation()
            try:
                initialize_scene(context.model, context.data, settle_steps=0)
                configure_task_scene(
                    context.model,
                    context.data,
                    task=task,
                    seed=0,
                    object_xy_range=0.0,
                    object_yaw_range_deg=0.0,
                    joint_noise=0.0,
                    settle_steps=100,
                )
                diagnostics = collision_diagnostics(context.model, context.data)
                self.assertFalse(
                    diagnostics["forbidden"],
                    f"{task}: {diagnostics['forbidden_contacts']}",
                )
            finally:
                context.close()

    def test_robot_table_contact_is_forbidden(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <geom name="table" type="box" pos="0 0 0" size="1 1 0.05"/>
                <body name="link3" pos="0 0 0.08">
                  <freejoint/>
                  <geom name="link3_collision" type="sphere" size="0.06"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        diagnostics = collision_diagnostics(model, data)
        self.assertTrue(diagnostics["forbidden"])
        self.assertIn(
            "robot_support_collision",
            {item["kind"] for item in diagnostics["forbidden_contacts"]},
        )

    def test_nonadjacent_link_contact_is_self_collision(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="link1" pos="0 0 0.3">
                  <freejoint/>
                  <geom name="link1_collision" type="sphere" size="0.08"/>
                </body>
                <body name="link3" pos="0.1 0 0.3">
                  <freejoint/>
                  <geom name="link3_collision" type="sphere" size="0.08"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        diagnostics = collision_diagnostics(model, data)
        self.assertTrue(diagnostics["forbidden"])
        self.assertIn(
            "self_collision",
            {item["kind"] for item in diagnostics["forbidden_contacts"]},
        )


if __name__ == "__main__":
    unittest.main()
