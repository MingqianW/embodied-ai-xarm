from __future__ import annotations

import unittest

import mujoco
import numpy as np

from simulation.runtime import initialize_scene
from simulation.runtime import load_simulation
from simulation.scene import (
    configure_task_scene,
    resolve_task,
    task_names,
)


class TaskConfigTests(unittest.TestCase):
    def test_all_raw_tasks_resolve(self) -> None:
        aliases = {
            "pick_up_the_red_block": "red_block",
            "pick_up_the_blue_block": "blue_block",
            "pick up the largest block": "largest_block",
            "pick_up_the_smallest_block": "smallest_block",
            "pick_up_the_red_pepper": "red_pepper",
            "place_the_red_pepper_in_the_ring": "place_red_pepper_in_ring",
        }
        for alias, expected in aliases.items():
            self.assertEqual(resolve_task(alias)[0], expected)

    def test_catalog_covers_six_tasks(self) -> None:
        self.assertEqual(
            set(task_names()),
            {
                "red_block",
                "blue_block",
                "largest_block",
                "smallest_block",
                "red_pepper",
                "place_red_pepper_in_ring",
            },
        )


class TaskSceneRuntimeTests(unittest.TestCase):
    def make_scene(self, task: str, seed: int = 7):
        context = load_simulation()
        initialize_scene(context.model, context.data, settle_steps=0)
        runtime, initial = configure_task_scene(
            context.model,
            context.data,
            task=task,
            seed=seed,
            object_xy_range=0.03,
            object_yaw_range_deg=15.0,
            joint_noise=0.01,
            settle_steps=2,
        )
        return context, runtime, initial

    def test_each_task_compiles_and_has_finite_target(self) -> None:
        for task in task_names():
            context, runtime, initial = self.make_scene(task)
            try:
                self.assertEqual(initial["task"], task)
                self.assertTrue(np.isfinite(runtime.metrics()["target_position"]).all())
                self.assertTrue(
                    all(
                        item["visible"] for item in initial["wrist_visibility"].values()
                    )
                )
            finally:
                context.close()

    def test_required_bodies_stay_in_initial_wrist_view_across_seeds(self) -> None:
        for task in task_names():
            for seed in (0, 7, 19):
                context, _, initial = self.make_scene(task, seed=seed)
                try:
                    self.assertTrue(
                        all(
                            item["visible"]
                            for item in initial["wrist_visibility"].values()
                        ),
                        msg=f"{task} seed={seed}: {initial['wrist_visibility']}",
                    )
                finally:
                    context.close()

    def test_size_tasks_share_layout_but_select_different_targets(self) -> None:
        largest_context, largest, largest_initial = self.make_scene("largest_block")
        smallest_context, smallest, smallest_initial = self.make_scene("smallest_block")
        try:
            self.assertEqual(
                largest_initial["active_bodies"],
                ["large_block", "small_block"],
            )
            self.assertEqual(
                smallest_initial["active_bodies"],
                ["large_block", "small_block"],
            )
            self.assertEqual(largest.target_body, "large_block")
            self.assertEqual(smallest.target_body, "small_block")
        finally:
            largest_context.close()
            smallest_context.close()

    def test_randomization_is_reproducible(self) -> None:
        context_a, _, initial_a = self.make_scene("largest_block", seed=123)
        context_b, _, initial_b = self.make_scene("largest_block", seed=123)
        try:
            self.assertEqual(initial_a["scene_xy_delta"], initial_b["scene_xy_delta"])
            self.assertEqual(
                initial_a["initial_body_positions"],
                initial_b["initial_body_positions"],
            )
        finally:
            context_a.close()
            context_b.close()

    def test_place_task_swaps_local_held_pepper_on_release(self) -> None:
        context, runtime, _ = self.make_scene("place_red_pepper_in_ring")
        try:
            self.assertFalse(runtime.released)
            self.assertEqual(runtime.target_body, "red_pepper")
            self.assertAlmostEqual(
                runtime.physical_gripper_raw_target(440.0),
                492.58,
            )
            self.assertEqual(runtime.active_target_body, "held_red_pepper")
            observation = {
                "observation/state": np.asarray(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 845.0],
                    dtype=np.float32,
                )
            }
            runtime.adjust_observation(observation)
            self.assertAlmostEqual(
                float(observation["observation/state"][6]), 492.58, places=3
            )
            held_id = mujoco.mj_name2id(
                context.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "held_red_pepper",
            )
            pepper_id = mujoco.mj_name2id(
                context.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "red_pepper",
            )
            pepper_joint = int(context.model.body_jntadr[pepper_id])
            self.assertEqual(
                int(context.model.jnt_type[pepper_joint]),
                int(mujoco.mjtJoint.mjJNT_FREE),
            )
            self.assertLess(
                mujoco.mj_name2id(
                    context.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "red_pepper_grasp_collision",
                ),
                0,
            )
            held_lobe = mujoco.mj_name2id(
                context.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "held_pepper_lobe_0",
            )
            self.assertEqual(int(context.model.geom_bodyid[held_lobe]), held_id)
            self.assertEqual(int(context.model.geom_contype[held_lobe]), 1)
            for hidden_name in ("red_pepper_lobe_0", "red_pepper_stem"):
                hidden_geom = mujoco.mj_name2id(
                    context.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    hidden_name,
                )
                self.assertEqual(int(context.model.geom_contype[hidden_geom]), 0)
                self.assertEqual(int(context.model.geom_conaffinity[hidden_geom]), 0)
            equality_names = {
                mujoco.mj_id2name(context.model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
                for index in range(context.model.neq)
            }
            self.assertNotIn(
                "red_pepper",
                " ".join(name or "" for name in equality_names),
            )
            position_before_release = context.data.xpos[held_id].copy()
            self.assertFalse(runtime.release_if_requested(600.0))
            self.assertTrue(runtime.release_if_requested(700.0))
            self.assertTrue(runtime.released)
            np.testing.assert_allclose(
                context.data.xpos[pepper_id], position_before_release, atol=1e-12
            )
            self.assertEqual(runtime.active_target_body, "red_pepper")
            self.assertIsNotNone(runtime.release_simulation_time_s)
            self.assertAlmostEqual(
                runtime.physical_gripper_raw_target(700.0),
                700.0,
            )
            self.assertGreater(float(context.data.xpos[pepper_id, 2]), 0.05)
        finally:
            context.close()

    def test_pick_pepper_uses_local_compound_geometry(self) -> None:
        context, runtime, _ = self.make_scene("red_pepper")
        try:
            for geom_name in (
                "red_pepper_lobe_0",
                "red_pepper_stem",
            ):
                geom_id = mujoco.mj_name2id(
                    context.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_name,
                )
                self.assertEqual(int(context.model.geom_contype[geom_id]), 1)
                self.assertEqual(int(context.model.geom_conaffinity[geom_id]), 1)
            self.assertLess(
                mujoco.mj_name2id(
                    context.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "red_pepper_grasp_collision",
                ),
                0,
            )
        finally:
            context.close()

    def test_lift_success_requires_sustained_steps(self) -> None:
        context, runtime, _ = self.make_scene("red_block")
        try:
            body_id = mujoco.mj_name2id(
                context.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "object",
            )
            joint_id = int(context.model.body_jntadr[body_id])
            qpos_addr = int(context.model.jnt_qposadr[joint_id])
            context.data.qpos[qpos_addr + 2] += 0.08
            mujoco.mj_forward(context.model, context.data)
            self.assertFalse(runtime.update_success()["task_success"])
            self.assertFalse(runtime.update_success()["task_success"])
            self.assertTrue(runtime.update_success()["task_success"])
        finally:
            context.close()


if __name__ == "__main__":
    unittest.main()
