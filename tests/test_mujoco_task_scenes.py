from __future__ import annotations

import unittest

import mujoco
import numpy as np

from sim_mujoco.remote_policy_observation import initialize_scene, load_simulation
from sim_mujoco.task_scenes import (
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
                        item["visible"]
                        for item in initial["wrist_visibility"].values()
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

    def test_place_task_transfers_pepper_on_open_command(self) -> None:
        context, runtime, _ = self.make_scene("place_red_pepper_in_ring")
        try:
            self.assertFalse(runtime.released)
            self.assertAlmostEqual(
                runtime.physical_gripper_target(440.0, 0.0196),
                0.0273,
            )
            observation = {
                "observation/state": np.asarray(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 845.0],
                    dtype=np.float32,
                )
            }
            runtime.adjust_observation(observation)
            self.assertAlmostEqual(
                float(observation["observation/state"][6]),
                float(runtime.spec["initial_gripper_raw"]),
                places=3,
            )
            self.assertFalse(runtime.release_if_requested(600.0))
            self.assertTrue(runtime.release_if_requested(700.0))
            self.assertTrue(runtime.released)
            self.assertAlmostEqual(
                runtime.physical_gripper_target(700.0, 0.0327),
                0.038,
            )
            pepper_id = mujoco.mj_name2id(
                context.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "red_pepper",
            )
            self.assertGreater(float(context.data.xpos[pepper_id, 2]), 0.05)
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
