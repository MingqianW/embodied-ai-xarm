from __future__ import annotations

import unittest

import numpy as np

from sim_mujoco.data_collection.ik_solver import solve_site_pose
from sim_mujoco.data_collection.oracle_controller import (
    OracleConfig,
    OracleStage,
    RED_PEPPER_CLOSED_GRIPPER_RAW,
    RED_PEPPER_GRASP_TCP_OFFSET_M,
    ScriptedOracleController,
    oracle_config_for_task,
)
from sim_mujoco.data_collection.task_success import (
    simulation_is_finite,
    update_task_success,
)
from sim_mujoco.environment import MuJoCoEnvironment


class ScriptedOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = MuJoCoEnvironment(task="red_block", settle_steps=100)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.environment.close()

    def test_ik_reaches_pregrasp_pose_without_mutating_environment(self) -> None:
        self.environment.reset(seed=0)
        controller = ScriptedOracleController(self.environment)
        before = self.environment.context.data.qpos.copy()
        solution = solve_site_pose(
            self.environment.context.model,
            self.environment.context.data,
            site_name=controller.config.tcp_site,
            target_position=np.asarray(
                [
                    controller.plan.object_position[0],
                    controller.plan.object_position[1],
                    controller.plan.object_position[2]
                    + controller.config.pregrasp_clearance_from_object_m,
                ]
            ),
            target_rotation=controller.plan.tcp_rotation,
            seed_joint_qpos=controller.plan.initial_arm_qpos,
        )
        self.assertTrue(solution.success)
        self.assertLess(solution.position_error_m, 1e-5)
        np.testing.assert_array_equal(
            self.environment.context.data.qpos,
            before,
        )

    def test_planned_actions_obey_per_step_limits(self) -> None:
        self.environment.reset(seed=0)
        config = OracleConfig()
        controller = ScriptedOracleController(self.environment, config)
        previous = controller.plan.initial_arm_qpos
        while not controller.terminal:
            stage = controller.stage
            action = controller.next_action()
            if action is None:
                break
            limit = (
                config.lift_max_joint_step_rad
                if stage == OracleStage.LIFT
                else config.max_joint_step_rad
            )
            self.assertLessEqual(
                float(np.max(np.abs(action[:6] - previous))),
                limit + 1e-6,
            )
            previous = action[:6].copy()
            controller.action_steps += 1

    def test_old_success_streak_cannot_accept_a_verification_drop(self) -> None:
        self.environment.reset(seed=3)
        controller = ScriptedOracleController(self.environment)
        metrics = self.environment.task_runtime.metrics()
        maximum_success_streak = 0
        while not controller.terminal:
            action = controller.next_action()
            if action is None:
                break
            self.environment.apply_action(action)
            self.environment.step_physics(controller.config.action_dt_s)
            metrics = update_task_success(self.environment)
            maximum_success_streak = max(
                maximum_success_streak, int(metrics["success_streak"])
            )
            collision = self.environment.safety_diagnostics()["collision"]
            controller.notify_post_step(
                task_metrics=metrics,
                collision=collision,
                simulation_finite=simulation_is_finite(self.environment),
            )
        self.assertEqual(
            controller.stage,
            OracleStage.FAILED,
        )
        self.assertGreaterEqual(maximum_success_streak, 3)
        stability = controller.stability_metadata()
        self.assertFalse(stability["stable_grasp_success"])
        self.assertEqual(stability["verification_steps_executed"], 20)
        self.assertEqual(controller.failure_reason, "stable_grasp_table_contact")

    def test_same_seed_produces_identical_action_plan(self) -> None:
        plans = []
        for _ in range(2):
            self.environment.reset(seed=17)
            controller = ScriptedOracleController(self.environment)
            actions = []
            while not controller.terminal:
                action = controller.next_action()
                if action is None:
                    break
                actions.append(action)
                controller.action_steps += 1
            plans.append(np.asarray(actions))
        np.testing.assert_array_equal(plans[0], plans[1])

    def test_oracle_parameter_overrides_are_explicit(self) -> None:
        pepper_default = oracle_config_for_task(
            "red_pepper",
            action_dt_s=0.1,
        )
        pepper = oracle_config_for_task(
            "red_pepper",
            action_dt_s=0.1,
            closed_gripper_raw=450.0,
            grasp_tcp_offset_from_object_m=0.005,
        )
        block = oracle_config_for_task("red_block", action_dt_s=0.1)

        self.assertEqual(
            pepper_default.closed_gripper_raw,
            RED_PEPPER_CLOSED_GRIPPER_RAW,
        )
        self.assertEqual(
            pepper_default.grasp_tcp_offset_from_object_m,
            RED_PEPPER_GRASP_TCP_OFFSET_M,
        )
        self.assertEqual(
            pepper.closed_gripper_raw,
            450.0,
        )
        self.assertEqual(pepper.grasp_tcp_offset_from_object_m, 0.005)
        self.assertEqual(
            block.closed_gripper_raw,
            OracleConfig.closed_gripper_raw,
        )
        self.assertEqual(
            block.grasp_tcp_offset_from_object_m,
            OracleConfig.grasp_tcp_offset_from_object_m,
        )


if __name__ == "__main__":
    unittest.main()
