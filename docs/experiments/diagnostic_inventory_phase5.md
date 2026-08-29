# Phase 5 diagnostic inventory

The Phase 5 usage graph found one production dependency: formal simulation
evaluation imported the physics trace recorder. That recorder is now a stable
measurement primitive; evaluation imports no experiment runner or analyzer.
The classifications below cover the 41 Python diagnostic candidates present at
the start of the phase.

| Original candidate | Classification | Result |
| --- | --- | --- |
| `sim_mujoco/gripper_slip_diagnostics.py` | REFACTOR | `diagnostics/simulation/gripper/trace.py` |
| `scripts/analyze_xarm_slip_trace.py` | REFACTOR | `diagnostics/simulation/gripper/analyze_trace.py` |
| `scripts/audit_generation_evaluation_physics.py` | REFACTOR | `diagnostics/simulation/physics/consistency.py` |
| `scripts/audit_real_sim_gripper_behavior.py` | REFACTOR | `diagnostics/real_sim/gripper/behavior.py` |
| `scripts/camera_calibration_lib.py` | REFACTOR | `diagnostics/simulation/camera/calibration.py` |
| `scripts/discover_raw_camera_data.py` | MERGE | `diagnostics/simulation/camera/cli.py discover` |
| `scripts/select_camera_calibration_frames.py` | MERGE | `diagnostics/simulation/camera/cli.py select` |
| `scripts/calibrate_cameras.py` | MERGE | `diagnostics/simulation/camera/cli.py fit` |
| `scripts/evaluate_camera_calibration.py` | MERGE | `diagnostics/simulation/camera/cli.py evaluate` |
| `scripts/render_calibration_frame.py` | MERGE | `diagnostics/simulation/camera/cli.py render` |
| `scripts/analyze_contact_model_realism_regression.py` | DELETE | Retired experiment analyzer |
| `scripts/analyze_friction_ablation.py` | DELETE | Retired experiment analyzer |
| `scripts/analyze_friction_search.py` | DELETE | Retired parameter search |
| `scripts/analyze_grip_force_vs_width.py` | DELETE | Retired experiment analyzer |
| `scripts/analyze_menagerie_forcerange_tuning.py` | DELETE | Retired parameter tuner |
| `scripts/analyze_policy_gripper_slip_matrix.py` | DELETE | Retired experiment matrix |
| `scripts/analyze_real_sim_gripper_trajectories.py` | DELETE | Overlapped retained behavior audit |
| `scripts/analyze_scripted_gripper_slip_experiments.py` | DELETE | Retired experiment analyzer |
| `scripts/analyze_split_pad_geometry_experiment.py` | DELETE | Retired experiment analyzer |
| `scripts/audit_kinematic_mapping.py` | DELETE | One-off audit with stale data paths |
| `scripts/compare_real_sim_datasets.py` | DELETE | Broad one-off comparison with stale data paths |
| `scripts/diagnose_place_initial_grasp.py` | DELETE | One-off debug program |
| `scripts/diagnose_xarm_slip.py` | DELETE | Replaced by formal evaluator trace capture |
| `scripts/prepare_friction_policy_variant.py` | DELETE | Retired experiment preparation |
| `scripts/refine_base_camera.py` | DELETE | Completed one-off calibration refinement |
| `scripts/render_current_camera_comparisons.py` | DELETE | Duplicated retained calibration library |
| `scripts/run_contact_model_realism_regression.py` | DELETE | Retired experiment runner |
| `scripts/run_friction_ablation.py` | DELETE | Retired experiment runner |
| `scripts/run_friction_search.py` | DELETE | Retired parameter search |
| `scripts/run_grip_force_vs_width.py` | DELETE | Retired experiment runner |
| `scripts/run_menagerie_gripper_validation.py` | DELETE | Duplicated maintained simulation tests |
| `scripts/run_scripted_gripper_slip_experiments.py` | DELETE | Retired experiment runner |
| `scripts/run_split_pad_geometry_experiment.py` | DELETE | Retired experiment runner |
| `scripts/summarize_baseline_final.py` | DELETE | Superseded evaluation reporting |
| `scripts/test_dual_cameras.py` | DELETE | Ad hoc local check |
| `scripts/test_gripper_control.py` | DELETE | Ad hoc local check |
| `scripts/test_local_observation.py` | DELETE | Ad hoc local check |
| `scripts/test_local_render.py` | DELETE | Ad hoc local check |
| `scripts/test_xarm6_control.py` | DELETE | Ad hoc local check |
| `scripts/tune_base_camera_interactive.py` | DELETE | Interactive one-off tuner |
| `scripts/tune_base_roll_fovy.py` | DELETE | Completed one-off calibration refinement |

No candidate met the high bar for `LEGACY_RETAIN`. The related split-pad
config, obsolete tests, and all 13 experiment-only `slurm/xarm_eval` launchers
were deleted with their implementations. Git history is the archive.
