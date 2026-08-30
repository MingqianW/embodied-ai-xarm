# Repository Reorganization Plan (Phase 1 Audit)

Status: planning only. Baseline: `18cfc93`. Audit branch:
`refactor/reorganize-repository`.

This document inventories the repository at the integration baseline and proposes
a future move-only-first reorganization. Phase 1 does **not** authorize source
moves, import rewrites, API changes, configuration consolidation, behavior
changes, deletions, commits, or pushes.

## 1. Safety and phase boundary

The audit started from a clean worktree at full commit
`18cfc9382f366d81278c49e0a60ab1d18fa0afa8`. No merge was in progress. The
known-good commit was initially checked out on
`integrate/local-sim-with-delta-20260827`; because the requested audit branch did
not exist, it was created directly at that commit before any file changes.

The following validated behavior is immutable throughout this plan: the local
MuJoCo scene and physics, xArm/gripper geometry and actuator convention, camera
calibration, task geometry and Place initialization semantics, state/action
contracts, six-task meanings, formal success criteria, and generation/evaluation
physics consistency.

## 2. Inventory and current architecture

There are 950 tracked paths. The high-level distribution at the audit baseline
is:

| Current area | Tracked paths | Actual role |
| --- | ---: | --- |
| `third_party/` | 667 | 666 vendored xArm ROS 2 paths and one OpenPI gitlink |
| `sim_mujoco/` | 140 | Simulation core, two data-generation generations, formal evaluation, diagnostics, tools, assets, and configs |
| `docs/` | 38 | Current guides mixed with dated results and migration handoffs |
| `tests/` | 35 | Flat cross-subsystem test suite |
| `slurm/` | 28 | Simulation-data orchestration and simulation diagnostics |
| `fine_tune/` | 18 | Real-data conversion/QA mixed with training utilities and reference artifacts |
| `policy_runtime/` | 13 | Simulator-independent policy client, contracts, safety, runners, logging, and recording |
| `scripts/` | 2 | Environment/manifest repository utilities |
| `environment/` | 2 | Environment documentation/requirements |
| repository root | 7 | Project metadata, environment files, README, and ad hoc real rollout entrypoint |

`sim_isaac/` has no tracked paths and is not part of this audit's production
architecture. Local virtual environments, caches, datasets, and outputs seen in
the worktree are ignored and are not repository content.

The present architecture is implementation-rich but workflow-poor:

- `sim_mujoco/` owns the canonical simulator, but also contains data, evaluation,
  diagnostic, comparison, export, and developer scripts.
- `sim_mujoco/data_collection/` is not real robot collection. It is scripted
  **simulation** collection support.
- `fine_tune/` combines real-data ingestion and dataset maintenance with actual
  training support.
- `run_pi_xarm.py` is the only tracked real-hardware rollout program; it is an
  ad hoc local-policy runner/recorder, not a formal real evaluator.
- `policy_runtime/` is already the best shared policy-runtime boundary.
- `slurm/xarm_eval/` is mostly simulation diagnostics despite its name.

## 3. Four core pipelines

### 3.1 Simulation data generation

| Concern | Current implementation |
| --- | --- |
| Canonical entrypoint | `python -m sim_mujoco.data_generation.cli` |
| Older entrypoints | `sim_mujoco/scripts/collect_oracle_data.py`, `collect_real_raw_sim_data.py`, `convert_mujoco_to_lerobot.py` |
| Core modules | `sim_mujoco/data_generation/`; scripted simulator support in `sim_mujoco/data_collection/` |
| Simulator | `sim_mujoco/environment.py` and the shared simulation modules/assets |
| Config | `sim_mujoco/config/data_generation/*.yaml`, `task_scenes.yaml`, camera YAML |
| State/action and schemas | 7D state/action; real-raw-compatible recorder, native simulator recorder, conversion adapters, shared `fine_tune/xarm_lerobot_writer.py` |
| Observations | Environment camera/state capture plus recorder adapters |
| Outputs | Roots resolved by `sim_mujoco/paths.py`; raw episodes, manifests/audits, then canonical LeRobot data |
| Tests | `test_mujoco_data_*`, episode/LeRobot/HF/v4/oracle tests |
| Cluster | `slurm/simulation_data/*` and export job |
| Gap/problem | `data_generation` and `data_collection` import each other, and the older and current orchestration paths coexist |

The `data_generation` package is the canonical current orchestrator. The
`data_collection` package remains useful implementation and should be moved,
not rewritten. Phase 2 should first break its cycle by placing shared generation
contracts/validators at a lower layer inside `data/sim/generation/`.

### 3.2 Real robot data collection

| Concern | Current implementation |
| --- | --- |
| Primary collector | Not tracked. `docs/commands/real_world_data_collection.md` invokes an external `real_world/collect_async_gripper_optimized.py` |
| Tracked raw-data support | `fine_tune/xarm_data_config.py`, `check_xarm_raw_quality.py`, `calculate_xarm_demo_time.py`, `rename_xarm_raw_task.py` |
| Conversion | `fine_tune/convert_xarm_raw_to_lerobot.py` |
| Canonical writer | `fine_tune/xarm_lerobot_writer.py` |
| Raw contract | Episode directory, `robot_log.csv`, `realsense_0/`, `realsense_1/`, `ts`, six joint radians plus `gripper_mm` |
| Output | Canonical LeRobot dataset with base/wrist images, 7D state/actions, and task |
| Tests | Conversion/pipeline tests exercise shared semantics; no tracked hardware collection test |
| Cluster | None |
| Gap/problem | The hardware collector and hardware adapter are external; collection is not reproducible from this repo alone |

### 3.3 Simulation evaluation

| Concern | Current implementation |
| --- | --- |
| Canonical formal entrypoint | `sim_mujoco/scripts/evaluate_xarm_policy.py` |
| Canonical core | `sim_mujoco/formal_evaluation/` |
| Older runtime/evaluators | `run_remote_policy_closed_loop.py`, `evaluate_remote_policy_automatic.py`, `evaluate_remote_policy_interactive.py` |
| Shared policy layer | `policy_runtime/` |
| Simulator adapters | `remote_policy_observation.py`, `remote_policy_control.py`, `recording.py` |
| Config | `formal_models/*.json`, `formal_xarm_pi05_eval*.json`, tasks/camera configs |
| Outputs | Formal provenance/results/summaries/videos/human-review and slip traces |
| Tests | Formal, remote-pipeline, runtime, human-review, video, and slip-trace tests |
| Cluster | No thin canonical six-task formal-evaluation job; `slurm/xarm_eval/` mostly drives diagnostics |
| Gap/problem | Legacy evaluation wrappers overlap with the formal path and shared evaluation helpers |

### 3.4 Real robot evaluation

| Concern | Current implementation |
| --- | --- |
| Tracked entrypoint | `run_pi_xarm.py` |
| What it does | Builds real observations, runs a local OpenPI policy, executes receding-horizon absolute actions, and optionally records the real raw schema |
| What it does not do | Formal task randomization, formal success measurement, shared result schema, failure taxonomy, provenance package, or aggregate metrics |
| Dependencies | External `real_world.xarm6.XARM6`, MultiRealsense, and OpenPI paths/checkpoint currently hard-coded under `/home/xingyu/...` |
| Evidence/results | `docs/real_robot_model_eval_log_260626.md` is a historical manual result table, not evaluator code |
| Tests/cluster | None |
| Missing functionality | A repository-owned hardware environment adapter and a formal real evaluation runner are not present |

No formal real evaluation implementation should be inferred from the simulator
or invented during the move phase. Phase 2 may move the existing runner intact;
design/implementation of missing formal evaluation is a later feature phase.

## 4. Canonical shared simulation ownership

| Concern | Canonical current source | Consumers |
| --- | --- | --- |
| Environment/reset/render | `sim_mujoco/environment.py` | generation, formal evaluation, collection, diagnostics |
| Arm/scene MJCF | `assets/xarm6/xarm6_arm.xml`, `xarm6_pick_scene.xml` | all simulator consumers |
| Robot source geometry | `third_party/xarm_ros2/xarm_description/...` | `generate_xarm6_mjcf.py` only |
| Gripper source/provenance | pick-scene builder and `MENAGERIE_XARM_GRIPPER_SOURCE.md` | canonical scene build/audit |
| Gripper mechanics/mapping | scene XML and `gripper_mapping.py` | environment, observation/control, generation, diagnostics |
| Joint mapping | `joint_mapping.py` | simulator/runtime/data conversion |
| Collision/contact handling | scene XML and `collision.py` | environment, evaluation, diagnostics |
| Camera configuration | `config/camera_calibration.yaml` with static MJCF cameras | environment/observation/rendering/calibration tools |
| Task scenes | `task_scenes.py` + `config/task_scenes.yaml` | generation and evaluation |
| Policy observation adapter | `remote_policy_observation.py` | simulation evaluation/tools |
| Policy control adapter | `remote_policy_control.py` | simulation evaluation/tools |
| Recording capture adapter | `recording.py` | shared runtime recorder |
| Path resolution | `paths.py` | simulation workflows and tools |

`gripper_slip_diagnostics.py` is explicitly diagnostic and is not part of the
canonical policy-facing simulator. None of the canonical simulation modules may
gain dependencies on data generation, evaluation, diagnostics, or training.

The two scene build scripts are distinct and should remain reproducible build
tools:

1. `generate_xarm6_mjcf.py` converts vendored xArm kinematics, inertial data,
   and meshes to `xarm6_arm.xml`.
2. `build_xarm6_pick_scene.py` combines that arm model, Menagerie gripper,
   calibrated cameras, objects, task geometry, and validated physics into
   `xarm6_pick_scene.xml`.

The checked-in XML files are generated source artifacts that are also runtime
inputs. They are `KEEP`, not cleanup candidates.

## 5. Data contracts

The policy/training-facing contract is already aligned across domains:

- `state`: seven absolute values `[j1_rad, ..., j6_rad, gripper_mm]`.
- `actions`: seven absolute next targets with the same order.
- base image: `observation/image` at runtime and `image` in the dataset.
- wrist image: `observation/wrist_image` at runtime and `wrist_image` in the
  dataset.
- `task`: normalized natural-language task prompt.
- canonical dataset: LeRobot episodes written by
  `fine_tune/xarm_lerobot_writer.py`.

Real raw episodes use `robot_log.csv`, image paths under `realsense_0/` and
`realsense_1/`, and `ts`. Conversion treats the next row's state as the current
row's absolute action and drops the last row. The simulator's
`real_raw_recorder.py` intentionally reproduces this contract.

`episode_recorder.py` additionally supports a richer simulator-native raw
contract (native and policy images, frame/episode indices, timestamp, state,
actions, task, and simulator metadata). This is a deliberate domain-specific
format, not a duplicate to delete.

Potential future `data/common/` ownership is proven for the 7D constants,
canonical LeRobot writer, task normalization, and small schema validators.
Extraction must preserve names, value order, units, next-frame semantics, and
serialized schemas exactly. Simulator metadata and raw capture remain
domain-specific.

## 6. Evaluation ownership

Genuinely shared today:

- remote policy transport and timeout handling;
- action chunk decoding and safety validation;
- observation-schema construction primitives;
- environment protocol and generic runners;
- episode JSON/CSV logging, manual labels, summaries, and video tiling.

These belong in `policy_runtime/` or, only where semantics are demonstrably
evaluation-specific, a future `evaluation/common/`.

Simulation-specific today:

- MuJoCo model loading, cameras, state extraction, reset/randomization;
- conversion from 7D policy targets to MuJoCo controls;
- collision, success, Place held/free state, slip, and physics checks;
- scene videos and simulator provenance.

Real-specific today:

- xArm/MultiRealsense setup and hardware action execution in `run_pi_xarm.py`.

Model identity, task identity, and result records are plausible future shared
concepts because equivalents exist in policy-runtime and formal-sim outputs.
Formal success logic is not shared: only simulation has a formal implementation.

## 7. File classification summary

Classification is conservative. `MOVE` means preserve implementation with
`git mv`; `MERGE` means consolidate only after tests prove equivalent behavior.

| Class | Paths / families | Rationale |
| --- | --- | --- |
| KEEP | canonical simulator XML/build inputs; `third_party/`; `policy_runtime/`; environment metadata; active configs; shared writer | Correct or shared implementation and validated input |
| MOVE | simulator core to `simulation/`; sim data packages to `data/sim/generation/`; formal evaluation to `evaluation/sim/`; diagnostics/scripts/configs/slurm/tests/docs to named subsystem directories; real data utilities to `data/real/`; training files to `training/` | Useful code in overloaded/misnamed locations |
| MERGE | old remote evaluators with formal/shared evaluation; remaining remote evaluation/recording helpers; generation/collection package boundary; duplicated 7D constants/task normalization; `slurm` common v3/v4 orchestration | Demonstrated overlap; no Phase 1 consolidation |
| DELETE_CANDIDATE | obsolete status/handoff snapshots after archival review; `fine_tune/data/.gitignore` if the new root ignore fully replaces it; broken/unsafe one-off dataset mutation tools only after replacement and provenance review | Nothing qualifies for immediate deletion |
| GENERATED | checked-in migration audit JSON/Markdown, validation reports, required-files manifest, training result reports; runtime outputs ignored by `.gitignore`; generated runtime-input XML noted separately | Reports are reproducible snapshots; do not silently delete |
| LEGACY | frozen split-pad/friction experiments and legacy camera config; prior remote evaluation path; dated tracker/evaluation/migration records | Required for reproducibility/history until explicit cleanup |
| REVIEW | `run_pi_xarm.py`; external-collector docs; `fine_tune/inspect_lerobot_parquet_sample.py` (apparent syntax error near line 584); destructive dataset tools; stale root README; historical reports with unclear retention policy | Ownership, correctness, or retention needs an owner decision |

No tracked `*.before_*`, cache, video, checkpoint, dataset, log, temporary file,
or other unquestionably accidental runtime artifact was found. Therefore Phase 1
deletes nothing.

## 8. Exact proposed move map

All moves below are future Phase 2 commands and should use `git mv` before any
content edit. Directory moves imply every tracked file currently beneath the
listed directory.

### 8.1 Simulation core and configs

| Current | Future | Action |
| --- | --- | --- |
| `sim_mujoco/__init__.py` | `simulation/__init__.py` | `git mv` |
| `sim_mujoco/environment.py` | `simulation/environment.py` | `git mv` |
| `sim_mujoco/collision.py` | `simulation/collision.py` | `git mv` |
| `sim_mujoco/gripper_mapping.py` | `simulation/gripper_mapping.py` | `git mv` |
| `sim_mujoco/joint_mapping.py` | `simulation/joint_mapping.py` | `git mv` |
| `sim_mujoco/task_scenes.py` | `simulation/task_scenes.py` | `git mv` |
| `sim_mujoco/paths.py` | `simulation/paths.py` | `git mv` |
| `sim_mujoco/recording.py` | `simulation/recording.py` | `git mv` |
| `sim_mujoco/remote_policy_observation.py` | `simulation/policy_observation.py` | `git mv` |
| `sim_mujoco/remote_policy_control.py` | `simulation/policy_control.py` | `git mv` |
| `sim_mujoco/assets/` | `simulation/assets/` | `git mv` as a directory |
| `sim_mujoco/constraints.txt` | `environment/mujoco-constraints.txt` | `git mv` |
| `sim_mujoco/README.md` | `docs/simulation/README.md` | `git mv` |
| `sim_mujoco/DATA_COLLECTION.md` | `docs/data/sim-collection-legacy.md` | `git mv`; retain context |
| `sim_mujoco/config/camera_calibration.yaml` | `configs/simulation/camera_calibration.yaml` | `git mv` |
| `sim_mujoco/config/task_scenes.yaml` | `configs/tasks/task_scenes.yaml` | `git mv` |
| `sim_mujoco/scripts/generate_xarm6_mjcf.py` | `tools/simulation/generate_xarm6_mjcf.py` | `git mv` |
| `sim_mujoco/scripts/build_xarm6_pick_scene.py` | `tools/simulation/build_xarm6_pick_scene.py` | `git mv` |
| `sim_mujoco/scripts/render_task_scenes.py` | `tools/simulation/render_task_scenes.py` | `git mv` |
| `sim_mujoco/scripts/smoke_test_headless_render.py` | `tools/simulation/smoke_test_headless_render.py` | `git mv` |
| `sim_mujoco/scripts/teleoperate_pick.py` | `tools/simulation/teleoperate_pick.py` | `git mv` |
| `sim_mujoco/scripts/test_gripper_control.py` | `tools/simulation/test_gripper_control.py` | `git mv` |
| `sim_mujoco/scripts/test_local_observation.py` | `tools/simulation/test_local_observation.py` | `git mv` |
| `sim_mujoco/scripts/test_local_render.py` | `tools/simulation/test_local_render.py` | `git mv` |
| `sim_mujoco/scripts/test_xarm6_control.py` | `tools/simulation/test_xarm6_control.py` | `git mv` |

Keep `third_party/openpi` and `third_party/xarm_ros2` in place. They are explicit
vendor boundaries, not application subsystems.

### 8.2 Simulation data generation and data contracts

| Current | Future | Action |
| --- | --- | --- |
| `sim_mujoco/data_generation/` | `data/sim/generation/` | `git mv` directory, then break the import cycle minimally |
| `sim_mujoco/data_collection/` | `data/sim/generation/collection_support/` | `git mv` directory intact first; flatten/consolidate only after parity review |
| `sim_mujoco/config/data_generation/` | `configs/data/sim/generation/` | `git mv` directory |
| `sim_mujoco/scripts/collect_oracle_data.py` | `data/sim/generation/legacy/collect_oracle_data.py` | `git mv`; retain legacy path |
| `sim_mujoco/scripts/collect_real_raw_sim_data.py` | `data/sim/generation/legacy/collect_real_raw_sim_data.py` | `git mv`; retain legacy path |
| `sim_mujoco/scripts/convert_mujoco_to_lerobot.py` | `data/sim/generation/legacy/convert_mujoco_to_lerobot.py` | `git mv`; retain legacy path |
| `sim_mujoco/scripts/diagnose_place_initial_grasp.py` | `diagnostics/simulation/data_generation/diagnose_place_initial_grasp.py` | `git mv` |
| `sim_mujoco/scripts/test_scripted_oracle.py` | `tools/data_sim/test_scripted_oracle.py` | `git mv` |
| `sim_mujoco/scripts/prepare_mujoco_hf_ready.py` | `tools/datasets/prepare_mujoco_hf_ready.py` | `git mv` |
| `sim_mujoco/scripts/upload_mujoco_dataset_to_hf.py` | `tools/datasets/upload_mujoco_dataset_to_hf.py` | `git mv` |
| `sim_mujoco/scripts/validate_mujoco_lerobot_dataset.py` | `tools/datasets/validate_mujoco_lerobot_dataset.py` | `git mv` |
| `sim_mujoco/scripts/validate_real_raw_sim_data.py` | `tools/data_sim/validate_real_raw_sim_data.py` | `git mv` |
| `sim_mujoco/scripts/export_lerobot_training_videos.py` | `tools/datasets/export_lerobot_training_videos.py` | `git mv` |
| `fine_tune/xarm_lerobot_writer.py` | `data/common/xarm_lerobot_writer.py` | `git mv`; preserve serialized schema |

### 8.3 Real data and training

| Current | Future | Action |
| --- | --- | --- |
| `fine_tune/convert_xarm_raw_to_lerobot.py` | `data/real/conversion/convert_xarm_raw_to_lerobot.py` | `git mv` |
| `fine_tune/check_xarm_raw_quality.py` | `data/real/validation/check_xarm_raw_quality.py` | `git mv` |
| `fine_tune/calculate_xarm_demo_time.py` | `data/real/tools/calculate_xarm_demo_time.py` | `git mv` |
| `fine_tune/rename_xarm_raw_task.py` | `data/real/tools/rename_xarm_raw_task.py` | `git mv`; flag as mutating tool |
| `fine_tune/xarm_data_config.py` | `data/real/config.py` | `git mv` |
| `fine_tune/xarm_data_config.example.json` | `configs/data/real/xarm_data_config.example.json` | `git mv` |
| `sim_mujoco/scripts/build_real_dataset_schema.py` | `data/real/validation/build_dataset_schema.py` | `git mv` |
| `fine_tune/check_lerobot_action_jumps.py` | `tools/datasets/check_lerobot_action_jumps.py` | `git mv` |
| `fine_tune/check_lerobot_openpi_outliers.py` | `tools/datasets/check_lerobot_openpi_outliers.py` | `git mv` |
| `fine_tune/inspect_lerobot_parquet_sample.py` | `tools/datasets/inspect_lerobot_parquet_sample.py` | `git mv` only after syntax/owner review |
| `fine_tune/trim_lerobot_edges.py` | `tools/datasets/trim_lerobot_edges.py` | `git mv`; flag as mutating tool |
| `fine_tune/delete_lerobot_task_parquets.py` | `tools/datasets/delete_lerobot_task_parquets.py` | `git mv`; flag as destructive tool |
| `fine_tune/openpi_xarm_config.py` | `training/openpi_xarm_config.py` | `git mv` |
| `fine_tune/openpi_train_debug.py` | `training/tools/openpi_train_debug.py` | `git mv` |
| `fine_tune/smoke_test_openpi_xarm_dataset.py` | `training/tools/smoke_test_openpi_xarm_dataset.py` | `git mv` |
| `fine_tune/openpi_pi05_colab_pipeline.ipynb` | `training/notebooks/openpi_pi05_colab_pipeline.ipynb` | `git mv` |
| `fine_tune/openpi_xarm_project_intro.pptx` | `docs/experiments/openpi_xarm_project_intro.pptx` | `git mv` |
| `fine_tune/data/.gitignore` | review after data-root migration | retain until replacement is proven |
| `run_pi_xarm.py` | `evaluation/real/run_policy.py` | `git mv` unchanged; formalization is later work |

### 8.4 Simulation evaluation and shared policy runtime

| Current | Future | Action |
| --- | --- | --- |
| `sim_mujoco/formal_evaluation/` | `evaluation/sim/` | `git mv` directory |
| `sim_mujoco/config/formal_models/` | `configs/evaluation/sim/models/` | `git mv` directory |
| `sim_mujoco/config/formal_xarm_pi05_eval*.json` | `configs/evaluation/sim/protocols/` | `git mv` all six exact matched files |
| `sim_mujoco/scripts/evaluate_xarm_policy.py` | `evaluation/sim/cli.py` | `git mv` |
| `sim_mujoco/scripts/build_xarm_human_review_manifest.py` | `evaluation/sim/tools/build_human_review_manifest.py` | `git mv` |
| `sim_mujoco/scripts/reclassify_xarm_evaluation_failures.py` | `evaluation/sim/tools/reclassify_failures.py` | `git mv` |
| `sim_mujoco/scripts/review_xarm_human_videos.py` | `evaluation/sim/tools/review_human_videos.py` | `git mv` |
| `sim_mujoco/scripts/summarize_xarm_evaluation.py` | `evaluation/sim/tools/summarize_evaluation.py` | `git mv` |
| `sim_mujoco/scripts/summarize_xarm_human_review.py` | `evaluation/sim/tools/summarize_human_review.py` | `git mv` |
| `sim_mujoco/scripts/validate_xarm_abc_evaluation.py` | `evaluation/sim/tools/validate_abc_evaluation.py` | `git mv` |
| `sim_mujoco/scripts/validate_xarm_category_video_coverage.py` | `evaluation/sim/tools/validate_category_video_coverage.py` | `git mv` |
| `sim_mujoco/scripts/run_remote_policy_closed_loop.py` | `evaluation/sim/legacy/run_remote_policy_closed_loop.py` | `git mv`; legacy until parity review |
| `sim_mujoco/scripts/evaluate_remote_policy_automatic.py` | `evaluation/sim/legacy/evaluate_remote_policy_automatic.py` | `git mv`; merge candidate |
| `sim_mujoco/scripts/evaluate_remote_policy_interactive.py` | `evaluation/sim/legacy/evaluate_remote_policy_interactive.py` | `git mv`; merge candidate |
| `sim_mujoco/remote_policy_evaluation.py` | `evaluation/sim/legacy/remote_policy_evaluation.py` | `git mv`; merge shared portions only |
| `sim_mujoco/scripts/run_remote_policy_dry_loop.py` | `tools/evaluation_sim/run_remote_policy_dry_loop.py` | `git mv` |
| `sim_mujoco/scripts/test_remote_policy_mujoco.py` | `tools/evaluation_sim/test_remote_policy_mujoco.py` | `git mv` |
| `sim_mujoco/scripts/test_remote_policy_once.py` | `tools/policy_runtime/test_remote_policy_once.py` | `git mv` |
| `policy_runtime/` | `policy_runtime/` | keep top-level shared package in Phase 2 |

### 8.5 Diagnostics and real-sim analysis

| Current file(s) | Future directory | Action |
| --- | --- | --- |
| `sim_mujoco/gripper_slip_diagnostics.py` | `diagnostics/simulation/gripper/core.py` | `git mv` |
| `analyze_grip_force_vs_width.py`, `run_grip_force_vs_width.py` | `diagnostics/simulation/gripper/` | `git mv` each |
| `analyze_menagerie_forcerange_tuning.py`, `run_menagerie_gripper_validation.py` | `diagnostics/simulation/gripper/` | `git mv` each |
| `analyze_policy_gripper_slip_matrix.py`, `analyze_scripted_gripper_slip_experiments.py`, `run_scripted_gripper_slip_experiments.py` | `diagnostics/simulation/gripper/` | `git mv` each |
| `analyze_xarm_slip_trace.py`, `diagnose_xarm_slip.py` | `diagnostics/simulation/gripper/` | `git mv` each |
| `analyze_friction_ablation.py`, `run_friction_ablation.py`, `prepare_friction_policy_variant.py` | `diagnostics/simulation/contact/legacy/` | `git mv`; frozen/legacy |
| `analyze_friction_search.py`, `run_friction_search.py` | `diagnostics/simulation/contact/legacy/` | `git mv`; frozen/legacy |
| `analyze_split_pad_geometry_experiment.py`, `run_split_pad_geometry_experiment.py` | `diagnostics/simulation/contact/legacy/` | `git mv`; frozen/legacy |
| `analyze_contact_model_realism_regression.py`, `run_contact_model_realism_regression.py` | `diagnostics/simulation/contact/` | `git mv` each |
| `config/diagnostics/legacy_split_pad_camera_calibration.yaml` | `configs/diagnostics/simulation/legacy_split_pad_camera_calibration.yaml` | `git mv` |
| camera calibration/discovery/evaluation/refinement/render/selection/tuning scripts and `test_dual_cameras.py` | `diagnostics/simulation/camera/` | `git mv` each; see script taxonomy below |
| `audit_generation_evaluation_physics.py` | `diagnostics/simulation/audit_generation_evaluation_physics.py` | `git mv` |
| `audit_kinematic_mapping.py` | `diagnostics/simulation/audit_kinematic_mapping.py` | `git mv` |
| `analyze_real_sim_gripper_trajectories.py`, `audit_real_sim_gripper_behavior.py`, `compare_real_sim_datasets.py` | `diagnostics/real_sim/` | `git mv` each |
| `summarize_baseline_final.py` | `diagnostics/real_sim/summarize_baseline_final.py` | `git mv` |

The exact camera set is:
`calibrate_cameras.py`, `camera_calibration_lib.py`,
`discover_raw_camera_data.py`, `evaluate_camera_calibration.py`,
`refine_base_camera.py`, `render_calibration_frame.py`,
`render_current_camera_comparisons.py`,
`select_camera_calibration_frames.py`, `test_dual_cameras.py`,
`tune_base_camera_interactive.py`, and `tune_base_roll_fovy.py`.

### 8.6 Cluster, tests, and docs

| Current | Future | Action |
| --- | --- | --- |
| `slurm/simulation_data/` | `cluster/generation_sim/` | `git mv` directory |
| `slurm/export_mujoco_training_videos.sbatch` | `cluster/generation_sim/export_training_videos.sbatch` | `git mv` |
| `slurm/xarm_eval/` | `cluster/diagnostics/` | `git mv` directory; name jobs by actual diagnostic purpose |
| `scripts/check_deltaai_mujoco_environment.py` | `tools/repository/check_deltaai_mujoco_environment.py` | `git mv` |
| `scripts/generate_mujoco_required_files_manifest.py` | `tools/repository/generate_mujoco_required_files_manifest.py` | `git mv` |
| `tests/test_mujoco_collisions.py`, `test_mujoco_gripper_motion.py`, `test_mujoco_joint_mapping.py`, `test_mujoco_paths.py`, `test_mujoco_task_scenes.py`, `test_menagerie_gripper_integration.py` | `tests/simulation/` | `git mv` each |
| data-generation tests listed in section 12 | `tests/data_sim/` | `git mv` each |
| formal/remote evaluation tests listed in section 12 | `tests/evaluation_sim/` | `git mv` each |
| policy-runtime tests listed in section 12 | `tests/evaluation_common/` | `git mv` each |
| diagnostic tests listed in section 12 | `tests/diagnostics/` | `git mv` each |
| `tests/test_openpi_smoke_contract.py` | `tests/training/test_openpi_smoke_contract.py` | `git mv` |
| `docs/simulation_data/` | `docs/data/simulation_generation/` | `git mv` directory, then repair links only |
| `docs/mujoco_openpi_remote_inference_runbook.md` | `docs/evaluation/sim/remote_inference_runbook.md` | `git mv` |
| `docs/mujoco_task_scenes.md` | `docs/simulation/task_scenes.md` | `git mv` |
| `docs/formal_xarm_model_evaluation.md` | `docs/evaluation/sim/formal_xarm_model_evaluation.md` | `git mv` |
| `docs/xarm_slip_diagnosis.md` | `docs/experiments/diagnostics/xarm_slip_diagnosis.md` | `git mv` |
| `docs/step2_openpi_finetuning.md` | `docs/training/openpi_finetuning.md` | `git mv` |
| `docs/commands/real_world_data_collection.md` | `docs/commands/real_world_data_collection.md` | Consolidated command reference; retain external dependency warning |
| `docs/data_collection_tracker.md`, `docs/training_data_tracker_260626.md`, `docs/training_data_tracker_260703.md` | `docs/experiments/data_collection/` | `git mv` individually |
| `docs/real_robot_model_eval_log_260626.md` | `docs/experiments/evaluation/real_robot_model_eval_log_260626.md` | `git mv` |
| `docs/mujoco_migration/` | `docs/experiments/migrations/mujoco/` | `git mv` directory |
| `docs/training_migration/` | `docs/experiments/migrations/training/` | `git mv` directory |

No `tests/data_real/` or `tests/evaluation_real/` source can be populated merely
by moving current tests; those coverage gaps must remain visible.

## 9. Duplicate implementation and configuration audit

### 9.1 Implementation overlap

1. `policy_runtime/evaluation.py` and
   `sim_mujoco/remote_policy_evaluation.py`: the simulator module already
   delegates some labels/summaries but retains overlapping JSON/CSV wrappers.
2. `policy_runtime/recording.py` and simulator evaluation recording: shared
   tiling/video mechanics versus simulator-specific frame capture.
3. `data_generation.collection` and `data_collection.*`: orchestration imports
   simulator collectors while `oracle_controller.py` imports generation config
   and stability, creating a real package cycle.
4. `episode_recorder.py` and `real_raw_recorder.py`: capture mechanics overlap,
   but output contracts differ intentionally; merge helpers only, not schemas.
5. 7D state/action constants and validation appear in `run_pi_xarm.py`, the
   shared writer, simulator converters, policy runtime, and formal evaluation.
6. Task/prompt normalization exists independently in the real runner and
   simulator registries/converters.
7. The older automatic/interactive remote evaluators overlap with the formal
   evaluator, but parity is not established; retain them as legacy first.
8. `common.sh` and `common_v4_10x.sh`, plus paired generation/conversion/audit
   jobs, duplicate cluster orchestration for versioned plans.

### 9.2 Multiple configuration sources

| Value | Potential canonical source | Other occurrences / classification |
| --- | --- | --- |
| Solver, timestep, contact/friction, actuator mechanics | `xarm6_pick_scene.xml` generated by the checked-in builders | Builder constants are legitimate generated-source duplication; diagnostic runtime overrides are experiments |
| Arm kinematics/inertia/meshes | vendored `third_party/xarm_ros2` inputs | Generated arm XML is a legitimate checked-in runtime artifact |
| Gripper actuator/driver range | scene builder + canonical XML, interpreted by `gripper_mapping.py` | Test assertions and diagnostic overrides are expected |
| Camera calibration | `config/camera_calibration.yaml` | Static MJCF camera entries are build output; legacy split-pad YAML is frozen |
| Object/task poses and Place held/free semantics | `task_scenes.yaml` + `task_scenes.py` | Generation plans and formal protocols select/override tasks but must not redefine semantics |
| Success thresholds | `task_scenes.yaml` for scene semantics; formal protocol JSON for evaluation protocol | Oracle defaults/data-generation config are separate workflow thresholds and need owner-by-field review |
| State/action dimension/order | future `data/common` contract; today shared writer/policy schemas are strongest evidence | Duplicated local constants must remain byte-for-byte equivalent until extraction |
| Dataset schema | `xarm_lerobot_writer.py` | Input adapters may legitimately describe raw schemas |
| Output roots | `sim_mujoco/paths.py` | YAML plans, formal protocols, scripts, and Slurm also specify run-specific output locations |
| Model identities A-D | `formal_models/*.json` | Intentional formal test identities, not duplication |
| Protocol v1/v2/smoke/video-all | six formal protocol JSON files | Versioned/frozen configs; possible future base-plus-overrides only after provenance guarantees |
| Generation v3/v4 | two generation YAMLs | Versioned/frozen plans; retain both |
| Cluster paths | Slurm common files/jobs | Machine-specific but repeatedly hard-coded; future env/config parameters |

## 10. `sim_mujoco/scripts` taxonomy

Every tracked script is accounted for below. Files named `test_*` here are
developer/integration CLIs, not automatically members of the pytest suite.

| Purpose | Scripts |
| --- | --- |
| Formal simulation evaluation | `evaluate_xarm_policy.py` |
| Legacy simulation policy evaluation | `run_remote_policy_closed_loop.py`, `evaluate_remote_policy_automatic.py`, `evaluate_remote_policy_interactive.py` |
| Evaluation review/validation/reporting | `build_xarm_human_review_manifest.py`, `reclassify_xarm_evaluation_failures.py`, `review_xarm_human_videos.py`, `summarize_xarm_evaluation.py`, `summarize_xarm_human_review.py`, `validate_xarm_abc_evaluation.py`, `validate_xarm_category_video_coverage.py` |
| Simulation generation/legacy collection | `collect_oracle_data.py`, `collect_real_raw_sim_data.py`, `convert_mujoco_to_lerobot.py`, `diagnose_place_initial_grasp.py`, `test_scripted_oracle.py`, `validate_real_raw_sim_data.py` |
| Dataset export/validation | `build_real_dataset_schema.py`, `export_lerobot_training_videos.py`, `prepare_mujoco_hf_ready.py`, `upload_mujoco_dataset_to_hf.py`, `validate_mujoco_lerobot_dataset.py` |
| Camera diagnostics | the eleven-file camera set enumerated in section 8.5 |
| Gripper/contact diagnostics | all `run_*`/`analyze_*` friction, split-pad, force, Menagerie, scripted-slip, policy-slip, and xArm-slip pairs; `prepare_friction_policy_variant.py`; `diagnose_xarm_slip.py` |
| Real-sim analysis | `analyze_real_sim_gripper_trajectories.py`, `audit_real_sim_gripper_behavior.py`, `compare_real_sim_datasets.py`, `summarize_baseline_final.py` |
| Physics/kinematics audit | `audit_generation_evaluation_physics.py`, `audit_kinematic_mapping.py` |
| Simulation build/render/dev tools | `build_xarm6_pick_scene.py`, `generate_xarm6_mjcf.py`, `render_task_scenes.py`, `smoke_test_headless_render.py`, `teleoperate_pick.py`, `test_gripper_control.py`, `test_local_observation.py`, `test_local_render.py`, `test_xarm6_control.py` |
| Policy connectivity/dev tools | `run_remote_policy_dry_loop.py`, `test_remote_policy_mujoco.py`, `test_remote_policy_once.py` |

`camera_calibration_lib.py` and several analysis modules are importable
implementations rather than thin wrappers. Run/analyze pairs must remain
separate until a later consolidation explicitly preserves cluster workflows.

## 11. Slurm classification

| Class | Exact jobs |
| --- | --- |
| Simulation generation infrastructure | `simulation_data/common.sh`, `common_v4_10x.sh`, `conversion.sbatch`, `conversion_v4_10x.sbatch`, `final_audit.sbatch`, `final_audit_v4_10x.sbatch`, `full_generation.sbatch`, `full_generation_v4_10x.sbatch`, `offline_tests.sbatch`, `pick_grasp_sweep.sbatch`, `place_grasp_sweep.sbatch`, `place_initial_diagnostic.sbatch`, `smoke.sbatch`, `smoke_v4_10x.sbatch`, `export_mujoco_training_videos.sbatch` |
| Simulation diagnostics | `xarm_eval/analyze_completed_menagerie_gripper.sbatch`, `analyze_menagerie_forcerange_recovery.sbatch`, `analyze_real_sim_gripper_trajectories.sbatch`, `audit_real_sim_gripper_behavior.sbatch`, `diagnose_slip.sbatch`, `diagnose_slip_matrix.sbatch`, `optimize_friction_and_video.sbatch`, `run_friction_ablation.sbatch`, `run_friction_policy_video.sbatch`, `run_grip_force_vs_width.sbatch`, `run_scripted_gripper_slip_suite.sbatch`, `tune_menagerie_gripper.sbatch`, `validate_menagerie_gripper.sbatch` |
| Canonical simulation evaluation | Missing |
| Real collection/evaluation | Missing |
| Training | Missing |

The sweep and offline-test jobs contain inline Python; common scripts contain
substantial validation and orchestration gates. Later phases should move
business validation into tested Python CLIs and keep Slurm responsible for
resources, environment setup, paths, and invoking those CLIs. Hard-coded
`/u/mw89` and `/work/nvme/...` paths should become explicit cluster
configuration without changing defaults during the move.

## 12. Test classification

| Future group | Current exact tests | Notes |
| --- | --- | --- |
| `tests/simulation/` | `test_menagerie_gripper_integration.py`, `test_mujoco_collisions.py`, `test_mujoco_gripper_motion.py`, `test_mujoco_joint_mapping.py`, `test_mujoco_paths.py`, `test_mujoco_task_scenes.py` | Unit/integration simulator contract |
| `tests/data_sim/` | `test_mujoco_data_conversions.py`, `test_mujoco_data_generation.py`, `test_mujoco_episode_recorder.py`, `test_mujoco_hf_safety.py`, `test_mujoco_lerobot_pipeline.py`, `test_mujoco_scripted_oracle.py`, `test_mujoco_v4_10x_plan.py` | Sim raw/canonical generation |
| `tests/evaluation_sim/` | `test_formal_xarm_evaluation.py`, `test_mujoco_chunk_execution.py`, `test_remote_policy_evaluation.py`, `test_remote_policy_pipeline.py`, `test_xarm_category_aware_videos.py`, `test_xarm_human_review.py`, `test_xarm_slip_trace.py` | Formal and legacy simulator evaluation |
| `tests/evaluation_common/` | `test_policy_runtime_actions.py`, `test_policy_runtime_config.py`, `test_policy_runtime_evaluation.py`, `test_policy_runtime_logging.py`, `test_policy_runtime_observation.py`, `test_policy_runtime_recording.py`, `test_policy_runtime_safety.py` | Shared policy runtime, not simulator-owned |
| `tests/diagnostics/` | `test_baseline_final_summary.py`, `test_friction_ablation.py`, `test_grip_force_vs_width.py`, `test_gripper_slip_diagnostics.py`, `test_menagerie_forcerange_tuning.py`, `test_real_sim_gripper_behavior_audit.py`, `test_split_pad_geometry_experiment.py` | Diagnostic and frozen experiment contracts |
| `tests/training/` | `test_openpi_smoke_contract.py` | External OpenPI/data contract |
| `tests/data_real/` | none | Coverage gap |
| `tests/evaluation_real/` | none | Coverage gap |

`test_friction_ablation.py` and split-pad tests depend on frozen experiment
artifacts/semantics and may be skipped when those artifacts are unavailable.
The OpenPI smoke contract requires the external submodule/environment/dataset.
MuJoCo rendering and remote-policy tests have native/GL/server dependencies.
These are prerequisites, not reasons to weaken or delete tests.

## 13. Documentation classification

| Class / future home | Current docs |
| --- | --- |
| Current simulation | `sim_mujoco/README.md`, `mujoco_task_scenes.md`, `mujoco_openpi_remote_inference_runbook.md`, `xarm_slip_diagnosis.md` (diagnostic subsection) |
| Current simulation data | all seven `docs/simulation_data/*` files, `sim_mujoco/DATA_COLLECTION.md`, `data_collection_tracker.md` |
| Current evaluation | `formal_xarm_model_evaluation.md` |
| Current training | `step2_openpi_finetuning.md` |
| Real collection | `commands/real_world_data_collection.md` (must clearly mark external collector), dated training-data trackers |
| Real evaluation experiment record | `real_robot_model_eval_log_260626.md` |
| Migration/handoff record | all `docs/mujoco_migration/*` and `docs/training_migration/*` |
| Generated report snapshot | migration validation/audit/manifest JSON+Markdown and training audit/distribution/post-evaluation JSON+Markdown |
| REVIEW | root `README.md` because it advertises absent/stale paths such as top-level `data_collection/` and tracked Isaac content |

Current user-facing docs should move into stable topical directories. Dated
trackers, execution results, migrations, chat handoffs, and generated audit
snapshots should move under `docs/experiments/` (with subdirectories such as
`migrations/` and `results/`) rather than remain mixed with current guidance.
Deletion should occur only after owners confirm that provenance is preserved by
Git history and a current replacement exists.

## 14. Cleanup candidates and required legacy material

Safe to delete **later only after review**:

- replaced chat handoff/status documents whose only purpose was transfer of a
  completed migration;
- regenerated validation/audit/manifests after their provenance value is judged
  unnecessary;
- `fine_tune/data/.gitignore` if the relocated output tree is completely covered
  by root ignore rules;
- obsolete dataset mutation/inspection tools only after a maintained replacement
  and explicit owner approval.

Must remain legacy/frozen until reproducibility requirements are retired:

- split-pad geometry and friction ablation/search run/analyze pairs;
- `legacy_split_pad_camera_calibration.yaml`;
- associated tests and Slurm jobs;
- versioned v3/v4 generation plans and v1/v2 formal protocols;
- the prior remote evaluation path until formal evaluator parity is demonstrated;
- dated experiment results that are the only record of a physical/simulation run.

Checked-in canonical XML is generated but must remain. No production source is a
Phase 1 or automatic Phase 2 deletion candidate.

## 15. Current dependency graph

Arrows mean “imports/uses.”

```text
root real runner ───────────────> external real_world + OpenPI

sim_mujoco/data_generation ─────> sim_mujoco/data_collection
             │                  ├> sim_mujoco core
             └──────────────────> fine_tune writer

sim_mujoco/data_collection ─────> sim_mujoco/data_generation config/stability
             ├──────────────────> sim_mujoco core
             ├──────────────────> policy_runtime
             └──────────────────> fine_tune writer

sim_mujoco/formal_evaluation ───> sim_mujoco/data_generation registry
             ├──────────────────> sim_mujoco core
             └──────────────────> policy_runtime

sim_mujoco core adapters ───────> policy_runtime
sim_mujoco scripts ─────────────> all of the above
```

The principal violation is the generation/collection cycle. The formal evaluator
also reaches “up” into a generation registry for task/model information.
Simulation core does not currently import generation/evaluation/diagnostics,
which is a critical property to preserve.

## 16. Recommended dependency graph

```text
third_party ───────────────> simulation build tools
                                  │
configs/tasks ────────────────────┤
                                  v
                             simulation
                         /         |        \
                        v          v         v
        data/sim/generation  evaluation/sim  diagnostics/simulation
                 |                 ^                |
                 v                 |                v
data/common <── data/real       policy_runtime  diagnostics/real_sim
    |              |                |
    +-------> canonical datasets    +-------> evaluation/real
                        |                         ^
                        v                         |
                     training              real hardware adapter
```

Rules:

1. `simulation` may use only standard/external libraries, canonical configs, and
   narrowly shared low-level contracts; never data, evaluation, diagnostics, or
   training.
2. `data/common` contains only proven cross-domain serialized contracts/helpers;
   it cannot import sim/real capture implementations.
3. Sim and real data adapters depend inward on `data/common` and converge only at
   the canonical dataset boundary.
4. `policy_runtime` stays hardware/simulator independent. Environment adapters
   implement its protocols.
5. Evaluation common code may define shared identities/result records only where
   both domains implement the same semantics. Success logic stays domain-owned.
6. Diagnostics consume production modules; production never consumes diagnostics.
7. Cluster jobs invoke public CLIs and do not become importable business logic.

## 17. Proposed final tree

Only directories justified by present code are shown. Empty data-real and
evaluation-real test packages should not be created until code exists.

```text
embodied-ai-xarm/
├── simulation/
│   ├── assets/xarm6/
│   ├── environment.py
│   ├── collision.py
│   ├── gripper_mapping.py
│   ├── joint_mapping.py
│   ├── task_scenes.py
│   ├── policy_observation.py
│   ├── policy_control.py
│   ├── recording.py
│   └── paths.py
├── data/
│   ├── common/
│   │   └── xarm_lerobot_writer.py
│   ├── sim/generation/
│   │   └── legacy/
│   └── real/
│       ├── conversion/
│       ├── validation/
│       └── tools/
├── evaluation/
│   ├── common/                 # only proven shared evaluation records
│   ├── sim/
│   │   ├── tools/
│   │   └── legacy/
│   └── real/
│       └── run_policy.py       # existing runner, not yet formal evaluation
├── policy_runtime/
├── training/
│   ├── notebooks/
│   └── tools/
├── diagnostics/
│   ├── simulation/
│   │   ├── camera/
│   │   ├── contact/legacy/
│   │   ├── data_generation/
│   │   └── gripper/
│   └── real_sim/
├── configs/
│   ├── simulation/
│   ├── tasks/
│   ├── data/sim/generation/
│   ├── data/real/
│   ├── evaluation/sim/{models,protocols}/
│   └── diagnostics/simulation/
├── cluster/
│   ├── generation_sim/
│   └── diagnostics/
├── tools/
│   ├── datasets/
│   ├── data_sim/
│   ├── evaluation_sim/
│   ├── policy_runtime/
│   └── simulation/
├── tests/
│   ├── simulation/
│   ├── data_sim/
│   ├── evaluation_common/
│   ├── evaluation_sim/
│   ├── diagnostics/
│   └── training/
├── docs/
│   ├── architecture/
│   ├── simulation/
│   ├── data/
│   ├── evaluation/
│   ├── training/
│   └── experiments/
├── third_party/
├── environment/
└── scripts/
```

## 18. Exact Phase 2 scope and gates

Phase 2 should be a **mechanical core-boundary move**, not the entire tree at
once:

1. Reconfirm clean Phase 1 handoff and baseline ancestry; create a separate
   Phase 2 branch/commit boundary.
2. Move `sim_mujoco` canonical core modules/assets to `simulation/` with
   `git mv`, preserving XML/config bytes and generated-scene build order.
3. Move only camera/task config files required by the core to `configs/`.
4. Update imports and path resolution mechanically for that slice; add temporary
   compatibility imports only if needed to keep downstream modules stable.
5. Do **not** move data generation, formal evaluation, diagnostics, Slurm, tests,
   fine-tune, or real runner in the same change.
6. Run focused simulation/path/task/joint/gripper/collision tests plus XML load,
   scene hashes/invariants, generation/evaluation physics audit, and
   `git diff --check`.
7. Verify high-similarity rename detection (`git diff --summary -M`) and prove
   no configuration/data/physics behavior changed.
8. Stop for review before the data/evaluation migration phases.

Later phases can then move data, evaluation, diagnostics/cluster, tests/docs, and
finally perform explicitly approved consolidation/cleanup. This audit does not
authorize any of those actions.
