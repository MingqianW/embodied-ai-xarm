# MuJoCo Task-Specific Scenes

The MuJoCo evaluation scene now contains an object catalog and activates a
task-specific subset at reset. The camera calibration, table, robot model, and
control semantics remain shared across tasks.

The source of truth is:

- Scene configuration: `simulation/config/task_scenes.yaml`
- Runtime scene selection: `simulation/scene/`
- Object geometry generation: `python -m simulation.tools.build_xarm6_pick_scene`
- Compiled scene: `simulation/assets/xarm6/xarm6_pick_scene.xml`

## Raw-data correspondence

The object selection and initial layouts were derived from the first
`realsense_0` frames under `fine_tune/data/xarm_pi05_data/raw`.

| MuJoCo task | Raw-data task | Active scene |
| --- | --- | --- |
| `red_block` | `pick_up_the_red_block` | one red block |
| `blue_block` | `pick_up_the_blue_block` | one blue block |
| `largest_block` | `pick up the largest block` | small red block and large blue block; blue is the target |
| `smallest_block` | `pick_up_the_smallest_block` | the same two-block layout; red is the target |
| `red_pepper` | `pick_up_the_red_pepper` | one compound-geometry red bell pepper |
| `place_red_pepper_in_ring` | `place_the_red_pepper_in_the_ring` | pepper initially held by the gripper and a white receiving ring |

The block dimensions, pepper shape, ring shape, and world positions are
simulation approximations. They reproduce the task semantics and visual cues
seen in the raw frames; they are not metrology-grade reconstructions of the
physical props.

## Preview every scene

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\render_task_scenes.py" `
  --task all `
  --seed 0
```

The command writes native base and wrist images plus initial-condition metadata
under `sim_mujoco/output/task_scene_preview/<task>/`.

## Inspect collision geometry

The xArm base and links 1-6 use simplified cylinder, capsule, and ellipsoid
collision geometry. The STL meshes are visual-only. The gripper base, fingers,
table, floor, and task objects also participate in collision detection.

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\render_task_scenes.py" `
  --task red_block `
  --show-collisions `
  --output-dir ".\sim_mujoco\output\collision_preview"
```

The generated `overview_collisions.png` hides the visual arm meshes and shows
the robot collision volumes in green. `base.png` and `wrist.png` retain a
semi-transparent visual/collision overlay.

Directly adjacent bodies are excluded from self-collision because their
primitive volumes overlap around joints by design. Contacts between
non-adjacent robot bodies are classified as `self_collision`; contacts between
the robot and table or floor are classified as `robot_support_collision`
(except the fixed base touching the floor). Either condition terminates a
closed-loop episode and is recorded in `diagnostics.json` and `result.json`.
Robot-object contacts remain allowed because they are required for grasping.

## Run one task

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\run_remote_policy_closed_loop.py" `
  --task largest_block `
  --max-policy-steps 10 `
  --execute-chunk-steps 1
```

The exact task label stored in raw `meta.json` is selected automatically as the
policy prompt, including underscores where the training data uses them.
`--prompt` is available only when an intentional prompt override is needed.

At reset, each task declares the bodies that must appear in the initial wrist
camera view. Pick tasks require their target object; size-comparison tasks
require both candidate blocks; the place task requires both the held pepper
and the ring. After settling, the reset checks these body centers against a
90% camera-frustum margin and refuses to start an episode if a required body is
outside the wrist view. The resulting normalized image coordinates are stored
under `initial_conditions.wrist_visibility` in `result.json`.

The same reset also requires `link1` through `link6`, the gripper base, and both
fingers to remain inside the base-camera safety margin. Those checks are stored
under `initial_conditions.base_robot_visibility`.

## Evaluate the full task suite

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\evaluate_remote_policy_interactive.py" `
  --task all `
  --episodes 10 `
  --max-policy-steps 80 `
  --execute-chunk-steps 1
```

Here, `--episodes 10` means ten episodes per task. Seeds reproduce the object
translation, object yaw, and initial joint noise. Results are grouped by task
and `summary.json` includes both the global result and `task_breakdown`.

## Success criteria

Pick tasks succeed when the configured target rises at least `0.05 m` above its
settled initial height for three consecutive policy steps. The place task
succeeds after the pepper has been released and remains inside the configured
ring radius at table height for three consecutive policy steps.

Automatic success is recorded in every step's `diagnostics.json` and the
episode `result.json`. Interactive human labels remain the final evaluation
label so that unstable grasps and visually invalid episodes can be rejected.
