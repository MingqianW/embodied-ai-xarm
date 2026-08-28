# MuJoCo Remote Policy Pipeline

This directory contains the xArm6 MuJoCo scene, calibrated camera setup, and
safe remote Pi0.5 policy runners.

The simulator-independent observation, image preprocessing, bounded OpenPI
transport, action decoding/safety, logging, recording, and evaluation schema
now live in `policy_runtime/`. `sim_mujoco/environment.py` implements the same
environment protocol as `sim_isaac/environment.py`; the existing
`remote_policy_*` modules remain compatibility-facing MuJoCo helpers.

For the full operator/Codex procedure, see
[`docs/mujoco_openpi_remote_inference_runbook.md`](../docs/mujoco_openpi_remote_inference_runbook.md).
For task-specific object layouts and full-suite evaluation, see
[`docs/mujoco_task_scenes.md`](../docs/mujoco_task_scenes.md).

## Prerequisites

- Python environment: `D:\miniconda\envs\mujoco-pi\python.exe`
- The MuJoCo pick scene: `sim_mujoco\assets\xarm6\xarm6_pick_scene.xml`
- Camera calibration: `sim_mujoco\config\camera_calibration.yaml`
- OpenPI client importable in the `mujoco-pi` environment.
- An SSH tunnel from local `127.0.0.1:18000` to the remote policy server.
- The remote Pi0.5 policy server already running on DeltaAI.

Run commands from the repository root:

```powershell
cd "D:\2026 summer project\embodied-ai-xarm"
```

## Manual Pick-Up Teleoperation

The xArm6 scene uses the UFACTORY xArm four-bar gripper architecture from the
MuJoCo Menagerie xArm7 model: paired driver hinges, follower fingers, inner
spring links, a split tendon, and closed-loop linkage constraints. It retains a
single gripper command and the existing raw xArm gripper state convention.

Start the interactive scene below, then click the MuJoCo viewer and use the
keyboard to move one joint at a time. The default scene contains the red block;
use `--task blue_block` or another task key to choose a different object.

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\teleoperate_pick.py" `
  --task red_block
```

Controls: arrow keys move the TCP in the world XY plane, `Page Up/Page Down`
move it along world Z, `O/C` opens/closes the gripper, `H` holds the current
position, `X` resets the task, and `Esc` quits. Adjust motion increments with
`--cartesian-step-mm` and `--gripper-step-mm` when needed.

Teleoperation uses only MuJoCo contact dynamics: objects are never attached to
the TCP or made non-colliding after contact.


## Observation Semantics

The OpenPI observation is:

```python
{
    "observation/image": base_image,
    "observation/wrist_image": wrist_image,
    "observation/state": state,
    "prompt": prompt,
}
```

Both images are RGB `uint8` arrays with shape `(224, 224, 3)`. MuJoCo renders at
the calibrated native resolution from `camera_calibration.yaml`, then applies
OpenPI `resize_with_pad`.

The state is `float32` with shape `(7,)`:

```text
[joint1, joint2, joint3, joint4, joint5, joint6, gripper_raw]
```

Joint values are radians from MuJoCo qpos. `gripper_raw` follows the training
range, where `50` is closed and `845` is open.

## Action Semantics

The remote policy returns actions with shape `(10, 7)`:

```text
[joint1, joint2, joint3, joint4, joint5, joint6, gripper_raw]
```

The first six values are treated as absolute joint targets after OpenPI output
postprocessing. The gripper value is an absolute raw target in the training
range.

Safety rules in `remote_policy_control.py`:

- Reject NaN, Inf, and wrong action shape.
- Initially use only `actions[0]`.
- Clamp each joint to at most `0.05` rad per policy update by default.
- Clamp joints to MuJoCo joint limits and actuator control limits.
- Clamp `gripper_raw` to `[50, 845]`.
- Convert gripper raw to MuJoCo half-width target using the calibrated mapping.
- Report every clipping event.

## Single Inference

This builds one real MuJoCo observation, calls the remote policy exactly once,
does not write `data.ctrl`, and saves the model inputs and result.

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\test_remote_policy_mujoco.py"
```

Outputs:

- `sim_mujoco\output\remote_policy_test\base_model_input.png`
- `sim_mujoco\output\remote_policy_test\wrist_model_input.png`
- `sim_mujoco\output\remote_policy_test\result.json`
- `sim_mujoco\output\remote_policy_test\validation_report.txt`

## Dry Loop

This repeatedly observes, infers, validates, and computes safe clamped targets
without applying policy actions.

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\run_remote_policy_dry_loop.py" `
  --iterations 5
```

Outputs are written one directory per iteration under:

```text
sim_mujoco\output\remote_policy_dry_loop\
```

Each iteration saves `base.png`, `wrist.png`, `observation.json`, `actions.npy`,
and `diagnostics.json`.

## Safe Closed Loop

This is the first script that executes policy actions in simulation. It runs a
preflight check before control and refuses to start unless every required check
passes.

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\run_remote_policy_closed_loop.py" `
  --task red_block `
  --max-policy-steps 20 `
  --execute-chunk-steps 1 `
  --max-joint-step 0.05
```

Use `--headless` to run without a MuJoCo viewer. Stop with `Ctrl+C`; if the
viewer is open, closing it also stops the loop.

`--execute-chunk-steps` accepts `1` through `10`, matching the policy action
horizon. After executing that many targets, the loop captures fresh base/wrist
images and joint state and requests a new action chunk. Start with `5` when
reducing inference overhead; `10` executes the complete predicted chunk before
re-observation.

## Task-Specific Scenes

Supported task keys are `red_block`, `blue_block`, `largest_block`,
`smallest_block`, `red_pepper`, and `place_red_pepper_in_ring`. The selected
task controls the active objects, target object, default training prompt,
initial state, randomization, and automatic success criterion.

Render all initial scenes before remote inference:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\render_task_scenes.py" `
  --task all
```

Add `--show-collisions` to generate a green collision-volume overview. See
[`docs/mujoco_task_scenes.md`](../docs/mujoco_task_scenes.md) for the collision
policy and debug output.

Evaluate every task with ten episodes per task:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" `
  ".\sim_mujoco\scripts\evaluate_remote_policy_interactive.py" `
  --task all `
  --episodes 10
```

## Troubleshooting

- If `openpi_client` cannot be imported, run with the `mujoco-pi` Python path.
- If connection fails, verify the SSH tunnel and remote policy server.
- If images are black or nearly uniform, inspect the saved PNGs and the camera
  calibration config.
- If preflight fails, read the printed PASS/FAIL table; closed-loop control will
  not begin until all checks pass.
- If clipping is frequent, inspect `diagnostics.json` before relaxing any safety
  limit.
