# Isaac Sim Repository Audit

Date: 2026-07-24

This audit records the repository state before the Isaac Sim integration. It is
the design baseline for extracting the simulator-independent policy runtime and
adding an Isaac adapter without replacing the working MuJoCo pipeline.

## Repository State

- Branch: `main`, three commits ahead of `origin/main`.
- The existing remote-policy MuJoCo implementation and its tests are
  uncommitted user work. They must be preserved.
- `third_party/openpi` is pinned at commit
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- The validated local interpreter is
  `D:\miniconda\envs\mujoco-pi\python.exe` (Python 3.11.15).
- Baseline verification on 2026-07-24: 18 unit tests passed.
- Isaac Sim modules and common installation directories were not detected.

## Existing MuJoCo Execution Flow

```text
sim_mujoco/assets/xarm6/xarm6_pick_scene.xml
  -> load_simulation()
  -> apply_camera_calibration()
  -> initialize_scene("home")
  -> render base_camera and wrist_camera at 640x480 RGB
  -> OpenPI resize_with_pad to 224x224
  -> read six arm qpos values and convert gripper half-width
  -> construct OpenPI observation
  -> WebsocketClientPolicy.infer()
  -> validate returned (10, 7) action chunk
  -> select actions[0]
  -> clamp joint step, joint/actuator limits, and gripper
  -> apply MuJoCo position-control target
  -> advance 0.02 seconds of simulation
  -> save images, arrays, JSON diagnostics, trajectory, and optional video
  -> optionally request a human success/failure/invalid label
```

The single-inference and dry-loop scripts stop before policy targets are
applied. The closed-loop runner performs preflight checks and executes one
action from each inferred chunk.

## File Responsibilities

### Observation generation and camera calibration

- `sim_mujoco/remote_policy_observation.py`: model loading, camera transforms,
  rendering, gripper conversion, canonical state, OpenPI observation, and
  observation validation.
- `sim_mujoco/config/camera_calibration.yaml`: active calibration and
  preprocessing configuration.
- `sim_mujoco/scripts/camera_calibration_lib.py`: calibration data discovery,
  rendering, comparison, optimization utilities, and JSON output.
- `sim_mujoco/scripts/build_xarm6_pick_scene.py`: applies the active camera
  configuration while producing the pick-scene MJCF.

### Remote inference

- `sim_mujoco/scripts/test_remote_policy_mujoco.py`: one real MuJoCo
  observation and one remote inference.
- `sim_mujoco/scripts/run_remote_policy_dry_loop.py`: repeated inference
  without applying policy targets.
- `third_party/openpi/packages/openpi-client/.../websocket_client_policy.py`:
  synchronous WebSocket transport.
- `docs/mujoco_openpi_remote_inference_runbook.md`: server, tunnel, checkpoint,
  and operating instructions.

### Action decoding and execution

- `sim_mujoco/remote_policy_control.py`: action shape/finiteness checks,
  first-action extraction, maximum joint-step clamp, joint/actuator limit
  clamp, gripper clamp/conversion, and control application.
- `sim_mujoco/scripts/run_remote_policy_closed_loop.py`: preflight, inference,
  target execution, physics stepping, output capture, and termination.

### Safety validation

- Full chunks are currently checked only for exact shape `(10, 7)` and finite
  values.
- Only the first action receives joint-delta, joint-limit, actuator-limit, and
  gripper checks.
- Clipping is recorded but does not reject a target.
- There is no bounded connection/inference timeout, stale-frame check, tracking
  error check, contact threshold, or simulation-progress watchdog.

### Video and interactive evaluation

- `sim_mujoco/remote_policy_evaluation.py`: JSON/CSV serialization, random
  initial conditions, frame layout, OpenCV writer fallback, labels, and
  summary metrics.
- `sim_mujoco/scripts/evaluate_remote_policy_interactive.py`: trial loop,
  video replay, manual labels/comments, resume, and final success summary.

## Policy Contracts

### Canonical state

```text
[joint1_rad, joint2_rad, joint3_rad, joint4_rad,
 joint5_rad, joint6_rad, gripper_raw]
```

- Array shape/dtype: `(7,)`, `float32`.
- Arm joints are MuJoCo qpos values in radians.
- The seventh value is named `gripper_mm` in training metadata and
  `gripper_raw` in the simulator code.
- Repository convention: `50` is closed and `845` is open.

The 50-to-845 value is an xArm SDK position convention. It must not be treated
as a physically measured millimetre aperture until the official gripper model
has been validated.

### OpenPI observation

```python
{
    "observation/image": np.ndarray((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.ndarray((224, 224, 3), dtype=np.uint8),
    "observation/state": np.ndarray((7,), dtype=np.float32),
    "prompt": str,
}
```

MuJoCo returns RGB images. The active path renders at `640x480` and applies
`openpi_client.image_tools.resize_with_pad` to obtain the policy images. No BGR
conversion occurs before inference.

### OpenPI transforms and action semantics

The fine-tuning configuration:

- repacks repository keys into OpenPI keys;
- uses the Libero input/output transforms;
- pads model state/actions to `action_dim=32`;
- uses `action_horizon=10`;
- converts the first six training action dimensions from absolute next-state
  values to deltas before normalization/model input;
- applies the inverse transform after policy output;
- leaves the gripper dimension absolute;
- loads normalization assets from the served checkpoint.

Consequently the simulator receives `(10, 7)` absolute joint/gripper targets.
Simulator adapters must not duplicate server-side padding or normalization.

## Timing

- Training demonstrations: 10 Hz.
- MuJoCo physics timestep: `0.002 s` (500 Hz).
- Current closed-loop command duration: `0.02 s` (10 physics steps).
- Current action prefix: one action per inference.
- Video default: 30 FPS in simulation time.

The `0.02 s` command duration is a validated legacy default but does not match
the 10 Hz demonstration interval. Extraction must preserve this default first;
the new control configuration must expose the period explicitly.

## Active Camera Configuration

The active configuration differs from the approximate base-camera values in
the task description.

Base camera:

```yaml
frame: world
position: [0.6998640343, -0.2034187962, 0.3691466327]
target: [0.3784729309, -0.171520319, 0.0715365659]
roll_deg: -12.5073468556
fovy_deg: 46.7280905796
```

Wrist camera:

```yaml
parent_body: gripper_base
position: [0.06432955545751386, -0.0014285874072126926, 0.08233427275157185]
target: [-0.01648605838780822, 0.014125880549260589, 0.33412490948712087]
roll_deg: -97.06428858407915
fovy_deg: 95.0
```

These values are only an initial estimate for Isaac. MuJoCo uses local negative
Z as the viewing direction; Isaac/USD camera axes, attachment frames, FOV
interpretation, RGBA output, and image orientation must be verified separately.

## Asset Sources

The authoritative robot source is already vendored:

- `third_party/xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro`
- `third_party/xarm_ros2/xarm_description/urdf/xarm6/`
- `third_party/xarm_ros2/xarm_description/urdf/gripper/`
- `third_party/xarm_ros2/xarm_description/meshes/`
- official xArm kinematics and inertial YAML files

The source supports `dof:=6`, `robot_type:=xarm`, and `add_gripper:=true`.
The arm joints are `joint1` through `joint6`; the gripper control joint is
`drive_joint` with mimic joints and `link_tcp` as the tool frame.

The MuJoCo model was derived from the official arm meshes/kinematics but uses a
simplified two-slide-joint gripper. It is not the preferred source for Isaac.

## Simulator-Independent Logic to Extract

- Canonical observation/action schemas and validation.
- OpenPI observation formatting and prompt handling.
- Image preprocessing and diagnostics.
- Remote transport, timeout handling, and timing.
- Action decoding and action-horizon validation.
- Maximum-delta and joint/gripper safety policy.
- Safety result reporting.
- Dry and receding-horizon loop orchestration.
- Structured JSON/JSONL logging and schema versions.
- Episode metadata, labels, scores, summaries, and configuration snapshots.
- Frame layout, codec fallback, frame-directory fallback, and video metadata.

## MuJoCo-Specific Logic to Retain

- MuJoCo imports and object lifecycle.
- MJCF/keyframe loading and reset.
- qpos/actuator/joint-limit lookup.
- MuJoCo camera calibration application and rendering.
- MuJoCo gripper mapping and `data.ctrl`.
- Physics stepping, viewer synchronization, contacts, and object reset.
- Overview-camera capture.

## Technical Debt and Required Refactor

1. `remote_policy_observation.py` mixes simulator access with policy formatting.
2. `remote_policy_control.py` imports MuJoCo even for pure array validation.
3. JSON, image, and response logging are duplicated across scripts.
4. Each script constructs the OpenPI WebSocket client directly.
5. The upstream client retries connection forever and exposes no inference
   watchdog.
6. Safety does not validate every action in a chunk before execution.
7. Clipped targets are always accepted; heavy clipping has no rejection policy.
8. Closed-loop timeout, hold-position, stale-frame, contact, tracking, and
   simulation-progress behavior is missing.
9. Evaluation data lacks simulator/checkpoint/schema/config snapshot fields,
   partial score, and a dedicated failure note.
10. `run_pi_xarm.py` contains another policy formatting/execution path. It is
    outside the initial two-simulator refactor but should later consume the same
    shared runtime.
11. Root dependency files do not fully describe the tested MuJoCo/OpenPI
    environment.

The first implementation stage will add a dependency-free shared core and keep
the existing MuJoCo module names as compatibility wrappers. Isaac modules will
use lazy imports so the ordinary test suite remains usable without Isaac Sim.
