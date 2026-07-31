# Local Isaac Sim Setup for xArm + OpenPI

## 1. Architecture

The repository has one policy stack and two simulator adapters:

```text
OpenPI WebSocket server
          ↑
policy_runtime/
  observation + image preprocessing + transport + action decoding
  safety + logging + evaluation + recording
          ↑
    ┌─────┴─────┐
MuJoCoEnvironment  IsaacEnvironment
```

`policy_runtime/` owns the exact OpenPI observation keys, RGB `uint8`
preprocessing to `(224, 224, 3)`, the canonical seven-dimensional state/action
schema, bounded WebSocket transport, whole-chunk safety validation, structured
episode output, and reusable video/evaluation utilities.

`sim_isaac/` owns only SimulationApp lifetime, USD/articulation access, scene
objects, camera acquisition, physics stepping, gripper conversion, contact and
stability diagnostics, and Isaac reset/recording hooks. `sim_mujoco/` retains
the corresponding MuJoCo-specific responsibilities. The detailed extraction
audit is in [ISAAC_SIM_REPOSITORY_AUDIT.md](ISAAC_SIM_REPOSITORY_AUDIT.md).

The policy-facing state and every decoded action use:

```text
[joint1, joint2, joint3, joint4, joint5, joint6, gripper_raw]
```

The six arm joints are radians. The final field is named `gripper_mm` in older
training/config interfaces, but repository evidence shows xArm SDK position
units: `50` closed and `845` open. It must not be interpreted as a validated
physical aperture. OpenPI padding, normalization, delta transforms, and inverse
action transforms remain server/checkpoint concerns; simulator adapters expose
only the canonical `(7,)` representation. The returned local chunk is expected
to be `(10, 7)` absolute targets.

## 2. Platform and installation assumptions

The supported initial path is one standalone Isaac Sim instance on Windows,
one xArm, one task object, and two RGB cameras. ROS 2, Docker, WSL, Isaac Lab,
depth sensors, synthetic data, and vectorized environments are not required.
Python modules from Isaac Sim must be loaded with its supplied launcher.
NVIDIA's standalone documentation requires creating `SimulationApp` before
other Omniverse imports and uses the distribution's `python.bat` on Windows:
[Standalone Python](https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/manual_standalone_python.html).

This repository does not install Isaac Sim, GPU drivers, or ROS 2. Start with
the read-only diagnostic:

```powershell
cd "D:\2026 summer project\embodied-ai-xarm"

.\sim_isaac\scripts\check_isaac_installation.ps1 `
  -OutputPath .\sim_isaac\output\installation_report.json
```

It reports Windows/Python/CPU details, NVIDIA GPU/driver/VRAM, `nvidia-smi`,
possible Isaac locations and launchers, disk space, Git LFS, optional ROS 2,
xArm source assets, OpenPI imports, and policy endpoint configuration. Exit
code `2` means Isaac Sim was not found; it is a diagnostic result, not a Python
crash.

After installing a compatible standalone Isaac Sim distribution, set only the
current PowerShell session:

```powershell
$env:ISAAC_SIM_PATH = "C:\path\to\isaac-sim"
$IsaacPython = Join-Path $env:ISAAC_SIM_PATH "python.bat"
& $IsaacPython -c "from isaacsim import SimulationApp; print('Isaac import OK')"
```

Do not permanently change `PATH`. The current implementation supports the
current Isaac 6 experimental RTX camera interface and the shipped legacy
camera/controller fallback. NVIDIA marks `isaacsim.sensors.camera` deprecated
in Isaac 6 and recommends `RtxCamera` plus `CameraSensor`:
[Camera sensors](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_camera.html) and
[camera migration guide](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/sensors_camera_to_experimental_rtx.html).

## 3. xArm asset preparation

The authoritative source is the vendored UFACTORY ROS description:

```text
third_party/xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro
third_party/xarm_ros2/xarm_description/meshes/
```

`sim_isaac/config/asset_import.yaml` selects xArm 6 plus the xArm gripper and
defines import options. The pipeline validates the source and 29 discovered
meshes, expands Xacro into a generated URDF, validates required joints, invokes
Isaac's version-isolated URDF importer, and writes a truthful JSON report.

Validate without generating anything:

```powershell
& $IsaacPython .\sim_isaac\scripts\prepare_xarm_asset.py --validate-only
```

Xacro expansion requires a `xacro` executable/module visible to that
interpreter. ROS 2 itself is optional. Once Xacro is available:

```powershell
& $IsaacPython .\sim_isaac\scripts\prepare_xarm_asset.py `
  --expand-xacro --import-usd --headless
```

Expected generated paths:

```text
sim_isaac/generated/xarm6_with_gripper.urdf
sim_isaac/generated/xarm6_with_gripper/xarm6_with_gripper.usda
sim_isaac/generated/asset_validation.json
```

Generated artifacts are ignored by Git. The importer first uses the current
`URDFImporter` API and falls back to legacy Kit commands. NVIDIA's current
import workflow and GUI fallback are documented at
[URDF Importer](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/import_urdf.html).
If programmatic import is incompatible with an installed build, open Isaac
Sim, use **File > Import**, select the generated URDF, apply the options from
`asset_import.yaml`, and save exactly to the configured USD destination. Do
not create an empty placeholder USD.

Inspect articulation names and required base/TCP frames:

```powershell
& $IsaacPython .\sim_isaac\scripts\inspect_xarm_asset.py
```

The command fails if any configured joint or required frame is absent.
`sim_isaac/config/robot.yaml` is the source of truth for canonical/Isaac joint
names, limits, home pose, action mode, gripper conversion, and frame names.
The gripper conversion is linear between policy `[50, 845]` and the imported
`drive_joint` range `[0, 0.85]` rad. Its physical aperture is explicitly marked
unvalidated until it is measured in the imported USD.

## 4. Scene, control, and safety configuration

Configuration files:

- `robot.yaml`: asset, joint/frame mapping, action semantics, limits, home pose,
  and gripper units.
- `cameras.yaml`: separate Isaac base/wrist extrinsics, FOV, resolution, clips,
  RGB orientation, crop, flip, and shared resize settings.
- `control.yaml`: physics/render/policy rates, endpoint, timeout, chunk prefix,
  safety thresholds, and recording limits.
- `tasks.yaml`: table, object, reset randomization, prompt, and lift scoring.
- `asset_import.yaml`: source Xacro, generated paths, arguments, importer
  options, and report path.

Precedence is:

```text
explicit CLI option
→ environment variable
→ selected YAML/local config
→ repository default
```

Supported environment variables are:

```text
ISAAC_SIM_PATH
OPENPI_POLICY_HOST
OPENPI_POLICY_PORT
OPENPI_CHECKPOINT
XARM_ASSET_PATH
ISAAC_OUTPUT_DIR
```

No credentials are stored. The default control path is 120 Hz physics, 30 Hz
rendering, and 10 Hz policy updates. Only one action is executed per inference;
CLI permits a conservative prefix of 1–5.

The whole `(H, 7)` chunk is validated sequentially before any action executes.
Checks include finite state/actions, shape, explicit absolute/delta mode,
configured joint limits, maximum per-step joint delta, gripper bounds, and a
heavy-clipping rejection threshold. In delta mode, shared safety converts the
validated sequence to absolute canonical targets before either simulator
boundary; adapters never infer semantics from magnitude. The Isaac adapter
additionally reports:

- finite articulation state and world transforms;
- joints within configured limits;
- maximum target tracking error;
- simulation time advancement;
- camera frame age;
- object/table penetration or tunneling indication;
- TCP/table clearance;
- maximum per-step contact impulse when force telemetry is available;
- real-time factor degradation.

`require_contact_impulse_sensor: false` is the honest default because contact
force buffers depend on the generated USD and installed Isaac version. The
limit is enforced whenever force data are available. Set it to `true` only
after `inspect_xarm_asset.py` reports usable telemetry; otherwise closed-loop
control intentionally stops. A timeout calls `hold_position()`. A rejected
chunk is not executed and its state, camera observations, chunk, and exact
reason are saved.

Isaac articulation actions follow NVIDIA's controller model:
[Articulation controller](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/articulation_controller.html).
ROS 2 is deliberately outside this first control path.

## 5. Cameras and observation validation

The active MuJoCo calibration remains unchanged. Isaac extrinsics in
`cameras.yaml` are separate initial estimates converted from MuJoCo's
negative-Z optical-frame convention. They are not claimed calibrated. In
particular, the imported wrist-link hierarchy and camera optical axis must be
verified visually.

The conservative default is two 320×240 RGB cameras and no depth, segmentation,
lidar, or synthetic-data annotators. Both raw captures pass through exactly the
same `policy_runtime.image_preprocessing.preprocess_policy_image()` used by
MuJoCo and OpenPI's `resize_with_pad`, producing RGB `uint8` 224×224 images.

Capture raw/policy views and a JSON report:

```powershell
& $IsaacPython .\sim_isaac\scripts\inspect_cameras.py `
  --output-dir .\sim_isaac\output\camera_inspection

& $IsaacPython .\sim_isaac\scripts\test_observation_pipeline.py --headless
```

Inspect for vertical inversion, mirroring, wrong color order, padding, wrist
occlusion, black frames, incorrect FOV, and target visibility. Change only
`sim_isaac/config/cameras.yaml`; never overwrite the MuJoCo calibration.

Compare the two simulators directly when both dependencies are available:

```powershell
& $IsaacPython .\sim_isaac\scripts\compare_mujoco_isaac.py `
  --capture-mujoco --capture-isaac `
  --output-dir .\sim_isaac\output\camera_comparison
```

Alternatively pass `--mujoco-base`, `--mujoco-wrist`, `--isaac-base`, and
`--isaac-wrist` PNG paths. `--landmarks landmarks.json` accepts:

```json
{
  "base": {"mujoco": [[100, 80]], "isaac": [[102, 79]]},
  "wrist": {"mujoco": [], "isaac": []}
}
```

The utility writes side-by-side and alpha-overlay PNGs plus dimensions, image
statistics, landmarks, and mean absolute pixel differences in
`comparison.json`.

## 6. Policy server and dry loop

Start the existing π0.5 checkpoint server using the MuJoCo runbook, then expose
it locally (normally `127.0.0.1:18000`) through the existing SSH tunnel. The
Isaac adapter does not change checkpoint normalization or server-side OpenPI
transforms.

Dry inference captures and validates observations and policy chunks but never
applies articulation targets:

```powershell
& $IsaacPython .\sim_isaac\scripts\run_policy_dry_loop.py `
  --host 127.0.0.1 `
  --port 18000 `
  --prompt "pick up the object" `
  --iterations 5 `
  --camera-debug `
  --seed 0 `
  --inference-timeout 10 `
  --output-dir .\sim_isaac\output\dry_loop
```

The event stream logs image/state shapes, dtype, RGB ordering, ranges, prompt,
raw response, decoded action shape/range/finiteness, whole-chunk safety,
inference latency, and total latency. Arrays use `.npy`; metadata uses
schema-versioned JSON/JSONL. Invalid observations/actions and dependency or
connection failures return nonzero exit codes.

## 7. Safe closed loop

Begin with one action per inference:

```powershell
& $IsaacPython .\sim_isaac\scripts\run_policy_closed_loop.py `
  --host 127.0.0.1 `
  --port 18000 `
  --prompt "pick up the object" `
  --max-policy-steps 20 `
  --execute-chunk-steps 1 `
  --record `
  --output-dir .\sim_isaac\output\closed_loop
```

The loop resets, observes, infers with a bounded timeout, validates the entire
chunk, executes only the selected prefix, steps at the configured rates,
reobserves, and replans. It holds on timeout or rejection and stops on
instability. Do not increase the prefix to 2–5 until one-action behavior,
camera calibration, gripper direction, contact telemetry, and tracking error
are verified.

Closing the Isaac application stops its runtime; `Ctrl+C` remains the emergency
operator stop. Use `--headless` for smoke tests. A visible run is required
before trusting contact-rich behavior.

## 8. Interactive evaluation and video

```powershell
& $IsaacPython .\sim_isaac\scripts\run_interactive_evaluation.py `
  --episodes 5 `
  --max-policy-steps 100 `
  --execute-chunk-steps 1 `
  --record
```

Each timestamped episode asks for `success`, `failure`, or `invalid`, permits a
0–1 partial score and short note, records automatic lift score and termination
reason, and continues. `--non-interactive` uses the configured lift threshold.
The run writes per-episode `episode.json`, `evaluation.json`, `events.jsonl`,
action arrays, a video or PNG fallback, final `episodes.csv`, and
`summary.json`.

Recording tiles a primary/base view and wrist view into a bounded 640×480
stream. It tries MP4 (`mp4v`) and falls back to a PNG directory when encoding
is unavailable. `max_frames` prevents unbounded disk use. Raw frames are not
retained unless the selected fallback or camera-debug mode requires them.
Outputs default below `sim_isaac/output/`, or `$env:ISAAC_OUTPUT_DIR`.

## 9. Tests

Level 1—ordinary Python, no Isaac or server:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -m unittest discover -s tests -v
& "D:\miniconda\envs\mujoco-pi\python.exe" -m unittest discover -s sim_isaac/tests -v
```

This covers config, mappings, gripper round trips, exact observations, action
shape/semantics, safety, image preprocessing, evaluation/recording, MuJoCo
compatibility, and import behavior. Isaac-dependent cases remain skipped.

Level 2—Isaac installed, no policy server:

```powershell
$env:RUN_ISAAC_TESTS = "1"
& $IsaacPython -m unittest sim_isaac.tests.test_isaac_runtime -v
Remove-Item Env:RUN_ISAAC_TESTS
```

This loads the asset, validates joint names/home pose, checks deterministic
reset, commands each arm joint and both gripper endpoints, renders both
cameras, validates observations, steps, and checks safety.

Level 3—Isaac plus policy server:

```powershell
$env:RUN_ISAAC_POLICY_TESTS = "1"
$env:OPENPI_POLICY_HOST = "127.0.0.1"
$env:OPENPI_POLICY_PORT = "18000"
& $IsaacPython -m unittest sim_isaac.tests.test_policy_integration -v
Remove-Item Env:RUN_ISAAC_POLICY_TESTS
```

This covers repeated dry inference, a short receding-horizon run, recording and
metadata, synthetic timeout, and rejected-action behavior. Pytest users can
select the declared `isaac`, `integration`, and `policy_server` markers. These
tests are never enabled in the default suite.

## 10. Adding a task

Add a named entry below `tasks:` in `sim_isaac/config/tasks.yaml` with prompt,
object type/name/size/mass/pose, table geometry, deterministic randomization
ranges, and lift/partial thresholds. Then run:

```powershell
& $IsaacPython .\sim_isaac\scripts\test_observation_pipeline.py `
  --prompt "new task instruction"
```

The initial scene adapter supports a single dynamic cube and fixed table. New
geometry types belong in `object_spawning.py`; policy formatting and safety do
not change.

## 11. VRAM and performance

The target laptop reports roughly 12 GB VRAM. Defaults intentionally use one
environment, two low-resolution RGB views, and no depth/semantic/lidar/domain
randomization. This configuration has not been run on this machine because
Isaac Sim is currently absent, so no memory-fit claim is made.

For out-of-memory failures:

1. Close other GPU applications and inspect `nvidia-smi`.
2. Keep one environment and disable recording.
3. Remain at or below 320×240 cameras.
4. Run headless and avoid viewer/render products not used by the policy.
5. Reduce render frequency while keeping physics at 60–120 Hz.
6. Disable unused Isaac extensions through the installed app configuration.
7. Inspect Isaac logs for allocation failures; do not weaken physics safety to
   hide an OOM.

Real-time-factor warnings are diagnostic, not grounds to execute a larger
action prefix.

## 12. Troubleshooting and known limitations

`IsaacDependencyError` or missing `isaacsim`:
: Use the installed distribution's `python.bat`, not Conda Python. Run the
  diagnostic and set `ISAAC_SIM_PATH`.

Generated USD missing:
: Run source validation, install/make Xacro visible, expand, then import. The
  repository deliberately contains no fabricated USD.

URDF importer API mismatch:
: Use the documented GUI fallback and configured import settings; inspect
  joints/frames afterward.

Missing `link_tcp` or `drive_joint`:
: Do not guess a replacement. Inspect the USD and update the explicit mapping
  only after confirming the authoritative name.

Camera black/stale or wrong orientation:
: Run `inspect_cameras.py`, verify backend/frame IDs/statistics, then adjust the
  separate Isaac extrinsics/flip/FOV. RTX camera APIs are experimental and may
  change between Isaac releases.

Closed loop stops with missing transform:
: The imported USD hierarchy does not expose the configured base/TCP path.
  Correct the asset/mapping; do not bypass transform safety.

Contact impulse unavailable:
: The importer did not expose force buffers. Keep
  `require_contact_impulse_sensor: false` during visual calibration, or author
  contact reporting in the USD and require it before contact-heavy evaluation.

Policy timeout:
: The bounded client holds position and closes its socket. Check the SSH tunnel,
  server allocation, and inference time rather than raising the timeout
  without investigation.

Known unvalidated items:

- Isaac Sim is not installed on the current machine, so generated USD, camera
  runtime, GPU memory, physics gains, collision fidelity, and recording codec
  have not been exercised locally.
- Isaac camera transforms are estimates pending saved-image comparison.
- The gripper's linear SDK-unit-to-radian mapping is not a measured aperture
  calibration.
- Contact impulses depend on force buffers in the generated articulation.
- The scene currently implements one cube-lift task, not the full MuJoCo task
  catalog.
- Evaluation success is a lift heuristic plus human label, not a learned task
  metric.

## 13. Future ROS 2 and Isaac Lab work

ROS 2 can later be added as a separate actuator/observation transport behind
the same `RobotEnvironment` protocol. Keep canonical units and mappings at that
boundary, add ROS bridge diagnostics, and compare direct articulation targets
against ROS controller targets before switching evaluation. Do not place ROS
messages in `policy_runtime/`.

For Isaac Lab, wrap the same canonical observation/action functions in one
single-environment task first. Preserve `robot.yaml`, `cameras.yaml`, episode
schema, safety validator, and OpenPI client. Only after parity should the scene
be vectorized or randomized. Isaac Lab evaluation should remain comparable to
the current MuJoCo and standalone Isaac rows rather than introduce a third
policy stack.
