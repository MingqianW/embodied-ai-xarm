# xArm MuJoCo Camera Calibration

This directory contains the completed local calibration run for the fixed base
camera (`realsense_0`) and wrist camera (`realsense_1`). The workflow never
connects to DeltaAI and does not run OpenPI inference.

## Environment

The run used `D:\miniconda\envs\mujoco-pi\python.exe`, Python 3.11.15. NumPy
was 1.26.4, and `python -m pip check` reported no broken requirements. No
dependencies were installed or upgraded. OpenCV images are converted from BGR
to RGB immediately after loading.

Verify the interpreter before reproducing the run:

```powershell
conda activate mujoco-pi
where python
python --version
python -m pip check
```

Do not run these commands from the old Python 3.13 environment.

## Raw Data Discovery

The source data is under `fine_tune/data/xarm_pi05_data/raw`. It contains 200
episodes and 23,096 synchronized robot rows across six tasks. Every row has
valid base and wrist images. All source images are RGB PNG files at 640x480.

The state order is `j1_rad, j2_rad, j3_rad, j4_rad, j5_rad, j6_rad,
gripper_mm`; the six arm values are radians. Despite their names, the three
`tcp_*_m` translation fields contain millimeters. See `dataset_discovery.json`
for the complete discovered schema and counts.

## Frame Selection

`selected_frames.json` contains 16 sharp, pose-diverse samples from 16 distinct
episodes: 12 calibration frames and four held-out validation frames. Selection
filters candidates by Laplacian sharpness, edge content, valid paired images,
and wrist-view contrast, then uses farthest-point sampling in normalized
six-joint space. The manifest records image paths, task, timestamp, source
resolution, joint values, and gripper value. It also stores a raw-data file
snapshot used to verify that the originals were not changed.

## Parameterization And Search

The base camera uses world-frame position, look-at target, optical-axis roll,
and vertical FOV. The wrist camera uses the same representation relative to
`gripper_base`. Translations are meters; internal angle calculations use
radians, while YAML roll and FOV values are degrees.

The geometric objective is a weighted symmetric distance-transform loss over
Canny edges. It emphasizes simulated-to-real edge alignment, includes edge
density and dark-occlusion penalties, and adds a wrist upper-boundary profile
term. Camera-to-target distance is weakly regularized. A seeded bounded random
search is followed by coordinate refinement; no RGB MSE or large learned model
is used.

Final parameters in `../config/camera_calibration.yaml`:

```yaml
base_camera:
  position: [0.4527602407, -0.6664764088, 0.4859253368]
  target: [0.4099162474, -0.0344442504, 0.0108]
  roll_deg: 3.1023791743
  fovy_deg: 43.8814065008
wrist_camera:
  parent_body: gripper_base
  position: [0.0674069555, 0.0057714126, 0.0865342728]
  target: [0.0018739416, -0.0168701195, 0.3221249095]
  roll_deg: -96.0922885841
  fovy_deg: 90.3695
```

## Results

Lower geometric loss is better.

| Split | Camera | Before | After | Improvement |
|---|---:|---:|---:|---:|
| Calibration (12) | Base | 0.4511 | 0.2609 | 42.2% |
| Calibration (12) | Wrist | 1.4730 | 1.2246 | 16.9% |
| Validation (4) | Base | 0.4192 | 0.2376 | 43.3% |
| Validation (4) | Wrist | 3.7216 | 0.3742 | 89.9% |

The contact sheets in `contact_sheets/` show real, simulated, and blended views
for every frame. The final views are upright and not mirrored; the tool remains
on the correct side, the wrist view looks through the fingers, and the task
workspace remains visible across all selected poses. Native renders are saved
at 640x480. Policy images are created afterward with OpenPI
`resize_with_pad(..., 224, 224)`, preserving the 4:3 image without stretching.

## Reproduce

Run from the repository root after activating `mujoco-pi`:

```powershell
python sim_mujoco/scripts/discover_raw_camera_data.py
python sim_mujoco/scripts/select_camera_calibration_frames.py --calibration-count 12 --validation-count 4 --max-episodes 36
python sim_mujoco/scripts/calibrate_cameras.py --trials 120 --optimization-width 160 --optimization-height 120
python sim_mujoco/scripts/build_xarm6_pick_scene.py
python sim_mujoco/scripts/evaluate_camera_calibration.py
```

Regenerate only the calibrated scene:

```powershell
python sim_mujoco/scripts/build_xarm6_pick_scene.py
```

Render one selected frame at native and policy resolution:

```powershell
python sim_mujoco/scripts/render_calibration_frame.py --sample-id sample_000 --camera both
```

The scene generator reads only `sim_mujoco/config/camera_calibration.yaml` for
camera parameters. `baseline_camera_calibration.yaml` preserves the original
hard-coded cameras, and `backups/` contains the pre-calibration scene generator.

## Outputs And Limitations

`before/`, `after/`, `overlays/`, and `contact_sheets/` contain the visual
comparisons. `calibration_metrics.json` contains aggregate and per-frame scores;
`logs/optimization_history.json` preserves the search history; and
`validation_report.json` records model, render-size, actuator, camera-config,
and raw-snapshot checks.

The remaining mismatch is both geometric and photometric. The simulated
gripper is a simplified dark cylinder and box-finger model, while the real
gripper has a large white rectangular housing and clear finger guards. The
simulated table, lighting, background, and objects also differ from the lab.
RealSense lens distortion and exact intrinsics were unavailable, and the
single simulated object is not reconstructed from each real frame. These
differences limit edge alignment, especially on calibration wrist frames, but
the held-out geometry and orientation remain stable. Improving the gripper
mesh/materials and reconstructing per-frame objects would address the largest
remaining errors without changing the calibrated camera interface.
