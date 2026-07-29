# xArm MuJoCo Camera Calibration

This workflow calibrates the fixed base camera (`realsense_0`) and wrist camera
(`realsense_1`) against locally recorded xArm images. It does not connect to
DeltaAI or run OpenPI inference.

## Versioned Inputs

- `../config/camera_calibration.yaml`: active camera extrinsics and FOV.
- `baseline_camera_calibration.yaml`: original cameras used for comparisons.
- `../assets/xarm6/xarm6_pick_scene.xml`: generated scene used by the tools.

Camera translations are in meters. YAML roll and vertical FOV values are in
degrees. The base camera uses world-frame position and target; the wrist camera
uses coordinates relative to `gripper_base`.

## Current Base Camera

```yaml
position: [1.34431495, -0.296122326, 0.707594031]
target: [0.4473593, -0.0402923386, 0.394251153]
roll_deg: 2.0
fovy_deg: 57.479524
```

This places the camera on the positive-X side of the workspace, facing the
front of the arm. From that view, decreasing world Y moves the camera left.
Changing `position` moves the camera, changing `target` changes where it looks,
and `roll_deg` rotates the image around its optical axis.

## Reproduce

Activate an environment containing the packages pinned in
`../constraints.txt`, then run from the repository root:

```powershell
python sim_mujoco/scripts/discover_raw_camera_data.py
python sim_mujoco/scripts/select_camera_calibration_frames.py --calibration-count 12 --validation-count 4 --max-episodes 36
python sim_mujoco/scripts/calibrate_cameras.py --trials 120 --optimization-width 160 --optimization-height 120
python sim_mujoco/scripts/tune_base_roll_fovy.py --final-roll 2 --final-fovy 57.479524
python sim_mujoco/scripts/build_xarm6_pick_scene.py
python sim_mujoco/scripts/evaluate_camera_calibration.py
```

Regenerate only the scene after manually changing the active camera config:

```powershell
python sim_mujoco/scripts/build_xarm6_pick_scene.py
```

Render a selected frame at native and policy resolution:

```powershell
python sim_mujoco/scripts/render_calibration_frame.py --sample-id sample_000 --camera both
```

## Generated Outputs

The scripts create frame manifests, metrics, validation reports, search logs,
renders, overlays, contact sheets, and temporary config snapshots under this
directory. These products are reproducible and ignored by Git. The source raw
data is read-only and remains under `fine_tune/data/`.

The remaining visual mismatch is partly photometric: the simulated gripper,
objects, table, lighting, and background are simplified, and exact RealSense
intrinsics and distortion were unavailable. Improving those scene assets is
more useful than retaining generated calibration images in source control.
