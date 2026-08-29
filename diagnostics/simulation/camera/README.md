# xArm MuJoCo Camera Calibration

This workflow calibrates the fixed base camera (`realsense_0`) and wrist camera
(`realsense_1`) against locally recorded xArm images. It does not connect to
DeltaAI or run OpenPI inference.

## Versioned Inputs

- `../../../simulation/config/camera_calibration.yaml`: active camera extrinsics and FOV.
- `baseline_camera_calibration.yaml`: original cameras used for comparisons.
- `../../../simulation/assets/xarm6/xarm6_pick_scene.xml`: generated scene used by the tools.

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
`../../../sim_mujoco/constraints.txt`, then run from the repository root. Raw
data is resolved through `data.real.config`, including the ignored local
`configs/data/real/xarm_data_config.json` override.

```powershell
python -m diagnostics.simulation.camera.cli discover
python -m diagnostics.simulation.camera.cli select --calibration-count 12 --validation-count 4 --max-episodes 36
python -m diagnostics.simulation.camera.cli fit --trials 120 --optimization-width 160 --optimization-height 120
python -m simulation.tools.build_xarm6_pick_scene
python -m diagnostics.simulation.camera.cli evaluate
```

`fit` is the sole mutating diagnostic command: it writes the active camera
configuration. Review that diff before rebuilding the generated scene. The
other commands do not change simulation calibration or physics values.

Regenerate only the scene after manually changing the active camera config:

```powershell
python -m simulation.tools.build_xarm6_pick_scene
```

Render a selected frame at native and policy resolution:

```powershell
python -m diagnostics.simulation.camera.cli render --sample-id sample_000 --camera both
```

## Generated Outputs

The CLI creates frame manifests, metrics, validation reports, search logs,
renders, overlays, contact sheets, and temporary config snapshots under this
directory. These products are reproducible and ignored by Git. Source raw data
is read-only.

The remaining visual mismatch is partly photometric: the simulated gripper,
objects, table, lighting, and background are simplified, and exact RealSense
intrinsics and distortion were unavailable. Improving those scene assets is
more useful than retaining generated calibration images in source control.
