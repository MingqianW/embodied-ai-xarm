# Diagnostics

This package contains the small set of maintained checks that are still useful
to the current xArm system. Diagnostics may observe simulation ground truth and
compare offline datasets, but they do not define task success, change policy
observations, or control real hardware. Camera `fit` is the one explicit
exception to read-only operation and writes only simulation camera calibration.

| Area | Maintained entry point | Boundary |
| --- | --- | --- |
| Compiled physics | `python -m diagnostics.simulation.physics.consistency` | Exact generation/evaluation comparison; no rendering or stepping |
| Evaluation trace | `diagnostics.simulation.gripper.trace` | Stable measurement primitive used by formal evaluation |
| Slip analysis | `python -m diagnostics.simulation.gripper.analyze_trace` | Data-only analysis of completed CSV traces |
| Camera calibration | `python -m diagnostics.simulation.camera.cli` | Offline raw-image/render checks; only `fit` writes simulation camera config |
| Real-sim gripper | `python -m diagnostics.real_sim.gripper.behavior` | Offline state/label comparison with explicit identifiability limits |

The package deliberately contains no experiment matrix runners, cluster
launchers, hardware control, training jobs, or legacy implementation archive.
Git history is the archive for retired diagnostics.
