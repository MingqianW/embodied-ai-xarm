# Real-robot evaluation boundary

This package represents only capabilities that exist in this repository. It is
safe to import offline and does not connect to an xArm or camera by import side
effect.

## Present capability

- local OpenPI policy inference through the externally installed OpenPI tree;
- xArm joint/gripper command execution through an external `real_world` wrapper;
- two RealSense RGB observations through an external camera wrapper;
- receding-horizon execution through the canonical action validator with
  xArm6 absolute joint limits, per-command joint-delta limits, and gripper clipping;
- explicit operator/workspace/emergency-stop authorization before motion;
- reverse-order cleanup for every robot/camera lifecycle exit, including
  partial initialization and policy-loading failures;
- raw RGB, state, action-chunk, and metadata logging;
- backend-explicit offline result records and shared human-review decisions.

## Missing or external capability

- no repository-owned xArm SDK adapter or camera driver;
- no object detector, pose estimator, or task-state tracker;
- no automatic real-task success or failure measurement;
- no automatic real failure diagnosis, slip measurement, or representative-video renderer;
- no validated formal real protocol, deterministic reset, trial matrix, or aggregate report;
- hardware limits, emergency stop, collision avoidance, supervision, and safe workspace
  remain operator/site responsibilities.

Accordingly, real results remain `unreviewed` until a human decision is attached.
Telemetry must never be presented as automated task success.
