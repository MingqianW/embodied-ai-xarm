# MuJoCo + OpenPI Pi0.5 Remote Inference Runbook

This is the operational reference for running the xArm6 MuJoCo simulation,
connecting it to the remote OpenPI Pi0.5 policy server on DeltaAI, validating
observations and actions, and running a safe closed-loop simulation.

It is intended to be readable by both a human operator and Codex working inside
the repository.

## 1. Repository And Environment

Local repository on the validated Windows setup:

```text
D:\2026 summer project\embodied-ai-xarm
```

On WSL, Linux, or Codespaces the path may differ. First locate the repository
root by finding:

```text
simulation/
third_party/openpi/
tests/
```

Preferred Windows interpreter:

```text
D:\miniconda\envs\mujoco-pi\python.exe
```

Expected Python version: Python 3.11.x.

Important packages:

- `mujoco`
- `numpy`
- `opencv-python`
- `openpi-client`
- `websockets`
- `PyYAML`

The OpenPI client should resolve from the editable local package:

```text
third_party/openpi/packages/openpi-client
```

Verify on Windows:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -c "import openpi_client; print(openpi_client.__file__)"
```

The printed path should point inside this repository.

## 2. Important Files

Core simulation scene:

```text
simulation/assets/xarm6/xarm6_pick_scene.xml
```

Camera calibration:

```text
simulation/config/camera_calibration.yaml
```

Reusable observation pipeline:

```text
policy_runtime/observation_builder.py
policy_runtime/image_preprocessing.py
simulation/observation/cameras.py
simulation/observation/state.py
simulation/observation/policy.py
```

Reusable action safety/control pipeline:

```text
policy_runtime/action_decoder.py
policy_runtime/safety.py
simulation/robot/control.py
simulation/robot/gripper_mapping.py
```

Single real-observation inference test:

```text
evaluation/sim/tools/test_remote_policy_mujoco.py
```

Dry-loop validation:

```text
evaluation/sim/tools/run_remote_policy_dry_loop.py
```

Safe closed-loop runner:

```text
evaluation/sim/legacy/run_remote_policy_closed_loop.py
```

Offline/unit tests:

```text
tests/evaluation_sim/test_remote_policy_pipeline.py
```

Maintained local validation:

```text
tests/simulation/
tests/diagnostics/
python -m diagnostics.simulation.camera.cli --help
```

Generated outputs:

```text
outputs/simulation/
```

This directory is ignored by Git.

## 3. Observation And Action Contracts

The policy input must be exactly:

```python
{
    "observation/image": base_image,
    "observation/wrist_image": wrist_image,
    "observation/state": state,
    "prompt": prompt,
}
```

Required shapes and dtypes:

```text
observation/image        uint8   (224, 224, 3)
observation/wrist_image  uint8   (224, 224, 3)
observation/state        float32 (7,)
prompt                   str
```

State order:

```text
[joint1, joint2, joint3, joint4, joint5, joint6, gripper_raw]
```

Remote policy output after local conversion:

```text
shape: (10, 7)
dtype: float32
```

Action order:

```text
[joint1, joint2, joint3, joint4, joint5, joint6, gripper_raw]
```

The first six dimensions are absolute joint targets after OpenPI server-side
postprocessing. The seventh dimension is the gripper raw target.

Known gripper raw range:

```text
50 to 845
```

In the current project mapping, high raw value means open:

```text
raw_closed = 50.0
raw_open = 845.0
sim_joint_min_rad = 0.005
sim_joint_max_rad = 0.85
```

Do not change the mapping based on visual guesswork. Verify against the existing
conversion functions and training-data convention first.

## 4. Camera Configuration

The source of truth is always:

```text
simulation/config/camera_calibration.yaml
```

Current calibrated parameters in this repository:

Base camera:

```yaml
position: [0.6998640343, -0.2034187962, 0.3691466327]
target: [0.3784729309, -0.171520319, 0.0715365659]
roll_deg: -12.5073468556
fovy_deg: 46.7280905796
```

Wrist camera:

```yaml
position: [0.06432955545751386, -0.0014285874072126926, 0.08233427275157185]
target: [-0.01648605838780822, 0.014125880549260589, 0.33412490948712087]
roll_deg: -97.06428858407915
fovy_deg: 95.0
```

Rendering convention:

- Render natively at `640 x 480`.
- Apply the existing OpenPI-compatible padding/resizing pipeline.
- Produce `uint8` RGB images with shape `(224, 224, 3)`.
- Do not render directly at `224 x 224`.
- Do not change calibration parameters to compensate for gripper appearance mismatch.
- Reuse the existing camera and preprocessing implementation.
- Do not create a second image convention.

## 5. Remote Policy Server On DeltaAI

DeltaAI configuration:

```text
User: mw89
Account: bfmk-dtai-gh
Partition: ghx4-interactive
Known validated compute node: gh031
Remote repository: ~/repos/openpi
Policy config: pi05_xarm
```

Checkpoint:

```text
/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/xarm_pi05_20260703_run1/30000
```

Start the policy server on the allocated GPU compute node:

```bash
cd ~/repos/openpi

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xarm \
  --policy.dir=/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/xarm_pi05_20260703_run1/30000
```

Successful startup includes:

```text
Loading model...
Finished restoring checkpoint...
Loaded norm stats...
server listening on 0.0.0.0:8000
```

JAX messages about unavailable ROCm or TPU backends are informational on an
NVIDIA GPU node and are not failures.

Verify from a login node:

```bash
ssh gh031 "ss -ltn | grep ':8000'"
```

Expected:

```text
LISTEN ... 0.0.0.0:8000
```

Direct WebSocket verification from the login node:

```bash
cd ~/repos/openpi

uv run python - <<'PY'
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(
    host="gh031",
    port=8000,
)

print("Direct WebSocket connection succeeded")
print("Server metadata:", client._server_metadata)
PY
```

Expected:

```text
Direct WebSocket connection succeeded
Server metadata: {}
```

## 6. SSH Tunnel From Local Machine

Use one local tunnel only. Do not leave multiple stale tunnels on the same port.

Recommended local port:

```text
18000
```

Windows PowerShell command:

```powershell
ssh -vvv `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=6 `
  -L 127.0.0.1:18000:gh031:8000 `
  mw89@gh-login01.delta.ncsa.illinois.edu
```

Keep this PowerShell window open.

A successful tunnel log includes:

```text
Connection to port 18000 forwarding to gh031 port 8000 requested.
channel ... open confirm
```

Check the local listener:

```powershell
netstat -ano | findstr "127.0.0.1:18000"
```

There should be one listener owned by `ssh.exe`.

Test the OpenPI client:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -c "from openpi_client import websocket_client_policy; c=websocket_client_policy.WebsocketClientPolicy(host='127.0.0.1', port=18000); print('WebSocket connected'); print('Server metadata:', c._server_metadata)"
```

Expected:

```text
WebSocket connected
Server metadata: {}
```

A plain TCP probe such as `Test-NetConnection` can cause a harmless WebSocket
handshake warning on the server because it does not send a valid WebSocket
upgrade request.

## 7. Local Preflight Validation

From the repository root:

```powershell
cd "D:\2026 summer project\embodied-ai-xarm"
```

Run syntax checks:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -m py_compile `
  ".\simulation\observation\policy.py" `
  ".\simulation\robot\control.py" `
  ".\evaluation\sim\tools\test_remote_policy_mujoco.py" `
  ".\evaluation\sim\tools\run_remote_policy_dry_loop.py" `
  ".\evaluation\sim\legacy\run_remote_policy_closed_loop.py"
```

Run offline tests:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -m pytest `
  tests/evaluation_sim/test_remote_policy_pipeline.py
```

Expected:

```text
13 passed
```

## 8. Single Real-Observation Inference

Run:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -m `
  evaluation.sim.tools.test_remote_policy_mujoco
```

Expected validation:

```text
base image shape: (224, 224, 3)
wrist image shape: (224, 224, 3)
state shape: (7,)
actions shape: (10, 7)
all finite: True
validation checks: PASS
```

Expected outputs:

```text
outputs/simulation/remote_policy_test/base_model_input.png
outputs/simulation/remote_policy_test/wrist_model_input.png
outputs/simulation/remote_policy_test/result.json
outputs/simulation/remote_policy_test/validation_report.txt
```

Open the model inputs on Windows:

```powershell
Invoke-Item ".\outputs\simulation\remote_policy_test\base_model_input.png"
Invoke-Item ".\outputs\simulation\remote_policy_test\wrist_model_input.png"
```

Check:

- object and arm are visible
- images are not vertically flipped
- colors are correct
- padding matches training preprocessing
- wrist view is not fully occluded
- images are not black or nearly uniform

This script must not execute actions.

## 9. Dry Loop

Run five iterations without applying policy actions:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -m `
  evaluation.sim.tools.run_remote_policy_dry_loop `
  --host 127.0.0.1 `
  --port 18000 `
  --prompt "pick up the object" `
  --iterations 5
```

The dry loop should:

- build a real observation
- request inference
- validate the action chunk
- compute a safe clamped target
- log the raw and safe targets
- never apply policy targets to `data.ctrl`

Stop on:

- NaN
- Inf
- wrong action shape
- invalid state
- joint-limit violation
- connection error

Review outputs under:

```text
outputs/simulation/remote_policy_dry_loop/
```

## 10. Safe Closed-Loop Simulation

Initial visible run, with 10 policy updates:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -m `
  evaluation.sim.legacy.run_remote_policy_closed_loop `
  --host 127.0.0.1 `
  --port 18000 `
  --prompt "pick up the object" `
  --max-policy-steps 10 `
  --execute-chunk-steps 1 `
  --max-joint-step 0.05 `
  --control-duration 0.02
```

Initial safety settings:

```text
execute-chunk-steps = 1
max-joint-step = 0.05 radians
control-duration = 0.02 seconds
gripper_raw clamp = [50, 845]
```

Do not increase `execute-chunk-steps` until single-step closed-loop behavior is
stable.

After single-step behavior is stable, values from `1` through the 10-step
policy horizon are supported. For example, `--execute-chunk-steps 5` executes
five targets, then uploads newly rendered base/wrist images and the resulting
joint state for the next inference. A value of `10` executes the complete
predicted chunk before re-observation.

Observe:

- movement direction
- oscillation
- sudden jumps
- gripper timing
- wrist camera visibility
- object contact
- target versus resulting qpos

Stop with `Ctrl+C`.

Headless smoke test:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -m `
  evaluation.sim.legacy.run_remote_policy_closed_loop `
  --headless `
  --max-policy-steps 3 `
  --execute-chunk-steps 1 `
  --max-joint-step 0.05 `
  --control-duration 0.02
```

This configuration has already passed on the validated setup.

## 11. Output Inspection

List recent step directories:

```powershell
Get-ChildItem ".\outputs\simulation\remote_policy_closed_loop" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 Name, LastWriteTime
```

Search for clipping or errors:

```powershell
Get-ChildItem ".\outputs\simulation\remote_policy_closed_loop" -Recurse -Filter diagnostics.json |
  Select-String -Pattern '"clipped": true|NaN|Infinity|error|exception'
```

Each policy step should preserve:

- base image
- wrist image
- current state
- raw action chunk
- safe target
- resulting qpos
- timing
- clipping diagnostics

No clipping was observed in:

- single inference test
- 5-step dry loop
- 3-step headless closed-loop smoke test
- 10-step visible closed-loop run on July 23, 2026

## 12. Safety Rules

Codex must not weaken these rules without explicit instruction.

- Reject non-finite states or actions.
- Require action shape `(10, 7)`.
- Execute only the first action in the initial implementation.
- Treat arm outputs as absolute joint targets.
- Clamp each arm target by MuJoCo joint range, actuator control range, and
  maximum per-update step of `0.05` radians.
- Clamp gripper raw to `[50, 845]`.
- Use the existing gripper raw-to-simulation mapping.
- Log unclamped and clamped targets.
- Stop on invalid state, invalid action, viewer close, `Ctrl+C`, or control exception.
- Keep the simulator paused while waiting for remote inference when possible.
- Do not change camera calibration.
- Do not edit raw training data.
- Do not change the checkpoint or normalization assets.

## 13. Codespace / Linux Notes

A GitHub Codespace is Linux, while the validated setup was run on Windows with a
local Conda environment.

Codex must not assume the Windows interpreter path exists in Codespaces.

In a Codespace:

```bash
uname -a
python --version
```

Create or reuse a Python 3.11 environment, then install the local OpenPI client:

```bash
python -m pip install -e third_party/openpi/packages/openpi-client
```

Install only missing simulation dependencies.

Use headless MuJoCo by default. Set a headless rendering backend if required:

```bash
export MUJOCO_GL=egl
```

Fallback when EGL is unavailable:

```bash
export MUJOCO_GL=osmesa
```

Do not expect an interactive MuJoCo window inside a Codespace unless a supported
graphical forwarding mechanism is configured.

Headless smoke command:

```bash
python -m evaluation.sim.legacy.run_remote_policy_closed_loop \
  --headless \
  --max-policy-steps 3 \
  --execute-chunk-steps 1 \
  --max-joint-step 0.05 \
  --control-duration 0.02
```

The SSH tunnel must be created inside the Codespace or forwarded into it. A
tunnel running only on the user's Windows host is not automatically visible
inside the Codespace.

Suggested Codespace tunnel:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -L 127.0.0.1:18000:gh031:8000 \
  mw89@gh-login01.delta.ncsa.illinois.edu
```

Interactive Duo authentication may make unattended Codespace operation
difficult. Codex must report this rather than attempting to bypass
authentication.

## 14. Codex Execution Procedure

When asked to run the project, Codex should follow this order.

1. Inspect:
   repository root, Python interpreter, required files, current Git status, and
   remote tunnel availability.
2. Preserve:
   do not overwrite `camera_calibration.yaml`, raw data, checkpoint
   configuration, or existing working scripts. Back up a file before substantial
   edits.
3. Test offline:
   `py_compile`, unit tests, local observation test, and camera rendering test.
4. Test connectivity:
   OpenPI client import, WebSocket connection to `127.0.0.1:18000`, and server
   metadata.
5. Run one real inference:
   `test_remote_policy_mujoco.py`, then inspect saved model input images and the
   validation report.
6. Run dry loop:
   `run_remote_policy_dry_loop.py --iterations 5`. Do not proceed if any check
   fails.
7. Run headless closed-loop smoke test:
   3 policy updates, 1 action per chunk, `0.05` rad max joint step, `0.02` s
   control duration.
8. Run visible short test only where a viewer is supported:
   10 policy updates.
9. Report:
   files changed, commands run, tests passed/failed, remote connection result,
   observation validation, action validation, clipping, closed-loop result,
   output directory, remaining risk, and exact next command.

## 15. Troubleshooting

`ModuleNotFoundError: openpi_client`

Use the correct interpreter or install the editable client:

```powershell
& "D:\miniconda\envs\mujoco-pi\python.exe" -m pip install -e ".\third_party\openpi\packages\openpi-client"
```

Linux/Codespace:

```bash
python -m pip install -e third_party/openpi/packages/openpi-client
```

`did not receive a valid HTTP response`

Check:

- one active tunnel only
- correct local port `18000`
- remote server still running
- compute node still allocated

Verify direct login-node connection to `gh031:8000`.

Multiple listeners on port `8000` or `18000`, Windows:

```powershell
netstat -ano | findstr "127.0.0.1:18000"
Get-Process -Id <PID>
```

Terminate stale SSH tunnels only after confirming their command lines.

Server logs `opening handshake failed`

A plain TCP probe may trigger this. A successful OpenPI client connection should
produce:

```text
connection open
Connection from (...) opened
```

Viewer unavailable in Codespace:

```text
--headless
```

Then inspect saved frames and diagnostics.

Simulation becomes unstable:

- reduce `max-joint-step`
- reduce `control-duration`
- reduce number of policy steps
- keep `execute-chunk-steps = 1`

Gripper direction appears reversed:

Do not change the mapping based on visual guesswork. Verify the existing
conversion functions and training-data convention first.

## 16. Recommended Next Command

On the validated Windows setup:

```powershell
cd "D:\2026 summer project\embodied-ai-xarm"

& "D:\miniconda\envs\mujoco-pi\python.exe" -m `
  evaluation.sim.legacy.run_remote_policy_closed_loop `
  --host 127.0.0.1 `
  --port 18000 `
  --prompt "pick up the object" `
  --max-policy-steps 10 `
  --execute-chunk-steps 1 `
  --max-joint-step 0.05 `
  --control-duration 0.02
```

On a Codespace or headless Linux environment:

```bash
export MUJOCO_GL=egl

python -m evaluation.sim.legacy.run_remote_policy_closed_loop \
  --host 127.0.0.1 \
  --port 18000 \
  --prompt "pick up the object" \
  --headless \
  --max-policy-steps 3 \
  --execute-chunk-steps 1 \
  --max-joint-step 0.05 \
  --control-duration 0.02
```
