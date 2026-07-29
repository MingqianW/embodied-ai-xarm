# MuJoCo demonstration data collection

The current multi-task collector writes simulation episodes directly in the
same raw directory/CSV/PNG format as the real xArm recorder. The older
red-block-only canonical pipeline is retained below for reproducibility.

## Collect the real-raw-compatible 200-episode set

The fixed collection plan is:

| Raw task folder | Total | Clean | Distractors |
|---|---:|---:|---:|
| `pick_up_the_red_pepper` | 50 | 50 | 0 |
| `pick_up_the_blue_block` | 25 | 15 | 10 |
| `pick_up_the_red_block` | 25 | 15 | 10 |
| `pick_up_the_smallest_block` | 25 | 15 | 10 |
| `pick up the largest block` | 25 | 15 | 10 |
| `place_the_red_pepper_in_the_ring` | 50 | 30 | 20 |

Each episode is written as:

```text
raw/<task>/episode_xxx/
  meta.json
  robot_log.csv
  gripper_events.csv
  realsense_0/*.png
  realsense_1/*.png
  realsense_2/*.png
```

`realsense_0` is the calibrated base camera, `realsense_1` is the wrist
camera, and `realsense_2` is the overview camera. All images are RGB uint8
640×480. `robot_log.csv` uses the exact real header and 10 Hz timestamps.
The existing real converter therefore derives each training action from the
next robot-log row exactly as it does for real recordings.

Collection (raw only; this command never converts or uploads):

```powershell
& 'D:\miniconda\envs\mujoco-pi\python.exe' `
  sim_mujoco\scripts\collect_real_raw_sim_data.py `
  --output-dir sim_mujoco\output\datasets\xarm_mujoco_200 `
  --object-xy-range 0.02 `
  --object-yaw-range-deg 10 `
  --joint-noise 0.005 `
  --max-attempts-per-episode 10 `
  --headless
```

Use the same arguments plus `--resume` after an interruption. Completed
episodes are never overwritten.

Strict raw validation:

```powershell
& 'D:\miniconda\envs\mujoco-pi\python.exe' `
  sim_mujoco\scripts\validate_real_raw_sim_data.py `
  --raw-root sim_mujoco\output\datasets\xarm_mujoco_200\raw `
  --output sim_mujoco\output\datasets\xarm_mujoco_200\raw_validation_report.json
```

The output produced on 2026-07-24 contains 200 successful episodes, 12,955
raw rows, 12,755 next-row training samples, and 38,865 PNG files. No
conversion or upload is part of this collection workflow.

## Legacy red-block canonical pipeline

The legacy pipeline reuses the same canonical writer as the real xArm
converter:

- real adapter: `fine_tune/convert_xarm_raw_to_lerobot.py`
- shared writer: `fine_tune/xarm_lerobot_writer.py`
- simulation adapter: `sim_mujoco/data_collection/lerobot_adapter.py`

No command in this document modifies the real recordings, the real LeRobot
dataset, an existing checkpoint, normalization statistics, camera
calibration, or collision geometry.

## Contract

Training state and action are both float32 vectors of shape `(7,)`:

```text
[joint1_rad, joint2_rad, joint3_rad,
 joint4_rad, joint5_rad, joint6_rad,
 gripper_raw]
```

`state_t` is captured immediately before execution. `action_t` is an absolute
joint/gripper target for the next 0.1 s interval. Data is recorded at 10 Hz,
which is 50 MuJoCo physics steps at the active 0.002 s timestep.

Stored LeRobot images are RGB uint8 `(480, 640, 3)` under keys `image` and
`wrist_image`. The existing OpenPI transform resizes them to 224×224. The
stored actions remain 7D; OpenPI converts joints 0–5 to deltas, leaves the
gripper absolute, and pads to 32 dimensions in the model transform.

The exact prompt is:

```text
pick up the red block
```

Use the configured interpreter in PowerShell:

```powershell
$python = 'D:\miniconda\envs\mujoco-pi\python.exe'
```

## 1. Test the scripted oracle

Fixed gate (must be 10/10):

```powershell
& $python sim_mujoco\scripts\test_scripted_oracle.py `
  --task red_block `
  --episodes 10 `
  --seed-start 0 `
  --object-xy-range 0 `
  --object-yaw-range-deg 0 `
  --joint-noise 0 `
  --record-video `
  --output-dir sim_mujoco\output\oracle_gate_fixed
```

Small-randomization gate (must be at least 18/20):

```powershell
& $python sim_mujoco\scripts\test_scripted_oracle.py `
  --task red_block `
  --episodes 20 `
  --seed-start 1000 `
  --object-xy-range 0.01 `
  --object-yaw-range-deg 5 `
  --joint-noise 0.005 `
  --output-dir sim_mujoco\output\oracle_gate_randomized
```

## 2. Collect fixed episodes (Stage A)

```powershell
& $python sim_mujoco\scripts\collect_oracle_data.py `
  --episodes 10 `
  --seed-start 0 `
  --task red_block `
  --action-hz 10 `
  --object-xy-range 0 `
  --object-yaw-range-deg 0 `
  --joint-noise 0 `
  --save-only-success `
  --record-video `
  --video-every 1 `
  --output-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_stage_a `
  --max-attempts 10 `
  --headless
```

## 3. Collect randomized episodes

Stage B performs exactly 30 attempts. `--allow-partial` means a failed oracle
attempt does not invalidate the run; failed attempts remain outside
`episodes/`.

```powershell
& $python sim_mujoco\scripts\collect_oracle_data.py `
  --episodes 30 `
  --seed-start 10000 `
  --task red_block `
  --action-hz 10 `
  --object-xy-range 0.01 `
  --object-yaw-range-deg 5 `
  --joint-noise 0.005 `
  --save-only-success `
  --record-video `
  --video-every 1 `
  --output-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_stage_b `
  --max-attempts 30 `
  --allow-partial `
  --headless
```

Stage C continues until 30 successful trajectories exist and records every
fifth successful episode:

```powershell
& $python sim_mujoco\scripts\collect_oracle_data.py `
  --episodes 30 `
  --seed-start 20000 `
  --task red_block `
  --action-hz 10 `
  --object-xy-range 0.03 `
  --object-yaw-range-deg 15 `
  --joint-noise 0.01 `
  --save-only-success `
  --record-video `
  --video-every 5 `
  --output-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_raw `
  --max-attempts 90 `
  --headless
```

## 4. Resume collection

Repeat every original argument exactly and add `--resume`. The collector
checks the saved configuration hash and rejects changed FPS, task,
randomization, seed, video cadence, target count, and maximum attempts.

For Stage C:

```powershell
& $python sim_mujoco\scripts\collect_oracle_data.py `
  --episodes 30 `
  --seed-start 20000 `
  --task red_block `
  --action-hz 10 `
  --object-xy-range 0.03 `
  --object-yaw-range-deg 15 `
  --joint-noise 0.01 `
  --save-only-success `
  --record-video `
  --video-every 5 `
  --output-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_raw `
  --max-attempts 90 `
  --headless `
  --resume
```

Completed episode directories are never overwritten. Stale `.staging`
directories intentionally stop resume so an interrupted attempt can be
inspected rather than silently discarded.

## 5. Inspect an episode

Metadata and aligned numeric arrays:

```powershell
Get-Content -Raw sim_mujoco\output\datasets\xarm_mujoco_red_block_raw\episodes\episode_000000\metadata.json

& $python -c "import numpy as np; p=np.load(r'sim_mujoco\output\datasets\xarm_mujoco_red_block_raw\episodes\episode_000000\observations.npz'); print({k:(p[k].shape,str(p[k].dtype)) for k in p.files})"
```

Open a recorded video:

```powershell
Start-Process sim_mujoco\output\datasets\xarm_mujoco_red_block_raw\episodes\episode_000000\combined.mp4
```

The GUI command is optional; all validation commands are headless.

## 6. Install the pinned canonical LeRobot/OpenPI environment

OpenPI pins LeRobot commit
`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` (`codebase_version=v2.1`).
Installing it changes a Python environment and downloads executable packages,
so do this only after explicit approval.

For canonical conversion only:

```powershell
uv pip install `
  --python 'D:\miniconda\envs\mujoco-pi\python.exe' `
  'lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
```

The full upstream OpenPI dependency set declares CUDA JAX and is primarily
Linux-oriented. On Windows, use a separately approved CPU-compatible
environment or the existing Colab workflow for the OpenPI batch gate; do not
silently mutate the working MuJoCo environment.

## 7. Convert to canonical LeRobot

Validate every raw image and aligned action first without writing:

```powershell
& $python sim_mujoco\scripts\convert_mujoco_to_lerobot.py `
  --input-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_raw `
  --output-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_lerobot `
  --repo-id MingqianW/xarm_mujoco_red_block_v1 `
  --dataset-name xarm_mujoco_red_block_v1 `
  --validate-only `
  --num-workers 4
```

Canonical conversion:

```powershell
& $python sim_mujoco\scripts\convert_mujoco_to_lerobot.py `
  --input-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_raw `
  --output-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_lerobot `
  --repo-id MingqianW/xarm_mujoco_red_block_v1 `
  --dataset-name xarm_mujoco_red_block_v1 `
  --copy-videos `
  --num-workers 4
```

Resume after adding new successful raw episodes:

```powershell
& $python sim_mujoco\scripts\convert_mujoco_to_lerobot.py `
  --input-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_raw `
  --output-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_lerobot `
  --repo-id MingqianW/xarm_mujoco_red_block_v1 `
  --dataset-name xarm_mujoco_red_block_v1 `
  --copy-videos `
  --num-workers 4 `
  --resume
```

`--overwrite` and `--resume` are mutually exclusive.

## 8. Validate the canonical dataset

First generate the real/sim distribution report:

```powershell
& $python sim_mujoco\scripts\compare_real_sim_datasets.py `
  --real-raw-root fine_tune\data\xarm_pi05_data\raw `
  --sim-raw-root sim_mujoco\output\datasets\xarm_mujoco_red_block_raw `
  --output-dir sim_mujoco\output\real_sim_comparison `
  --overwrite
```

Full validation, including one OpenPI batch:

```powershell
& $python sim_mujoco\scripts\validate_mujoco_lerobot_dataset.py `
  --dataset-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_lerobot `
  --raw-input-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_raw `
  --repo-id MingqianW/xarm_mujoco_red_block_v1 `
  --comparison-csv sim_mujoco\output\real_sim_comparison\distribution_comparison.csv `
  --output-dir sim_mujoco\output\dataset_validation `
  --python 'PATH_TO_APPROVED_OPENPI_PYTHON'
```

For local schema debugging only, `--skip-openpi-batch` leaves that mandatory
gate as a warning; it is not evidence of full completion.

## 9. Prepare Hugging Face files

```powershell
& $python sim_mujoco\scripts\prepare_mujoco_hf_ready.py `
  --dataset-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_lerobot `
  --raw-input-dir sim_mujoco\output\datasets\xarm_mujoco_red_block_raw `
  --output-dir sim_mujoco\output\hf_ready\xarm_mujoco_red_block_v1 `
  --repo-id MingqianW/xarm_mujoco_red_block_v1
```

This writes `README.md`, `DATASET_CARD.md`, `UPLOAD_PLAN.md`,
`MIXED_REAL_SIM_PLAN.md`, and a SHA256/size `MANIFEST.json`. It does not
contact Hugging Face.

## 10. Dry-run upload

Authenticate without printing a token:

```powershell
hf auth login
```

For older clients:

```powershell
huggingface-cli login
```

Then run the default no-upload inspection:

```powershell
& $python sim_mujoco\scripts\upload_mujoco_dataset_to_hf.py `
  --local-dir sim_mujoco\output\hf_ready\xarm_mujoco_red_block_v1 `
  --repo-id MingqianW/xarm_mujoco_red_block_v1 `
  --private `
  --dry-run
```

The dry run verifies every manifest hash, lists files and ignored files,
reports byte size and remote-repository status, and performs no writes.

## 11. Final confirmed upload

Do **not** run this command until the user has reviewed the validation report,
domain-gap warnings, dataset card, and dry-run output and then explicitly
confirmed the exact repository ID.

```powershell
& $python sim_mujoco\scripts\upload_mujoco_dataset_to_hf.py `
  --local-dir sim_mujoco\output\hf_ready\xarm_mujoco_red_block_v1 `
  --repo-id MingqianW/xarm_mujoco_red_block_v1 `
  --private `
  --commit-message 'Add MuJoCo red-block scripted-oracle dataset v1' `
  --upload `
  --yes
```

Actual upload requires both `--upload` and `--yes`, refuses `--dry-run`,
refuses an existing repository with a different dataset identity, supports
resumable `upload_large_folder`, and never deletes remote files.
