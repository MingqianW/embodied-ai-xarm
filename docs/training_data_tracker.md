# Training Data Tracker

Use this file to track the dataset that is actually used for OpenPI fine-tuning. Keep it in sync when raw episodes are added, renamed, quality-checked, converted to LeRobot, or pushed to Hugging Face.

## Current Dataset

| Field | Value |
|---|---|
| Robot | xArm 6 |
| Control rate | 10 Hz |
| Cameras | `image` = base scene camera, `wrist_image` = wrist camera |
| State format | `[j1_rad, j2_rad, j3_rad, j4_rad, j5_rad, j6_rad, gripper_mm]` |

## Dataset Versions

Add one row every time the training dataset changes. Use the manifest and `meta/info.json` from the converted dataset as the source of truth.

| Date | Dataset Version | Episodes  | Tsks Included | Repo ID / Path  | Notes |
|---|---|---:|---|---|------|
| 2026-06-26 | `v1` | 50  |  red pepper | `local/xarm_pi05_data`  | First version with purely clean data of one task |
