# Training Data Tracker

Use this file to track the dataset that is actually used for OpenPI fine-tuning. Keep it in sync when raw episodes are added, renamed, quality-checked, converted to LeRobot, or pushed to Hugging Face.

## Current Dataset

| Field | Value |
|---|---|
| Robot | xArm 6 |
| Control rate | 10 Hz |
| Cameras | `image` = base scene camera, `wrist_image` = wrist camera |
| State format | `[j1_rad, j2_rad, j3_rad, j4_rad, j5_rad, j6_rad, gripper_mm]` |

## OpenPI Base Default Training Settings

These are the default values from OpenPI `TrainConfig` when a field is not explicitly overwritten in a named config.

| Field | OpenPI Default |
|---|---|
| Optimizer | `optimizer.AdamW()` |
| EMA decay | `0.99` |

### OpenPI Default Learning-rate Schedule

When `lr_schedule=_optimizer.CosineDecaySchedule()` is used without arguments, record it as:

| Field | Value |
|---|---:|
| Warmup steps | `1,000` |
| Peak LR | `2.5e-5` |
| Decay steps | `30,000` |
| Decay LR | `2.5e-6` |

This means the learning rate warms up from a tiny initial value to `2.5e-5`, then cosine-decays toward `2.5e-6`.

## Dataset Versions

Add one row every time the training dataset changes. Use the manifest and `meta/info.json` from the converted dataset as the source of truth.

| Date | Dataset Version | Episodes  | Tasks Included | Repo ID / Path  | Notes |
|---|---|---:|---|---|------|
| 2026-06-26 | `v1` | 50  | `pick_up_the_red_pepper` | `local/xarm_pi05_data`  | First version with purely clean data of one task |
| 2026-07-03 | `v2` | 150 | `pick up the largest block` (25), `pick_up_the_blue_block` (25), `pick_up_the_red_block` (25), `pick_up_the_red_pepper` (50), `pick_up_the_smallest_block` (25) | `local/xarm_pi05_data` | Second version with mixed data of different pick-up tasks |

## Hyperparameter Settings

Record the OpenPI training settings used with each dataset version.

| Date | Dataset Version | Config Name | Base Checkpoint | Batch Size | Train Steps | Save Interval | Learning Rate | EMA | Fine-tuning Mode | Notes |
|---|---|---|---|---:|---:|---:|---|---|---|---|
| 2026-06-26 | `v1` | `pi05_xarm_full_finetune` | `gs://openpi-assets/checkpoints/pi05_base/params` | 16 | 30,000 | 5,000 | OpenPI default: cosine schedule, warmup `1,000`, peak `2.5e-5`, decay steps `30,000`, decay LR `2.5e-6` | OpenPI default: `0.99` | Full fine-tuning from pi0.5 base | Main config for the 50-episode one-task dataset |
| 2026-07-03 | `v2` | `pi05_xarm` | `/content/drive/MyDrive/embodied_ai_xarm/openpi_checkpoints/pi05_xarm_full_finetune/pi05_xarm_full_finetune/25000/params` | 16 | 20,001 | 5,000 | Cosine schedule, warmup `500`, peak `1e-5`, decay steps `20,000`, decay LR `1e-6` | `0.999` | Continued full fine-tuning from v1 checkpoint |  |

## Data Shuffling Note

During OpenPI training, data shuffling is enabled.So shuffling changes which samples appear together in a mini-batch.
