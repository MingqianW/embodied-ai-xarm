# Data Collection Tracker

Use this file to track the current multi-object pick-up dataset and compare the base pi0.5 policy with the xArm fine-tuned policy.

## Dataset Summary

| Field | Value |
|---|---|
| Robot | xArm 6 |
| Task type | Pick up target object |
| Camera setup | 1st base scene camera + wrist camera |
| State/action format | `[j1_rad, j2_rad, j3_rad, j4_rad, j5_rad, j6_rad, gripper_mm]` |
| Action convention | next-frame absolute state |
| Control rate | 10 Hz |


## Data collection Progress

| Task  | Target Episodes | Done | Scene Mix | Notes |
|---|---:|---:|---|---|
| `pick up the red pepper` | 50 | 50 | clean object-only scenes | pure trajectories |
| `pick up the light blue block` | 25 | 25 | 15 clean, 10 with distractors | contains other objects/blocks |
| `pick up the red block` | 25 | 25 | 15 clean, 10 with distractors | contains other objects/blocks |
| `pick up the smallest block` | 25 | 25 | 15 clean, 10 with distractors | size-based prompt |
| `pick up the largest block` | 25 | 25 | 15 clean, 10 with distractors | size-based prompt |
| `pick up the blackboard eraser` | 50 | 0 | 30 clean, 20 with distractors | non-block object |
| `place red pepper on color paper` | 50 | 50 | 30 clean, 20 with distractors | size-based prompt |
| **Total** | **250** | **200** |  |  |

