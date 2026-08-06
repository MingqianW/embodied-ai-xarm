# v4 strict 10x-real simulation run

Dataset version:

```text
xarm_mujoco_clean_multitask_stable_v4_10x_real
```

Real dataset:

```text
/work/nvme/bfmk/mw89/datasets/lerobot/local/xarm_pi05_20260703
```

The real dataset contains 198 accepted episodes. The simulation target
is exactly ten times the real count for each canonical task.

| Task | Real | Simulation |
|---|---:|---:|
| red_pepper | 50 | 500 |
| blue_block | 24 | 240 |
| red_block | 25 | 250 |
| smallest_block | 24 | 240 |
| largest_block | 25 | 250 |
| place_red_pepper_in_ring | 50 | 500 |
| **Total** | **198** | **1980** |

All episodes use clean scenes with zero distractors. Camera calibration,
oracle control, Pick stability validation, and Place initialization and
release validation inherit the validated v3 pipeline.

Outputs:

```text
/work/nvme/bfmk/mw89/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v4_10x_real
/work/nvme/bfmk/mw89/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v4_10x_real
/work/nvme/bfmk/mw89/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_stable_v4_10x_real
/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real
```
