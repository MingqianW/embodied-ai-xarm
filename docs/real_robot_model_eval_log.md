# Real Robot Model Evaluation Log

Use this file to record tests of trained OpenPI xArm policies on the physical robot.

## Evaluation Setup

| Field | Value |
|---|---|
| Date |  |
| Operator |  |
| Robot | xArm 6 |
| Robot IP | 192.168.1.209 |
| Camera setup | base camera: 0, wrist camera: 1 |
| OpenPI repo path | `/home/xingyu/pi_0.5/openpi` |
| Robot repo path | `/home/xingyu/robot/xarm-calibrate-hanyang` |
| Run script | `run_pi_xarm.py` |
| Config name | `pi05_xarm_full_finetune` |
| Checkpoint path |  |
| Training step |  |
| Dataset version / notes |  |

## Runtime Parameters

| Parameter | Value |
|---|---:|
| Prompt |  |
| Rollout cycles |  |
| Actions executed per inference |  |
| Max joint delta per command | 0.04 rad |
| Joint speed | 0.25 |
| Joint acceleration | 1.0 |
| Gripper min / max | 167 / 845 mm |
| Gripper speed | 1500 |
| Control delay | 0.15 s |

## Safety Checklist

- [ ] Emergency stop is reachable.
- [ ] Workspace is clear.
- [ ] Robot starts from a known safe pose.
- [ ] Gripper opens and closes correctly before policy rollout.
- [ ] Base and wrist camera images are correctly assigned.
- [ ] Inference-only test was inspected before executing actions.
- [ ] First action joint deltas are within expected range.

## Trial Log

| Trial | Task / Prompt | Initial Scene | Checkpoint | Success | Failure Mode | Notes |
|---:|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |

## Detailed Trial Notes

### Trial 1

- Prompt:
- Initial object positions:
- First action:
- Observed behavior:
- Result:
- Failure reason, if any:
- Video / image link:

### Trial 2

- Prompt:
- Initial object positions:
- First action:
- Observed behavior:
- Result:
- Failure reason, if any:
- Video / image link:

### Trial 3

- Prompt:
- Initial object positions:
- First action:
- Observed behavior:
- Result:
- Failure reason, if any:
- Video / image link:

## Aggregate Results

| Task | Trials | Successes | Success Rate | Common Failure Modes |
|---|---:|---:|---:|---|
| pick_up_dark_blue_block |  |  |  |  |
| pick_up_light_blue_block |  |  |  |  |
| pick_up_red_block |  |  |  |  |
| place_red_on_blue |  |  |  |  |

## Failure Mode Tags

Use these tags consistently in the trial table.

- `no_reach`: arm does not move close enough to object
- `bad_camera_mapping`: base/wrist camera assignment appears swapped or wrong
- `bad_depth_or_occlusion`: visual input is blocked or misleading
- `wrong_object`: robot reaches for the wrong object
- `missed_grasp`: gripper closes but does not capture object
- `weak_grasp`: object slips after grasp
- `bad_gripper`: gripper command is too open, too closed, or delayed
- `unsafe_motion`: action is too abrupt or near workspace limits
- `task_confusion`: behavior does not match prompt
- `recovery_failure`: robot cannot recover after a failed grasp

## Follow-Up Actions

- [ ] Adjust runtime safety limits:
- [ ] Collect additional demonstrations:
- [ ] Remove or relabel problematic demonstrations:
- [ ] Recompute norm stats:
- [ ] Train next checkpoint:
- [ ] Update `run_pi_xarm.py`:

