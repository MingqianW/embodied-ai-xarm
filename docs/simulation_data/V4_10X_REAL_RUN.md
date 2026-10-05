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

## FarmShare collection, 2026-10-05

The pinned real dataset revision is
`1bbb721baef1e152515a33a50fa4552686877ce1`. The FarmShare run accepted
and converted 1,980 simulation episodes, with exactly the task counts above
and no distractor episodes. The final audit reported `READY_FOR_TRAINING`,
192,169 frames, canonical prompts passing, and all raw and converted images
decodable. The source was recorded on branch
`deployment/farmshare-sim-evaluation` at GitHub commit
`dcb90357fcd19e145b6596a5b56e3c02c29e6d32`.

Runtime data and evidence are outside the source tree:

```text
/scratch/users/mw27/cotraining-feasibility-20261004/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v4_10x_real
/scratch/users/mw27/cotraining-feasibility-20261004/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v4_10x_real
/scratch/users/mw27/cotraining-feasibility-20261004/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real/DATASET_AUDIT.md
/home/users/mw27/xarm-work/cotraining-feasibility-20261004/LARGE_GENERATION_HANDOFF.md
```

The full simulation pool has 1,980 episodes. Even episode indices select 990
training episodes and odd indices hold out 990; each task count is halved. The
earlier feasibility run used 158 real training episodes. The approved full
training configuration instead selects all 198 pinned real episodes, without
a real held-out subset. The training domain sampler is configured separately
with `P(real)=0.1` and `P(sim)=0.9`.

The earlier 158-real/990-sim train-only normalization was validated in Slurm
job `1782674` against its selected datasets and manifest. It does not match
the approved 198-real/990-sim full-training pool; the latter requires a fresh
normalization asset before use. Each selected physical frame contributes once;
the domain sampling probabilities do not reweight the statistics. The earlier
asset's evidence is retained at:

```text
/home/users/mw27/xarm-work/cotraining-feasibility-20261004/large-normalization-evidence.json
/scratch/users/mw27/cotraining-feasibility-20261004/openpi_assets/pi05_xarm_real_sim_alpha09/xarm_pi05_real_sim_alpha09_trainonly_v4_10x_v1/norm_stats.json
/scratch/users/mw27/cotraining-feasibility-20261004/openpi_assets/pi05_xarm_real_sim_alpha09/xarm_pi05_real_sim_alpha09_trainonly_v4_10x_v1/mixed_normalization_manifest.json
```

The raw `realsense_0` and `realsense_1` streams become LeRobot `image` (base)
and `wrist_image` (wrist), respectively. `realsense_2` is an overview video
for review and is not used as a model input. The Place smoke episode has both
model-view videos under the smoke root at
`accepted/place_red_pepper_in_ring/episode_000/realsense_{0,1}.mp4`.
Full model training has not been launched.
