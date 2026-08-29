# OpenPI π0.5 xArm Real+MuJoCo Continuation Audit

Last updated: `2026-07-29T13:35:33-05:00`
Audit host: `gh-login03.delta.ncsa.illinois.edu` (login node, no Slurm allocation)
Mode: read-only readiness audit; training was not started

## Executive conclusion

The step-directory `30000` checkpoint has the components needed for a true
resume: raw training parameters, Adam optimizer state, EMA parameters, a scalar
step, and normalization assets. A weight loader pointed only at `params` would
instead be a weight-only warm start.

The run is **not ready to launch**. The requested MuJoCo root
`/work/nvme/bfmk/mw89/mujoco_datasets` does not exist, so there is no actual
simulation dataset to validate or mix. In addition, the current LeRobot loader
accepts one repository and has no weighted multi-dataset sampler, the stored
step must be restored on a compute node to confirm an off-by-one convention,
and the current checkpoint code couples the resume source with the output run
directory.

No dataset, checkpoint, training configuration, remote repository, or
authentication state was modified. No normalization computation, upload,
Slurm submission, or training was performed.

## OpenPI repository

- Path: `/u/mw89/repos/openpi`
- Branch: `main`
- Commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Subject: `update output objects to support batching (#975)`
- Remote: `https://github.com/Physical-Intelligence/openpi.git` (no embedded
  credentials)
- Root `AGENTS.md`: absent
- Root `PROJECT_STATUS.md`: absent
- Working tree: modified
  - Modified: `src/openpi/training/config.py`
  - Untracked: `checkpoints`, `slurm/`

The local config change adds `LeRobotXArmDataConfig` and π0.5 xArm
configurations. It is pre-existing user work and was not changed.

### Active `pi05_xarm` configuration

| Setting | Current value |
|---|---|
| Model | π0.5, `action_dim=32`, horizon `10`, continuous state |
| Data | `LeRobotXArmDataConfig(repo_id="local/xarm_pi05_20260703")` |
| Prompt | Taken from dataset task |
| Initialization | Base π0.5 weight loader |
| Batch size | `16` |
| Seed / workers | `42` / `2` |
| Optimizer | AdamW, β₁ `0.9`, β₂ `0.95`, ε `1e-8`, weight decay `1e-10`, clip `1.0` |
| EMA | `0.999` |
| LR | 1,000-step warmup to `2.5e-5`, cosine decay over 30,000 steps to `2.5e-6` |
| Train steps | `30001` |
| Save / log | `10000` / `100` |
| Keep period | `5000` |
| Resume | `false` |
| W&B | enabled |

The xArm-specific LR override is commented out, so the general `TrainConfig`
schedule above is active. The exact entrypoint is:

```bash
uv run scripts/train.py pi05_xarm --exp-name <run-name>
```

Tyro supplies the CLI. `scripts/train.py` is the training entrypoint.

The untracked historical Slurm script uses an overwrite option and routes
logs, dataset/cache state, and W&B state through HOME. It is not suitable for
reuse under current policy.

## Python and JAX environment

The existing environment is `/u/mw89/repos/openpi/.venv`, CPython 3.11 on
`aarch64`. Package metadata shows:

| Package | Version |
|---|---:|
| JAX / jaxlib | `0.5.3` / `0.5.3` |
| Flax | `0.10.2` |
| Orbax Checkpoint | `0.11.13` |
| Optax | `0.2.4` |
| NumPy | `1.26.4` |
| LeRobot | `0.1.0` |
| datasets | `3.6.0` |
| PyTorch | `2.7.1` |
| W&B | `0.19.11` |
| Tyro | `0.9.22` |
| uv CLI | `0.11.28` |

JAX was not imported on the login node. A compute-node smoke restore is still
required before training.

`/u/mw89/scripts/openpi_env.sh` currently sends datasets, checkpoints, logs,
Hugging Face, uv, Torch, and W&B state to HOME. Future execution should instead
set explicit cache and output roots under `/work/nvme/bfmk/mw89`; the helper
was not edited.

## Checkpoint audit

Source:

```text
/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/
  xarm_pi05_20260703_run1/30000
```

The run root contains retained manager steps `10000`, `20000`, and `30000`,
plus `wandb_id.txt`. The W&B identifier was not read.

### Complete logical structure

```text
30000/
├── _CHECKPOINT_METADATA
├── assets/
│   └── local/xarm_pi05_20260703/norm_stats.json
├── params/
│   ├── _METADATA
│   ├── _sharding
│   ├── array_metadatas/process_0
│   ├── d/
│   ├── manifest.ocdbt
│   └── ocdbt.process_0/{manifest.ocdbt,d/}
└── train_state/
    ├── _METADATA
    ├── _sharding
    ├── array_metadatas/process_0
    ├── d/
    ├── manifest.ocdbt
    └── ocdbt.process_0/{manifest.ocdbt,d/}
```

Orbax metadata identifies:

- `params`: 51 leaves. With EMA enabled, this separate item is the EMA
  inference parameter tree.
- `train_state.params`: 51 raw training-parameter leaves.
- `train_state.opt_state`: 106 leaves, including Adam `mu`, `nu`, and count.
- `train_state.step`: one scalar leaf.
- `model_def` and `ema_params` in `train_state`: intentionally serialized as
  `None`; restore reconstructs the model definition, optimizer transform, and
  EMA convention from the current configuration and merges the separate
  `params` item back as EMA parameters.
- Assets: checkpoint normalization JSON is present and matches the repository
  xArm normalization asset.
- Standalone training config: absent.

### True resume versus warm start

True resume is possible if the same model tree, AdamW transform, and EMA
convention are reconstructed. It restores raw parameters, optimizer moments
and counter, EMA parameters, and global step. It does not restore a dataloader
iterator or the exact shuffled-sample position.

`CheckpointWeightLoader` pointed at the checkpoint `params` item restores only
EMA/inference weights. It does not restore optimizer state, separate raw
training parameters, EMA lineage, or global step and is therefore a
**weight-only warm start**.

### Step-number issue

The loop updates the state before saving it under the loop-index directory.
Consequently, directory `30000` is expected to contain
`train_state.step=30001`. That scalar was not loaded on the login node.

If a compute-node restore confirms `30001`, exactly 20,000 more updates require:

```text
num_train_steps = 50001
executed loop indices = 30001..50000
final manager directory = 50000
final serialized state.step = 50001
```

Setting `num_train_steps=50000` would instead execute 19,999 more updates and
finish with manager directory `49999`.

Another blocker is that current code uses one checkpoint-manager directory for
both restore and output. A clean new run directory requires a later approved
change that separates the resume source from the output manager. An approved
checkpoint copy is an alternative but is less desirable. No such change or
copy was made.

See `CHECKPOINT_RESUME_AUDIT.json` for the machine-readable result.

## Real dataset

The expected external identifier `MingqianW/xarm_pi05_20260703` was not needed
or contacted. The dataset actually named by the config and checkpoint assets
is:

```text
repo_id: local/xarm_pi05_20260703
path: /work/nvme/bfmk/mw89/datasets/lerobot/local/xarm_pi05_20260703
```

Empty HOME placeholder directories exist for the expected external identifier;
they are not the training source.

| Property | Verified value |
|---|---|
| LeRobot codebase version | `v2.1` |
| Robot | `xarm6` |
| Episodes | `198` |
| Frames | `22618` |
| FPS | `10` |
| Tasks | `6` |
| Split | `train: 0:198` |
| Videos | `0`; images are embedded in Parquet |
| Episode files | `198`; metadata and Parquet row totals agree |

Task distribution:

| Task | Episodes | Frames |
|---|---:|---:|
| pick up the largest block | 25 | 3,201 |
| pick up the blue block | 24 | 3,275 |
| pick up the red block | 25 | 3,394 |
| pick up the red pepper | 50 | 4,610 |
| pick up the smallest block | 24 | 3,506 |
| place the red pepper in the ring | 50 | 4,632 |

Schema:

- `image`, `wrist_image`: decoded `uint8` RGB, `[480, 640, 3]`; stored as
  Parquet structs containing bytes and a path.
- `state`: `float32[7]`.
- `actions`: `float32[7]`.
- `timestamp`: `float32[1]`.
- `frame_index`, `episode_index`, `index`, `task_index`: `int64[1]`.
- State and action order:
  `[joint_1_rad, joint_2_rad, joint_3_rad, joint_4_rad, joint_5_rad,
  joint_6_rad, gripper_mm]`.
- Raw actions are the next-frame absolute joint/gripper target; the final raw
  row is omitted. OpenPI converts joint dimensions 0–5 to delta actions
  relative to state and leaves gripper absolute.
- `episode_index` is contiguous `0..197`, `frame_index` resets within each
  episode, and `index` is global.

The observed gripper extent is approximately 211–843 mm; checkpoint q01/q99 is
217.952–841.9888 mm.

A mapped `v30` copy exists elsewhere under the dataset root but is not selected
by the config or checkpoint. Its historical episode-index metadata needs
separate validation and it should not be substituted automatically.

## MuJoCo dataset discovery

The requested root does not exist:

```text
/work/nvme/bfmk/mw89/mujoco_datasets
```

A bounded, maximum-depth-four inspection within the approved work root found no
MuJoCo-named or simulation LeRobot candidate. Candidate count is therefore
zero. Episode count, frame count, FPS, prompt, schema, validation status,
normalization, failed-oracle exclusion, and OpenPI loadability are all
unverified.

The repository converter is useful only as an intended contract, not evidence
that a dataset satisfies it. It defaults to
`MingqianW/xarm_mujoco_red_block_v1`, does not push by default, intends LeRobot
v2.1 at 10 Hz, uses prompt `pick up the red block`, and shares the xArm writer
for two 480×640 RGB views plus 7-D state/action. It is designed to accept only
manifest-approved completed successes and reject non-success episodes.
Whether failed oracle episodes were actually excluded cannot be established
without the converted dataset and its source manifest/validation report.

## Exact schema comparison

The actual real schema is verified. The converter intends the same image keys,
dimensions, state/action order, absolute raw action convention, 10 Hz cadence,
prompt, and v2.1 indexing convention. **No real-versus-simulation match is
verified**, because no actual simulation dataset was found.

The checks that remain mandatory are:

1. Decode representative images and compare dtype, shape, RGB ordering, and
   camera-key assignment.
2. Confirm joint units/order, gripper units/range/direction, and action temporal
   alignment.
3. Check metadata totals, split bounds, episode continuity, frame-index reset,
   and global index uniqueness.
4. Prove successful-only selection using the source manifest and validator
   result.
5. Compare transformed state/action distributions and out-of-range rates
   against the checkpoint bounds.
6. Perform one LeRobot metadata load and then an OpenPI batch validation on a
   Slurm compute node.

The field-by-field machine-readable comparison is in
`DATASET_SCHEMA_COMPARISON.json`.

## Normalization comparison and recommendation

The checkpoint asset and the OpenPI xArm repository asset agree. π0.5 uses
q01/q99 normalization. The action statistics are measured after OpenPI turns
the first six joint targets into deltas while retaining the absolute gripper.

| Array | q01 | q99 |
|---|---|---|
| state | `[-0.584843, 0.042942, -1.313352, 0.017081, 0.187208, -0.928361, 217.952]` | `[-0.076990, 0.682192, -0.747379, 0.019338, 0.965850, 0.240858, 841.9888]` |
| transformed actions | `[-0.051479, -0.229062, -0.074855, -0.000132, -0.235415, -0.153322, 217.952]` | `[0.086753, 0.175145, 0.053563, 0.000127, 0.323579, 0.117236, 841.9888]` |

There are no MuJoCo or combined statistics to compare. No statistic computation
was started.

The safest continuation strategy is to keep the checkpoint's real-data
q01/q99 unchanged. Changing normalization while restoring optimizer and EMA
state would alter the representation seen by an already-trained model. First
measure how much simulation data falls outside the existing real bounds.
Combined quantiles cannot be safely reconstructed from two sets of summary
quantiles; they require a later approved computation over the actual weighted
data stream.

## Dataloader mixing support

The current LeRobot path constructs exactly one `LeRobotDataset` from one
`repo_id` and gives it to a seeded Torch `DataLoader` with shuffle enabled.

| Capability | Current LeRobot path |
|---|---|
| Multiple datasets | No |
| Weighted sampling | No |
| Concatenation | No built-in config path |
| Task balancing | No |
| Seeded initial shuffle | Yes, from config seed |
| Restore exact shuffle position | No |

An RLDS/DROID path uses `DLataset.sample_from_datasets(weights=...)`, but it is
not applicable to this LeRobot xArm pipeline.

The preferred future implementation is a focused multi-LeRobot sampler that
chooses examples by configured weights with a deterministic seed. It should not
physically duplicate episodes. No loader code was changed in this audit.

## Recommended sampling strategies

### A. Preserve real-robot performance

Start with **80:20 real:simulation by sampled examples**. Preserve exposure to
all six real tasks; if task balancing is added, balance within the real share
without inflating the simulation fraction. Move toward 75:25 only after schema,
distribution, and baseline evaluation checks pass.

### B. MuJoCo-specialist checkpoint

Use **25:75 real:simulation**, with a practical real-data floor of 20–25%.
This makes the red-block simulation task dominant while retaining a real-data
anchor against catastrophic forgetting.

Both are provisional until the actual simulation dataset size, diversity, and
quality are known.

## LR, optimizer, EMA, saving, and evaluation

For a true resume:

- Restore raw parameters, Adam state and count, EMA parameters, and scalar
  step.
- Do not restart warmup.
- For the preservation run, retain the existing schedule. At step 30,000 it is
  at the `2.5e-6` floor, so continuation remains at that floor.
- For a simulation specialist, a later explicit step-offset decay from
  `2.5e-6` toward `1e-6` across 20,000 updates could be evaluated, but the
  current schedule has no offset facility and no implementation was made.
- After confirming stored step `30001`, set `num_train_steps=50001`.
- Save every 5,000 loop steps, targeting manager directories `35000`, `40000`,
  `45000`, and `50000`.
- The JAX training loop has no native evaluation interval. Evaluate separately
  at the source baseline and saved continuations after explicit approval.

## DeltaAI readiness

- Account: `bfmk-dtai-gh` (verified; 622 allocation hours shown at audit time)
- Production partition: `ghx4`, maximum wall time 2 days
- Interactive partition: `ghx4-interactive`, maximum wall time 2 hours
- Node: 4 × NVIDIA GH200 120 GB, 288 CPUs, 488,000 MB memory
- Work filesystem: 2.0 TB total, 534 GB used, 1.5 TB available, 27% full;
  inode usage 2%
- A dedicated work allocation quota was not verified.
- Existing work cache directories include Hugging Face/Xet and pip caches.

Proposed later Slurm request:

```text
account: bfmk-dtai-gh
partition: ghx4
nodes/tasks: 1/1
GPU: 1 × GH200
CPUs: 16
memory: 220G
wall time: 12:00:00
exclusive: no
```

This leaves margin over the historical 30,000-step run while recognizing that
mixing and checkpointing can change throughput. No sbatch file was created and
no job was submitted.

The read-only Slurm-accounting spot check using an ambiguous historical numeric
query matched an unrelated array job, so it is not used as evidence for the
OpenPI runtime recommendation.

## Authentication and caches

No secret value was read or printed.

- Hugging Face: a mode-0600 token file exists in the current HOME cache and
  authenticated `whoami` succeeded; the identity was not printed.
- W&B: package/CLI `0.19.11` is present and local status reports a configured
  API key, redacted. External synchronization is not authorized by this task.
  Disable W&B for continuation or use a separately approved offline workflow.
- Future caches should be explicit subdirectories under
  `/work/nvme/bfmk/mw89/caches`, not the current HOME destinations.

## Proposed run identity

Preservation-oriented initial proposal:

```text
run name:
xarm_pi05_real80_sim20_continue_30000_to_50000_v1

output:
/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/
  xarm_pi05_real80_sim20_continue_30000_to_50000_v1
```

The directory was not created.

## Unresolved blockers

1. MuJoCo root and dataset are absent; all actual simulation metadata and
   validation remain unknown.
2. Weighted multi-LeRobot sampling is not implemented.
3. The checkpoint scalar step must be restored on a compute node.
4. Restore source and output destination are coupled in current code.
5. The checkpoint contains no standalone training configuration.
6. Evaluation is separate from the JAX training loop.
7. Historical Slurm/environment helpers violate current HOME and overwrite
   conventions.
8. Dedicated work and personal HOME quotas remain unverified.
9. External W&B synchronization is not approved.

## Required next actions, without launching training

1. Place or identify the converted MuJoCo dataset at an approved work path and
   provide its source manifest/validator report.
2. Run the bounded schema and distribution validations listed above.
3. Prepare, review, and approve a separate-source checkpoint restore change
   and weighted LeRobot sampler.
4. Restore metadata/state on a short Slurm compute allocation to verify the
   scalar step and one mixed batch.
5. Only then edit a new continuation config and prepare a non-overwriting
   sbatch script in a separate task.
