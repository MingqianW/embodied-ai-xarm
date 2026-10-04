# Simulation-data generation

## Task generators

Generation is task-centered: every canonical task owns one or more named
generators, while MuJoCo execution, raw recording, acceptance, conversion, and
LeRobot writing remain shared. The v4 plan explicitly allocates every
registered Pick and Place variant.

Use an explicit exact allocation when a task has more than one registered
generator:

```yaml
tasks:
  red_block:
    # existing prompt, seed, object, and oracle fields stay unchanged
    generators:
      scripted_pick: {episodes: 10}
      scripted_pick_side_approach_v1: {episodes: 5}
      scripted_pick_yaw15_v1: {episodes: 5}
      scripted_pick_waypoint_lift_v1: {episodes: 5}
  place_red_pepper_in_ring:
    # existing prompt, seed, object, and oracle fields stay unchanged
    generators:
      direct_place: {episodes: 30}
      direct_place_left_approach_v1: {episodes: 10}
      direct_place_right_approach_v1: {episodes: 10}
```

For a different plan, each allocation must total that task's `episodes`. Each accepted raw episode and
the collection manifest record `generator_id`, `generator_version`, task, and
resolved seed. Add a generator by creating a task-owned factory under
`data/sim/generation/tasks/<task>/generators/`, registering it explicitly in
`core/registry.py`, then adding it to a config allocation.

Every Pick task (`red_block`, `blue_block`, `red_pepper`, `smallest_block`, and
`largest_block`) provides these non-default geometric variants:

- `scripted_pick_side_approach_v1`: a 25 mm side-offset pregrasp followed by
  a centered diagonal descent;
- `scripted_pick_opposite_side_approach_v1`: the mirrored 25 mm side-offset
  pregrasp and centered diagonal descent;
- `scripted_pick_yaw15_v1`: the same centered grasp with a 15-degree TCP yaw;
- `scripted_pick_yaw_minus15_v1`: the mirrored -15-degree TCP yaw;
- `scripted_pick_diagonal_approach_v1`: a 20 mm diagonal pregrasp offset,
  followed by the centered grasp;
- `scripted_pick_waypoint_lift_v1`: an elevated side waypoint before the
  centered grasp, followed by a 10 mm diagonal lift.
- `scripted_pick_opposite_waypoint_lift_v1`: the mirrored waypoint and lift
  path.

`place_red_pepper_in_ring` provides two non-default variants:

- `direct_place_left_approach_v1`: a higher left-front preplace position,
  followed by a centered release into the ring;
- `direct_place_right_approach_v1`: the mirrored right-rear preplace position,
  followed by the same centered release.
- `direct_place_high_center_v1`: a higher centered preplace pose before the
  vertical release;
- `direct_place_left_rear_approach_v1` and
  `direct_place_right_front_approach_v1`: the two remaining diagonal preplace
  positions, each followed by a centered release.

All variants preserve task text, final grasp/release semantics, the 7D action
contract, gripper settings, and acceptance criteria. Their exact target poses
and geometric parameters are stored in each episode's `oracle_plan` metadata.
The defaults are unchanged; add these IDs explicitly to an allocation only
when creating a new dataset plan.

## Object-layout randomization

Place now uses the existing free `red_pepper` throughout reset, validation,
transfer and release. The canonical four-bar reset initializes its TCP-relative
pose once and then settles through ordinary contact physics; it does not swap
a fixed held fixture into a free body. Both generation plans declare this same
body identity, initial raw target and transform, and configuration validation
rejects mismatches with `simulation/config/task_scenes.yaml`. The task text,
randomization profile, camera/gripper mapping and acceptance thresholds are
unchanged. Preserve existing outputs from the earlier held-fixture convention;
use a new run root and provenance rather than resuming that data after the
reset correction. The default Place generator version is unchanged; the
source/configuration and initial-condition metadata identify this reset.

The canonical simulation config owns named randomization profiles. The current
v4 10x-real plan and formal evaluation v3 both select `clean_wide_v4`. It samples every active task object within
±10 cm of its nominal position, with independent offsets and the comparison
block separation gate. The chosen profile and actual per-object XY deltas are
stored in every raw episode's initial-condition metadata. Do not change a
profile for an existing dataset or evaluation root; create a named profile and
run generation and evaluation smoke/audit first.

The v4 plan explicitly allocates all eight Pick trajectories and all six Place
trajectories.  `--smoke --smoke-all-generators` therefore validates each of the
46 task/generator pairs once under the same wide layout profile used for its
full collection.

To collect one accepted smoke episode for every generator declared in an
explicit allocation, use `--smoke-all-generators`. It is opt-in and does not
change normal `--smoke`, which still collects one episode per task.

For a bounded direct run (the output still must be an authorized generation
root):

```powershell
& $python -m data.sim.generation.cli generate `
  --config $config --task red_block --generator scripted_pick --episodes 10 `
  --output $smoke --overwrite
```

## Windows local

Run from the repository root. Do not set `MUJOCO_GL=egl` on Windows.

```powershell
$python = "D:\miniconda\envs\mujoco-pi\python.exe"
$env:XARM_WORK_ROOT = "D:\xarm-work"
$config = "configs\data\sim\generation\clean_multitask_paired_trajectory_v1.yaml"
$dataset = "xarm_mujoco_clean_multitask_paired_trajectory_v1"
$raw = "$env:XARM_WORK_ROOT\mujoco_datasets\raw\$dataset"
$converted = "$env:XARM_WORK_ROOT\mujoco_datasets\local\$dataset"
$smoke = "$env:XARM_WORK_ROOT\mujoco_datasets\smoke\$dataset"
$log = "$env:XARM_WORK_ROOT\logs\$dataset"

& $python -m data.sim.generation.cli inspect --config $config
& $python -m diagnostics.simulation.environment.check
```

### Smoke

```powershell
& $python -m data.sim.generation.cli generate `
  --config $config --output $smoke --smoke --overwrite

& $python -m data.sim.generation.cli audit `
  --config $config --raw $smoke --report-dir $log `
  --decode-all-images --smoke

Invoke-Item "$log\SMOKE_AUDIT.md"
explorer "$smoke\accepted"
```

The smoke audit must pass for all six tasks.

### Full paired-v1 collection

```powershell
& $python -m data.sim.generation.cli generate `
  --config $config --output $raw --overwrite

& $python -m data.sim.generation.cli audit `
  --config $config --raw $raw --report-dir $log --decode-all-images

& $python -m data.sim.generation.cli convert `
  --config $config --raw $raw --output $converted --overwrite

& $python -m data.sim.generation.cli audit `
  --config $config --raw $raw --converted $converted `
  --report-dir $log --decode-all-images

& $python -m data.sim.generation.cli handoff --config $config
```

Windows smoke is verified locally. DeltaAI is the recommended production path
for the complete dataset.

## DeltaAI

Run on a DeltaAI login node:

```bash
export XARM_REPOSITORY=/u/mw89/repos/embodied-ai-xarm
export XARM_WORK_ROOT=/work/nvme/bfmk/mw89
export OPENPI_ROOT=/u/mw89/repos/openpi
export XARM_PYTHON="$OPENPI_ROOT/.venv/bin/python"
export XARM_SLURM_ACCOUNT=bfmk-dtai-gh
export XARM_SLURM_PARTITION=ghx4
export XARM_CLUSTER_LOG_ROOT="$XARM_WORK_ROOT/logs/cluster"
cd "$XARM_REPOSITORY"
```

Submit one phase at a time and wait for success before continuing:

```bash
"$XARM_PYTHON" -m cluster.cli submit sim-data-preflight --param plan=paired-v1
"$XARM_PYTHON" -m cluster.cli submit sim-data-initialize --param plan=paired-v1
"$XARM_PYTHON" -m cluster.cli submit sim-data-smoke --param plan=paired-v1
# Review smoke artifacts here.
"$XARM_PYTHON" -m cluster.cli submit sim-data-generate --param plan=paired-v1
"$XARM_PYTHON" -m cluster.cli submit sim-data-convert --param plan=paired-v1
"$XARM_PYTHON" -m cluster.cli submit sim-data-audit --param plan=paired-v1
```

Monitor a job with:

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,JobName%32,State,Elapsed,ExitCode,MaxRSS
```

## Paired-v1 outputs

```text
$XARM_WORK_ROOT/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_paired_trajectory_v1
$XARM_WORK_ROOT/mujoco_datasets/raw/xarm_mujoco_clean_multitask_paired_trajectory_v1
$XARM_WORK_ROOT/mujoco_datasets/local/xarm_mujoco_clean_multitask_paired_trajectory_v1
$XARM_WORK_ROOT/logs/xarm_mujoco_clean_multitask_paired_trajectory_v1
```

See [DATASET_SCHEMA.md](../simulation_data/DATASET_SCHEMA.md) for the raw and
converted directory layouts, state/action semantics, image streams, and
manifest contracts.

## Paired trajectory scene groups (v1, opt-in)

`clean_multitask_paired_trajectory_v1.yaml` is a schema-v2 plan for evaluating
trajectory diversity independently from the existing `clean_wide_v4` scene
profile. Its meaning differs deliberately from an episode allocation:
`scenes_per_task: N` produces one accepted episode for every enabled family
member in each of N initialized scenes. A Pick task with eight members produces
`8N` episodes; the Place task with six members produces `6N`. v4 remains
`legacy_episodes` and is not reinterpreted.

Each `(task, scene_index, family_member)` derives SHA-256 seeds from stable
identifiers. The scene seed uses base seed, task, and scene; trajectory seed
also uses family member and retry. Each attempt reconstructs the same scene
from its scene seed and records an actual initialized-state fingerprint before
sampling trajectory parameters. Member order changes or an added member cannot
alter another member's seeds.

The versioned `collection.trajectory_profile` supports `fixed`, `uniform`, and
`truncated_normal` distributions. Values are sampled once per episode, bounded,
and recorded. Member parameters override task parameters, which override
defaults. Supported Pick values are XY approach/waypoint/lift offsets (m),
pregrasp/lift clearance (m), wrist yaw (deg), and approach/lift speed scales;
Place supports preplace XY offset (m), preplace height (m), and transfer speed.
The v1 validator uses conservative +/-40 mm XY and +/-25 degree limits.

Enabled members are the existing validated centered, mirrored-side, diagonal,
yaw, and waypoint/lateral-lift Pick paths; and centered, cardinal/diagonal
preplace, and high-transfer Place paths. Curved paths are not enabled because
the current controller is a collision-checked joint-space waypoint state
machine without a validated Cartesian-curve feasibility check. Pepper bounds
are deliberately conservative because initial grasp/release clearance is tight.

A failed member retries only from the same scene and member identity with a new
trajectory seed. Exhaustion marks its scene group incomplete, retaining
diagnostics and successful members for resume. Resume requires an exact saved
run configuration; incomplete groups cannot pass raw audit or conversion.

```powershell
$paired = "configs\data\sim\generation\clean_multitask_paired_trajectory_v1.yaml"
$pairedSmoke = "$env:XARM_WORK_ROOT\mujoco_datasets\smoke\xarm_mujoco_clean_multitask_paired_trajectory_v1"
$pairedRaw = "$env:XARM_WORK_ROOT\mujoco_datasets\raw\xarm_mujoco_clean_multitask_paired_trajectory_v1"
$pairedLocal = "$env:XARM_WORK_ROOT\mujoco_datasets\local\xarm_mujoco_clean_multitask_paired_trajectory_v1"
$pairedLog = "$env:XARM_WORK_ROOT\logs\xarm_mujoco_clean_multitask_paired_trajectory_v1"

& $python -m data.sim.generation.cli inspect --config $paired
# Bounded smoke: one scene per task, one episode for every enabled family.
& $python -m data.sim.generation.cli generate --config $paired --output $pairedSmoke --smoke --overwrite
& $python -m data.sim.generation.cli audit --config $paired --raw $pairedSmoke --report-dir $pairedLog --smoke --decode-all-images

# Full v1 plan (the checked-in example has N=2).
& $python -m data.sim.generation.cli generate --config $paired --output $pairedRaw --overwrite
& $python -m data.sim.generation.cli audit --config $paired --raw $pairedRaw --report-dir $pairedLog --decode-all-images
& $python -m data.sim.generation.cli convert --config $paired --raw $pairedRaw --output $pairedLocal --overwrite
& $python -m data.sim.generation.cli generate --config $paired --output $pairedRaw --resume
```

The paired audit checks complete tuple coverage, common initialized-state
fingerprints, duplicate/missing members, attempt and acceptance counts,
sampled versus accepted parameters, and target-relative TCP path length,
peak height, timing, orientation, and near-duplicate warnings.

For resume, permissions, and failure recovery, see
[DELTA_AI_RUNBOOK.md](../simulation_data/DELTA_AI_RUNBOOK.md) and
[TROUBLESHOOTING.md](../simulation_data/TROUBLESHOOTING.md).
