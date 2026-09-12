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
$config = "configs\data\sim\generation\clean_multitask_stable_v4_10x_real.yaml"
$dataset = "xarm_mujoco_clean_multitask_stable_v4_10x_real"
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

# Requires an explicit multi-generator allocation in $config.
# Writes one accepted smoke episode for every listed generator.
& $python -m data.sim.generation.cli generate `
  --config $config --output $smoke --smoke --smoke-all-generators --overwrite

& $python -m data.sim.generation.cli audit `
  --config $config --raw $smoke --report-dir $log `
  --decode-all-images --smoke

# For --smoke-all-generators generation, use the matching audit mode.
& $python -m data.sim.generation.cli audit `
  --config $config --raw $smoke --report-dir $log `
  --decode-all-images --smoke --smoke-all-generators

Invoke-Item "$log\SMOKE_AUDIT.md"
explorer "$smoke\accepted"
```

The smoke audit must pass for all six tasks.

### Full v4 10x-real collection

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
"$XARM_PYTHON" -m cluster.cli submit sim-data-preflight --param plan=v4-10x
"$XARM_PYTHON" -m cluster.cli submit sim-data-initialize --param plan=v4-10x
"$XARM_PYTHON" -m cluster.cli submit sim-data-smoke --param plan=v4-10x
# Review smoke artifacts here.
"$XARM_PYTHON" -m cluster.cli submit sim-data-generate --param plan=v4-10x
"$XARM_PYTHON" -m cluster.cli submit sim-data-convert --param plan=v4-10x
"$XARM_PYTHON" -m cluster.cli submit sim-data-audit --param plan=v4-10x
```

Monitor a job with:

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,JobName%32,State,Elapsed,ExitCode,MaxRSS
```

## v4 10x-real outputs

```text
$XARM_WORK_ROOT/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_stable_v4_10x_real
$XARM_WORK_ROOT/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v4_10x_real
$XARM_WORK_ROOT/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v4_10x_real
$XARM_WORK_ROOT/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real
```

See [DATASET_SCHEMA.md](../simulation_data/DATASET_SCHEMA.md) for the raw and
converted directory layouts, state/action semantics, image streams, and
manifest contracts.

For resume, permissions, and failure recovery, see
[DELTA_AI_RUNBOOK.md](../simulation_data/DELTA_AI_RUNBOOK.md) and
[TROUBLESHOOTING.md](../simulation_data/TROUBLESHOOTING.md).
