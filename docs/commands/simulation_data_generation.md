# Simulation-data generation

## Task generators

Generation is task-centered: every canonical task owns one or more named
generators, while MuJoCo execution, raw recording, acceptance, conversion, and
LeRobot writing remain shared. The v3 and v4 configs omit `generators` and
therefore resolve to their preserved defaults: `scripted_pick` for every Pick
task and `direct_place` for `place_red_pepper_in_ring`.

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

The allocations above match v3's 25 red-block and 50 place episodes. For a
different plan, each allocation must total that task's `episodes`. Each accepted raw episode and
the collection manifest record `generator_id`, `generator_version`, task, and
resolved seed. Add a generator by creating a task-owned factory under
`data/sim/generation/tasks/<task>/generators/`, registering it explicitly in
`core/registry.py`, then adding it to a config allocation.

Every Pick task (`red_block`, `blue_block`, `red_pepper`, `smallest_block`, and
`largest_block`) provides these non-default geometric variants:

- `scripted_pick_side_approach_v1`: a 25 mm side-offset pregrasp followed by
  a centered diagonal descent;
- `scripted_pick_yaw15_v1`: the same centered grasp with a 15-degree TCP yaw;
- `scripted_pick_waypoint_lift_v1`: an elevated side waypoint before the
  centered grasp, followed by a 10 mm diagonal lift.

`place_red_pepper_in_ring` provides two non-default variants:

- `direct_place_left_approach_v1`: a higher left-front preplace position,
  followed by a centered release into the ring;
- `direct_place_right_approach_v1`: the mirrored right-rear preplace position,
  followed by the same centered release.

All variants preserve task text, final grasp/release semantics, the 7D action
contract, gripper settings, and acceptance criteria. Their exact target poses
and geometric parameters are stored in each episode's `oracle_plan` metadata.
The defaults are unchanged; add these IDs explicitly to an allocation only
when creating a new dataset plan.

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
$config = "configs\data\sim\generation\clean_multitask_stable_v3.yaml"
$dataset = "xarm_mujoco_clean_multitask_stable_v3"
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

### Optional full v3

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
"$XARM_PYTHON" -m cluster.cli submit sim-data-preflight --param plan=v3
"$XARM_PYTHON" -m cluster.cli submit sim-data-initialize --param plan=v3
"$XARM_PYTHON" -m cluster.cli submit sim-data-smoke --param plan=v3
# Review smoke artifacts here.
"$XARM_PYTHON" -m cluster.cli submit sim-data-generate --param plan=v3
"$XARM_PYTHON" -m cluster.cli submit sim-data-convert --param plan=v3
"$XARM_PYTHON" -m cluster.cli submit sim-data-audit --param plan=v3
```

Use `--param plan=v4-10x` for the 1,980-episode plan.

Monitor a job with:

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,JobName%32,State,Elapsed,ExitCode,MaxRSS
```

## v3 outputs

```text
$XARM_WORK_ROOT/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_stable_v3
$XARM_WORK_ROOT/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v3
$XARM_WORK_ROOT/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v3
$XARM_WORK_ROOT/logs/xarm_mujoco_clean_multitask_stable_v3
```

See [DATASET_SCHEMA.md](../simulation_data/DATASET_SCHEMA.md) for the raw and
converted directory layouts, state/action semantics, image streams, and
manifest contracts.

For resume, permissions, and failure recovery, see
[DELTA_AI_RUNBOOK.md](../simulation_data/DELTA_AI_RUNBOOK.md) and
[TROUBLESHOOTING.md](../simulation_data/TROUBLESHOOTING.md).
