# DeltaAI runbook

These commands assume no persistent terminal. Every heavy phase is a standalone
`sbatch` job and continues independently after SSH disconnect. Codex itself does
not continue after a disconnect.

## Login-node checks

```bash
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
printf 'SLURM_NODELIST=%s\n' "${SLURM_NODELIST:-}"
cd /u/mw89/repos/embodied-ai-xarm
git branch --show-current
git rev-parse HEAD
git status --short
accounts
sinfo -s
```
Validate the config without rendering:

```bash
/u/mw89/repos/openpi/.venv/bin/python \
  -m data.sim.generation.cli inspect \
  --config configs/data/sim/generation/clean_multitask_stable_v3.yaml
```

The four exact roots are fixed in the YAML. Do not substitute a parent, sibling,
symlink, implicit output variable, or `_v4` path.

## Initialize logs and run offline tests

Initialization requires the explicit scoped replacement flag:

```bash
/u/mw89/repos/openpi/.venv/bin/python \
  -m data.sim.generation.cli inspect \
  --config configs/data/sim/generation/clean_multitask_stable_v3.yaml \
  --initialize-log-root --overwrite

sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/offline_tests.sbatch
```

For a physical Pick-grasp regression investigation (not a dataset phase), use
the documented compute-only diagnostic and inspect its JSON summary:

```bash
sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/pick_grasp_sweep.sbatch
cat "$LOG_ROOT/diagnostics/pick_grasp_sweep_JOB_ID/sweep_summary.json"
```

Revalidate the canonical Place initial grasp across 20 deterministic seeds
after changing reset poses, gripper actuation, or pepper contact geometry:

```bash
sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/place_grasp_sweep.sbatch
sacct -j JOB_ID --format=JobID,JobName%32,State,Elapsed,ExitCode
cat "$LOG_ROOT/diagnostics/place_grasp_sweep_JOB_ID/summary.json"
```

## Smoke, full generation, conversion, audit

Submit one phase at a time and inspect its result before the next:

```bash
sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/smoke.sbatch
sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/full_generation.sbatch
sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/conversion.sbatch
sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/final_audit.sbatch
```

Do not submit full generation until `SMOKE_AUDIT.md` says `PASS` and all six
contact sheets have been explicitly inspected. Do not submit conversion until
the raw summary says `complete: true` and `RAW_DATASET_AUDIT.md` says `RAW_PASS`.

## Monitor and resume after disconnect

```bash
LOG_ROOT=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3
cat "$LOG_ROOT/CODEX_STATUS.md"
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,JobName%32,State,Elapsed,ExitCode
tail -n 100 "$LOG_ROOT/slurm/xarm-v3-PHASE-JOB_ID.out"
tail -n 100 "$LOG_ROOT/slurm/xarm-v3-PHASE-JOB_ID.err"
```

`CODEX_STATUS.md`, `CODEX_STATUS.json`, and `status/<phase>-<job>.json` contain
resolved paths, commit, job history, failures, next action, and exact resumption
commands. Collection retains atomic accepted/failed state, but a new full run
intentionally starts with `--overwrite`; use the documented `generate --resume`
only after inspecting an interrupted run and confirming the saved config matches.

## Permissions and final locations

The jobs apply `delta_bfmk` ownership, group read/traverse, directory setgid,
and default ACLs when supported. Final verification is recorded in
`PERMISSIONS_REPORT.txt` using `namei` and `getfacl`.

```text
/work/nvme/bfmk/mw89/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v3
/work/nvme/bfmk/mw89/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v3
/work/nvme/bfmk/mw89/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_stable_v3
/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3
```
