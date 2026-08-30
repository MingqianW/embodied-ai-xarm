# DeltaAI simulation-data runbook

All Slurm submission now goes through `cluster.cli`. The generic job script
contains no task, dataset, evaluation, or training logic; each workflow invokes
the canonical package CLI shown by `cluster.cli show`.

For a single copy-and-run command reference covering both Windows local and
DeltaAI generation, see
[simulation_data_generation.md](../commands/simulation_data_generation.md).

## Configure and audit

The checked-in defaults preserve the original DeltaAI deployment:

```bash
export XARM_REPOSITORY=/u/mw89/repos/embodied-ai-xarm
export XARM_WORK_ROOT=/work/nvme/bfmk/mw89
export OPENPI_ROOT=/u/mw89/repos/openpi
export XARM_PYTHON="$OPENPI_ROOT/.venv/bin/python"
export XARM_SLURM_ACCOUNT=bfmk-dtai-gh
export XARM_SLURM_PARTITION=ghx4
cd "$XARM_REPOSITORY"

"$XARM_PYTHON" -m cluster.cli list
"$XARM_PYTHON" -m cluster.cli show sim-data-smoke --param plan=v3
"$XARM_PYTHON" -m cluster.cli submit sim-data-smoke --param plan=v3 --dry-run
```

`XARM_WORK_ROOT` changes the four versioned roots and their exact-root safety
allowlist together. A parent, sibling, or symlink is still rejected. Set
`XARM_CLUSTER_LOG_ROOT` only when cluster run records and Slurm stdout/stderr
need a separate location.

## Run a data plan

Initialization deliberately replaces only the exact versioned log root. Review
the dry-run command first, submit it once, then advance one audited phase at a
time:

```bash
"$XARM_PYTHON" -m cluster.cli submit sim-data-preflight --param plan=v3
"$XARM_PYTHON" -m cluster.cli submit sim-data-initialize --param plan=v3 --dry-run
"$XARM_PYTHON" -m cluster.cli submit sim-data-initialize --param plan=v3
"$XARM_PYTHON" -m cluster.cli submit sim-data-smoke --param plan=v3
# Review SMOKE_AUDIT.md and every contact sheet.
"$XARM_PYTHON" -m cluster.cli submit sim-data-generate --param plan=v3
# Require complete=true and RAW_PASS.
"$XARM_PYTHON" -m cluster.cli submit sim-data-convert --param plan=v3
"$XARM_PYTHON" -m cluster.cli submit sim-data-audit --param plan=v3
```

Use `--param plan=v4-10x` with the same six workflows for the 1,980-episode plan.
Every job writes a machine-readable record under
`$XARM_CLUSTER_LOG_ROOT/runs/WORKFLOW/JOB_ID.json`; data phases also update the
existing `CODEX_STATUS` files under the dataset log root.

## Monitor and resume

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,JobName%32,State,Elapsed,ExitCode,MaxRSS
tail -n 100 "$XARM_CLUSTER_LOG_ROOT/slurm/xarm-WORKFLOW-JOB_ID.out"
cat "$XARM_CLUSTER_LOG_ROOT/runs/WORKFLOW/JOB_ID.json"
```

Collection retains atomic accepted/failed state, but the maintained full-run
workflow intentionally starts with the canonical CLI's explicit `--overwrite`.
Use its `--resume` capability directly only after inspecting the interrupted
run and confirming the saved config is identical.

Cluster jobs apply the canonical `delta_bfmk` permission policy. The final
audit also runs `namei` and `getfacl`; their output is retained in the Slurm log
and the command/result sequence is retained in the run record.
