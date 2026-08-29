# DeltaAI / Slurm execution layer

`cluster/` is the only maintained Slurm orchestration tree in this repository.
It owns resource requests, deployment paths, runtime environment variables,
submission dependencies, log locations, ordered CLI invocation, and run
provenance. Scientific and training behavior remains in `simulation/`, `data/`,
`evaluation/`, `diagnostics/`, and `training/`.

The only sbatch file is `jobs/run_workflow.sbatch`. It validates that it is in
Slurm and delegates to `python -m cluster.cli run`. Resource options are passed
directly to `sbatch` by `cluster.cli`; they are not hidden in sourced files that
Slurm cannot parse. Inspect any resolved request and command before submission:

```bash
python -m cluster.cli list
python -m cluster.cli show WORKFLOW [--param NAME=VALUE ...]
python -m cluster.cli command WORKFLOW [--param NAME=VALUE ...]
python -m cluster.cli submit WORKFLOW --dry-run [--param NAME=VALUE ...]
```

`submit` without `--dry-run` is the only operation that calls `sbatch`. The
generic runner refuses local execution unless `--allow-local` is deliberately
supplied.

## DeltaAI environment contract

The preserved deployment defaults are:

```bash
export XARM_REPOSITORY=/u/mw89/repos/embodied-ai-xarm
export XARM_WORK_ROOT=/work/nvme/bfmk/mw89
export OPENPI_ROOT=/u/mw89/repos/openpi
export XARM_PYTHON="$OPENPI_ROOT/.venv/bin/python"
export XARM_SLURM_ACCOUNT=bfmk-dtai-gh
export XARM_SLURM_PARTITION=ghx4
```

The validated simulation environment expects Python 3.11, an allocated NVIDIA
GPU for EGL jobs, `MUJOCO_GL=egl`, and `PYOPENGL_PLATFORM=egl`. The submitter
must provide an existing OpenPI checkout and environment with the dependencies
required by the selected canonical CLI. No module name or module version is
guessed: prior jobs used the absolute OpenPI virtual-environment interpreter,
so this layer does the same. Hugging Face authentication is an external user or
cluster secret when a selected upstream operation needs remote private assets;
credentials are never accepted as workflow parameters or written to run records.

Common cache variables are derived from `XARM_WORK_ROOT`:

- `HF_LEROBOT_HOME=$XARM_WORK_ROOT/mujoco_datasets`
- `MUJOCO_OUTPUT_ROOT` and `MUJOCO_DATASET_ROOT` under `XARM_WORK_ROOT`
- `HF_HOME`, `HF_HUB_CACHE`, and `HF_DATASETS_CACHE` under `caches/huggingface`
- `UV_CACHE_DIR=$XARM_WORK_ROOT/caches/uv`

`XARM_CLUSTER_LOG_ROOT` defaults to `$XARM_WORK_ROOT/logs/cluster`. All roots,
the account, partition, interpreter, and OpenPI checkout can be overridden.
POSIX deployment paths remain POSIX when commands are audited from Windows.

## Workflows and resource ownership

| Workflow | Canonical owner | Resources | Required parameters |
|---|---|---|---|
| `sim-data-preflight` | `data.sim.generation.cli inspect` | 4 CPU, 24G, 30m, no GPU | optional `plan=v3|v4-10x` |
| `sim-data-initialize` | `data.sim.generation.cli inspect` | 2 CPU, 8G, 30m, no GPU | optional `plan` |
| `sim-data-smoke` | data CLI plus environment diagnostic | 8 CPU, 64G, 2h, 1 GPU | optional `plan` |
| `sim-data-generate` | `data.sim.generation.cli` | 8 CPU, 64G, 12h, 1 GPU | optional `plan` |
| `sim-data-convert` | `data.sim.generation.cli` | 8 CPU, 64G, 6h, 1 GPU | optional `plan` |
| `sim-data-audit` | data CLI plus OS ACL inspection | 8 CPU, 64G, 6h, 1 GPU | optional `plan` |
| `environment-check` | `diagnostics.simulation.environment.check` | 4 CPU, 24G, 2h, 1 GPU | none |
| `physics-consistency` | `diagnostics.simulation.physics.consistency` | 4 CPU, 24G, 30m, no GPU | `output` |
| `export-training-videos` | `tools.datasets.export_lerobot_training_videos` | 8 CPU, 64G, 2h, 1 GPU | none |
| `formal-sim-evaluation` | `evaluation.sim.cli` | 16 CPU, 128G, 12h, 1 GPU | `model_spec`, `host` |
| `training-preflight` | `training.cli preflight` | 4 CPU, 24G, 30m, no GPU | `config`, `dataset_paths` |
| `training` | `training.cli train` | 16 CPU, 220G, 12h, 1 GPU | `config`, `exp_name` |

The data and export profiles preserve their former tracked requests. The
training profile matches the documented DeltaAI OpenPI request. The evaluation
client profile is repository-owned but still requires measurement on DeltaAI;
it is not claimed to be tuned. CPU-only preflight and physics checks do not
reserve a GPU.

## Former Slurm job disposition

| Former files | Classification | Disposition |
|---|---|---|
| `common.sh`, `common_v4_10x.sh` | common environment | **MERGE** into `cluster.config` and the generic runner |
| v3/v4 smoke, generation, conversion, and audit sbatch families | data generation | **MERGE/REFACTOR** into plan-parameterized `sim-data-*` workflows |
| `export_mujoco_training_videos.sbatch` | export | **REFACTOR** into `export-training-videos` and the canonical dataset tool |
| `offline_tests.sbatch` | test/preflight | **DELETE**; stale paths and cluster CI logic were superseded by local tests and canonical preflight |
| `pick_grasp_sweep.sbatch`, `place_grasp_sweep.sbatch` | obsolete diagnostics | **DELETE**; completed one-off tuning with inline Python and stale targets |
| `place_initial_diagnostic.sbatch` | obsolete diagnostic | **DELETE**; targeted a removed script and embedded result aggregation |

No job met the bar for `LEGACY_RETAIN`; Git history is the archive. The former
`slurm/` tree is empty and removed, so it no longer competes with `cluster/`.

## Data generation and dependencies

The same launchers serve both versioned plans; scientific differences remain in
their checked-in YAML configs:

```bash
python -m cluster.cli submit sim-data-preflight --param plan=v4-10x --dry-run
python -m cluster.cli submit sim-data-initialize --param plan=v4-10x
python -m cluster.cli submit sim-data-smoke --param plan=v4-10x
```

Review smoke artifacts before full generation. The required order is
initialize, smoke and human review, generate, convert, then audit. Interactive
review means the default runbook does not silently chain every phase. After a
phase has been reviewed, dependencies can be explicit:

```bash
GEN_JOB=$(python -m cluster.cli submit sim-data-generate --param plan=v4-10x \
  | sed -n 's/^submitted_job_id=//p')
python -m cluster.cli submit sim-data-convert --param plan=v4-10x \
  --dependency "afterok:$GEN_JOB"
```

No retained workflow previously used a Slurm array, and generation owns exact
task/seed ordering internally. Phase 7 therefore introduces no array mapping.

## Formal simulation evaluation

The evaluator is a client of an already-running OpenPI policy server. The
server host is required so the cluster layer cannot pretend that localhost is
valid across separately scheduled nodes:

```bash
python -m cluster.cli submit formal-sim-evaluation --dry-run \
  --param model_spec=configs/evaluation/sim/models/A.json \
  --param host=POLICY_SERVER_HOST
```

The protocol defaults to the canonical formal v2 config and its exact isolated
output root. Optional `protocol`, `output_root`, `port`, `timeout`, and `resume`
parameters remain subject to the evaluator's validation. The evaluator rejects
unverified server provenance, incomplete checkpoints, output-root mismatch,
and invalid episodes. Starting the upstream server is still a manual external
OpenPI operation because this repository has no validated canonical server
launcher; the cluster layer does not fabricate one.

There is deliberately no real-robot job. Real evaluation remains inside the
human-controlled safety boundary documented by `evaluation/real/`.

## Training

Preflight example (semicolon separates repeated `DATASET_ID=PATH` values):

```bash
python -m cluster.cli submit training-preflight --dry-run \
  --param config=pi05_xarm \
  --param 'dataset_paths=DATASET_ID=/path/to/dataset'
```

Training requires an explicit canonical config and experiment name. Assets and
checkpoints default to `$XARM_WORK_ROOT/openpi_assets` and
`$XARM_WORK_ROOT/openpi_checkpoints`, outside the checkout. `assets_dir` and
`checkpoint_dir` are optional overrides. Checkpoint initialization, warm-start,
or full-resume mode belongs to the selected immutable training config and is not
overridden by Slurm. The canonical OpenPI adapter still refuses execution-disabled
historical configs; the launcher does not turn incomplete evidence into a
runnable experiment.

## Outputs, logging, and provenance

Generated datasets, formal evaluation results, logs, checkpoints, videos, and
reports default under `XARM_WORK_ROOT`, not source packages. Simulation-data
replacement still accepts only exact versioned roots and records an inventory
before replacement.

Slurm stdout and stderr use
`$XARM_CLUSTER_LOG_ROOT/slurm/%x-%j.{out,err}`. Job names include the workflow
and plan/config/model identity when available. Every run writes
`$XARM_CLUSTER_LOG_ROOT/runs/WORKFLOW/JOB_ID.json` containing resolved commands,
parameters, resources, return codes, timestamps, repository/OpenPI revisions,
working-tree status, Slurm identity/dependency, host, Python, non-secret runtime
environment, and GPU inventory when available.

No jobs are submitted during repository validation. The generated commands,
DeltaAI paths, external policy-server reachability, GPU/EGL visibility, datasets,
checkpoints, and actual resource sufficiency still require manual validation on
DeltaAI before production use.
