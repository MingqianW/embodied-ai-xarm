# Documentation map

## Current architecture

- `architecture/REPOSITORY_ARCHITECTURE.md`: authoritative package ownership
  and dependency direction.
- `architecture/EVALUATION_ARCHITECTURE.md`: evaluation-specific contracts and
  real-robot boundary.
- `data/TRAINING_DATA_CONTRACT.md`: shared training record semantics.
- `simulation_data/ARCHITECTURE.md`: simulation-data generation ownership.

`architecture/REPOSITORY_REORGANIZATION_PLAN.md` is migration history, not a
current operating guide.

## Current runbooks

- `commands/simulation_data_generation.md`
- `simulation_data/DELTA_AI_RUNBOOK.md`
- `formal_xarm_model_evaluation.md`
- `mujoco_openpi_remote_inference_runbook.md`
- `mujoco_task_scenes.md`
- `training/README.md` and `training/openpi_finetuning.md`
- `../cluster/README.md`
- `../diagnostics/README.md`
- `../evaluation/real/README.md`

## Experiment and migration records

`experiments/` and `mujoco_migration/` preserve dated observations, generated
reports, and migration evidence. Their paths and commands describe the state
at the time recorded and are not supported entrypoints. Dated tracker and
evaluation-log files in this directory are likewise records rather than
architecture definitions.
