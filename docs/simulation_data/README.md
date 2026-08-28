# Reusable MuJoCo simulation-data pipeline

This pipeline generates clean, oracle-controlled xArm MuJoCo demonstrations in
the real-recorder-compatible raw layout and converts them to the repository's
canonical LeRobot/OpenPI format. It supports five Pick tasks and one Place task.
The camera source of truth is
`simulation/config/camera_calibration.yaml`; collection never substitutes older
calibration values.

The versioned collection plan is
`sim_mujoco/config/data_generation/clean_multitask_stable_v3.yaml`. It requests
200 accepted episodes, six canonical prompts, and zero distractor episodes.

## Quick start

Run lightweight config inspection on a login node:

```bash
cd /u/mw89/repos/embodied-ai-xarm
/u/mw89/repos/openpi/.venv/bin/python \
  -m sim_mujoco.data_generation.cli inspect \
  --config sim_mujoco/config/data_generation/clean_multitask_stable_v3.yaml
```

Run physics, rendering, collection, conversion, and full decoding through the
self-contained Slurm jobs documented in [DELTA_AI_RUNBOOK.md](DELTA_AI_RUNBOOK.md).
The operational order is offline tests, smoke, explicit contact-sheet review,
full generation, conversion, and final audit.

Stable CLI interfaces are:

```bash
python -m sim_mujoco.data_generation.cli generate --config CONFIG --output RAW --overwrite
python -m sim_mujoco.data_generation.cli convert --config CONFIG --raw RAW --output CONVERTED --overwrite
python -m sim_mujoco.data_generation.cli audit --config CONFIG --raw RAW --converted CONVERTED --report-dir LOG
python -m sim_mujoco.data_generation.cli inspect --config CONFIG
```

`--overwrite` is always explicit. The safety layer accepts only the four exact
v3 roots in the config, rejects symlinks and parent/sibling paths, writes a
pre-overwrite inventory outside the replaced root, and records an overwrite
marker. No sibling dataset or earlier version is eligible.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): components and extension points
- [TASKS_AND_PROMPTS.md](TASKS_AND_PROMPTS.md): IDs, prompts, and aliases
- [DATASET_SCHEMA.md](DATASET_SCHEMA.md): raw and converted contracts
- [DELTA_AI_RUNBOOK.md](DELTA_AI_RUNBOOK.md): exact cluster commands
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): diagnosis and safe recovery
