# Repository instructions for coding agents

These instructions apply to every change in this repository. Read the relevant
package README and `docs/architecture/REPOSITORY_ARCHITECTURE.md` before making
an architectural change.

## Project purpose

This repository provides reusable infrastructure for xArm embodied-AI
experiments. Its five primary capabilities are:

1. simulation data generation;
2. real-robot data collection;
3. simulation evaluation;
4. real-robot evaluation; and
5. VLA model training with real and simulated data.

The repository integrates with the external real-robot collection stack and
owns the downstream discovery/conversion of collected raw data.

The primary research goal is to improve real-robot policy performance using
simulation data. Simulation performance alone is not the final objective.

## Repository map

| Primary capability | Canonical repository area | Scope |
| --- | --- | --- |
| Simulation data generation | `data/sim/generation/` and `simulation/` | Generation/recording/acceptance and canonical MuJoCo scenes, observations, and robot/control mappings. |
| Real-robot data collection | `docs/commands/real_world_data_collection.md` and `data/real/` | Commands plus discovery/conversion of externally collected raw data. Hardware acquisition/control remains in the external xArm collection stack. |
| Simulation evaluation | `evaluation/sim/` | MuJoCo policy evaluation, formal protocols, evidence, videos, and reports. |
| Real-robot evaluation | `evaluation/real/` and `docs/commands/real_robot_evaluation.md` | Explicitly authorized xArm policy evaluation and result recording; never infer success automatically. |
| VLA training with real and simulated data | `training/` and `docs/training/` | Dataset selection, normalization, mixing, OpenPI adaptation, experiment configuration, and preflight. |

Do not create duplicate simulation, robot, camera, gripper, task, or dataset
implementations outside these canonical owners.

## Change discipline

- Work only within the user's requested scope. Preserve unrelated uncommitted
  changes; never reset, clean, overwrite, commit, push, or modify `third_party/`
  unless the user explicitly asks.
- Prefer the smallest change that preserves existing public behavior. Do not
  add compatibility shims, aliases, duplicate CLIs, or mirrored package trees
  without a documented migration and a concrete consumer that requires them.
- Put new runtime products (datasets, checkpoints, videos, logs, reports, and
  caches) outside source packages. Do not commit generated artifacts or secrets.

## Architecture and ownership

- Follow the dependency direction in
  `docs/architecture/REPOSITORY_ARCHITECTURE.md`. In particular, `simulation`
  must not import data, training, evaluation, diagnostics, or cluster code;
  `cluster` invokes canonical CLIs only through subprocesses; and `third_party`
  remains a leaf boundary.
- Use the existing single owners rather than recreating concepts:
  - task IDs and prompts: `data.common.task_identity`;
  - 7D state/action records: `data.common.schema` and `data.common.records`;
  - LeRobot serialization: `data.common.lerobot_writer`;
  - MuJoCo resources/configuration: `simulation.resources` and
    `simulation/config/`;
  - training normalization and mixing: `training.normalization` and
    `training.mixing`.
- Task-specific simulation-data-generation code belongs only in
  `data/sim/generation/tasks/<task_id>/generators/`. Do not introduce a second
  task registry, duplicate prompt list, or mirrored task folder elsewhere.
- Keep shared generation concerns (scene/reset mechanics, recording,
  acceptance, conversion, and identity) centralized. Task generators should
  not write ad-hoc dataset formats or bypass the shared recorder.
- Generation and simulation evaluation must use the same canonical simulation
  stack: assets, scene schema, task definitions, cameras, and robot/control
  mappings. Evaluation uses its own seeds; it may add explicitly named
  out-of-distribution profiles, which must not silently enter training data.

## Data and training conventions

- Real and simulation datasets must expose the compatible canonical training
  interface: two RGB images, 7D state, 7D next-frame action, and canonical task
  text. Perform conversions at dataset or environment boundaries, never inside
  model code.
- Do not silently change action, state, image, task-text, or normalization
  semantics. Version an intentional breaking data or checkpoint contract and
  provide an explicit migration or compatibility decision.
- Keep real/simulation mixing policy separate from dataset implementation. Do
  not hard-code a real-to-simulation ratio in the trainer; declare it in the
  experiment configuration and make its sampling semantics reproducible.
- Reuse upstream OpenPI training infrastructure where practical. The project
  owns adaptation/configuration at its OpenPI boundary, not a forked optimizer
  or training loop.

## Safety and external boundaries

- Treat real-robot motion, physical collection, checkpoint use, policy-server
  launch, and cluster submission as externally gated operations. Do not infer
  missing hardware, credentials, remote files, or human approval.
- Do not launch GPU-intensive, long-running, or Slurm jobs unless the user
  explicitly requests that execution.
- Preserve the canonical 7D action contract and gripper conversion semantics.
  Any intentional contract change requires updates to its producer, consumer,
  validation, tests, and user-facing documentation in the same change.

## Quality bar

- Add or update focused tests for changed behavior. Run the narrowest relevant
  test command first; report commands not run and failures that are unrelated
  to the change.
- For changed configuration, validate that it loads and resolves. For changed
  simulation behavior, run the applicable focused simulation or scene
  regression checks when the required runtime is available. When a shared
  generation/evaluation setting changes, verify both resolve the intended
  canonical setting.
- Do not weaken, delete, or rewrite tests merely to hide a regression.
- Follow the documentation-update policy below for every user-visible change.
- Before handoff, inspect the diff for accidental changes and check formatting
  and import boundaries appropriate to the affected package.

## Documentation-update policy

Update documentation in the same change whenever behavior, configuration, or
an operator workflow changes. Do not leave a correct implementation paired with
stale commands or examples.

| Change type | Documentation that must be reviewed and updated when affected |
| --- | --- |
| Simulation data generation: task selection, episode counts, generators, scene settings, outputs, conversion, audit, or cluster workflow | `docs/commands/simulation_data_generation.md`; update the relevant versioned plan/runbook under `docs/simulation_data/` when a named dataset plan changes. |
| Real-world data collection or conversion | `docs/commands/real_world_data_collection.md`. State clearly whether a command runs in this repository or an external hardware collection stack. |
| Simulation or real-robot evaluation | `docs/commands/real_robot_evaluation.md` for real hardware; `docs/formal_xarm_model_evaluation.md`, `configs/evaluation/sim/protocols/`, and `docs/architecture/EVALUATION_ARCHITECTURE.md` as affected for simulation. Create or update a `docs/commands/` guide when a simulation-evaluation operator workflow changes. Document safety gates and whether success is automatic or human-reviewed. |
| Training, datasets, mixing, normalization, OpenPI setup, checkpoint requirements, or launch commands | `docs/training/openpi_finetuning.md`. |
| Package ownership, dependency direction, canonical paths, or externally owned boundaries | `docs/architecture/REPOSITORY_ARCHITECTURE.md`; update `AGENTS.md` too if the durable agent rule itself changes. |

When updating a command document:

1. Use commands, flags, config names, paths, environment variables, and output
   locations that exist in the current implementation.
2. State prerequisites, inputs, outputs, validation/audit steps, and any
   destructive or real-hardware safety gate.
3. Distinguish verified local commands from environment-specific or external
   commands; do not present unverified historical commands as ready to run.
4. Link to the canonical schema, configuration, or runbook instead of copying
   large contracts into multiple documents.

## Definition of done for code changes

Before reporting a code change complete, confirm that the requested behavior is
implemented, relevant tests and configuration checks have been run or their
limitations reported, existing behavior is not unintentionally changed, and
the architecture/configuration/documentation rules above are satisfied. The
handoff must state files changed, behavior changed, verification performed, and
remaining limitations. For a read-only task, report findings and evidence
instead of claiming implementation work.
