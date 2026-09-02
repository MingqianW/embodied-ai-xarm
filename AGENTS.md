# Repository instructions for coding agents

These instructions apply to every change in this repository. Read the relevant
package README and `docs/architecture/REPOSITORY_ARCHITECTURE.md` before making
an architectural change.

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

## Safety and external boundaries

- Treat real-robot motion, physical collection, checkpoint use, policy-server
  launch, and cluster submission as externally gated operations. Do not infer
  missing hardware, credentials, remote files, or human approval.
- Preserve the canonical 7D action contract and gripper conversion semantics.
  Any intentional contract change requires updates to its producer, consumer,
  validation, tests, and user-facing documentation in the same change.

## Quality bar

- Add or update focused tests for changed behavior. Run the narrowest relevant
  test command first; report commands not run and failures that are unrelated
  to the change.
- Keep operator-facing commands and configuration names accurate in `docs/`.
  Update architecture documentation when ownership or dependency direction
  changes.
- Before handoff, inspect the diff for accidental changes and check formatting
  and import boundaries appropriate to the affected package.
