# Canonical task folders

This is the repository-wide implementation owner for the six canonical xArm
tasks in `data.common.task_identity`. Every task has a folder named exactly by
its `task_id`, with task-owned generator implementations below `generators/`.

Simulation scenes, evaluation protocols, training datasets, diagnostics, and
real-data utilities consume those canonical identities; they must not create
parallel task registries or duplicate task folders. Shared cameras, MuJoCo
assets, reset configuration, recording, acceptance, conversion, and prompts
remain in their existing centralized owners.
