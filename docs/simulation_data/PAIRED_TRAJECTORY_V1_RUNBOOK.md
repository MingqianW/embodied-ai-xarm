# Paired Trajectory Diversity v1 Runbook

This opt-in schema-v2 plan generates one accepted episode per enabled trajectory
family from each identical initialized scene. It preserves the canonical two RGB
images, 7D state, 7D next-frame action, 10 Hz recording, task text, gripper
conversion, and stable grasp/release acceptance contracts.

For `S = scenes_per_task` and `K` enabled members, each task targets exactly
`S x K` accepted episodes. Each member allocation must equal `S`; each task
allocation must equal `S x K`. Existing schema-v1 episode-count plans retain
their legacy semantics.

The collector reconstructs every member attempt from the same scene seed and
hashes its actual MuJoCo initialized state and initial-condition record. Its
trajectory seed includes base seed, task, scene index, member ID, and retry
index using SHA-256—not Python `hash()`. Retries retain scene and member while
changing only trajectory seed.

An exhausted member marks its group incomplete and keeps successes and failure
diagnostics. Resume requires an exact saved run configuration; profile/member
changes require a new dataset root. Incomplete groups are rejected by raw audit
and cannot enter conversion. Use the exact inspect, smoke, audit, generation,
conversion, and resume commands in
[simulation_data_generation.md](../commands/simulation_data_generation.md).

Review the audit's coverage, fingerprints, attempt/acceptance counts, parameter
coverage, target-relative TCP paths, lengths, peak heights, phase timing,
orientation, and near-duplicate warnings before claiming a diversity gain.
