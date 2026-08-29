# Phase 6 training inventory and semantic record

This inventory was captured before the training refactor. It records ownership
and prevents the new architecture from laundering one-off tools into core code.

| Former/current item | Classification | Phase 6 disposition |
|---|---|---|
| old xArm config snippet | CONFIG / OPENPI_ADAPTER | merged into structured configs and lazy adapter; deleted duplicate snippet |
| old OpenPI training debug copy | LEGACY / DELETE_CANDIDATE | deleted large modified copy of upstream training loop |
| dataset inspection/outlier/jump scripts | VALIDATION / DATA TOOL | moved to `tools/datasets/` |
| parquet deletion and edge trimming | DATA TOOL (mutating) | moved to `tools/datasets/`; explicit apply behavior preserved |
| OpenPI dataset smoke | VALIDATION | refactored under `training/validation/` |
| Colab notebook | LEGACY / DELETE_CANDIDATE | deleted after its copy-paste config semantics were extracted into tested configs |
| project presentation | EXPERIMENT | moved to `docs/experiments/training/` |
| historical training migration reports | MIGRATION / scientific history | moved to `docs/experiments/migrations/training/` |
| nested data ignore | LEGACY | deleted after root ignore protected the existing untracked archive location |
| simulation-data sbatch files | CLUSTER_WRAPPER / DATA | retained for Phase 7; only stale writer path updated |

The old tracked config snippet defined the xArm repack mapping, six-joint delta
mask, absolute gripper, Pi0.5 dimensions, three real-only configs, and explicit
optimizer/EMA/LR differences. Those fields have structured regression tests.
The upstream default values were read from vendored OpenPI commit `15a9616`.
The historical tracker and later Delta audit showed that `pi05_xarm` was reused
with different dataset, checkpoint, LR, step, and save settings. Those variants
are separate registry entries rather than one misleading composite config.

Historical A/B/C source code was not in this repository or its history. The
tracked evaluation specs prove the stable names, checkpoint/norm identities,
and the key mixing differences: A is 8+8 per batch; B is an exact 1:10
sample-level schedule; C shuffles whole trajectories with no source quota.
The new registry preserves exactly those claims and labels the unavailable
external execution adapter instead of fabricating it.
