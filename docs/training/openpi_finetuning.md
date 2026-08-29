# xArm OpenPI training

The repository owns experiment identity, canonical dataset declarations,
real/simulation mixing, normalization selection, preflight, and the xArm data
adapter. Physical-Intelligence/OpenPI remains the optimization engine. The
canonical architecture and historical experiment table are in
[`README.md`](README.md).

## Inspect and validate

List configs and print a resolved config without importing OpenPI:

```bash
python -m training.cli list
python -m training.cli show pi05_xarm
```

Preflight accepts explicit local paths, keyed by the dataset ID shown in the
resolved config:

```bash
python -m training.cli preflight pi05_xarm \
  --dataset-path real_xarm_pi05_data=/path/to/lerobot/local/xarm_pi05_data
```

Preflight does not download data, compute normalization statistics, initialize
a model, or launch training. Missing remote datasets, checkpoint assets, or
runtime dependencies appear as `unresolved`; a report can pass its static
checks while correctly reporting `launch_ready: false`.

The dataset-specific OpenPI smoke check remains available for a real local
LeRobot dataset:

```bash
python -m training.validation.openpi_smoke \
  --dataset-dir /path/to/lerobot/local/xarm_pi05_data \
  --repo-id local/xarm_pi05_data \
  --output-json /tmp/xarm_openpi_smoke.json
```

## Training delegation

Only single-LeRobot configs supported by the vendored OpenPI revision can be
delegated directly:

```bash
python -m training.cli train pi05_xarm \
  --exp-name xarm_pi05_run \
  --assets-base-dir /work/assets \
  --checkpoint-base-dir /work/checkpoints \
  --execute
```

`--execute` is mandatory. The wrapper writes
`project_resolved_config.json` beside the run and then calls OpenPI's existing
training loop; it does not implement an optimizer.

Historical A/B/C configs are inspectable and their samplers are tested, but
the vendored OpenPI commit has no multi-LeRobot loader. Their original focused
loader existed only as untracked code in an external OpenPI checkout. The CLI
therefore refuses to launch A/B/C rather than flattening datasets or changing
sampling semantics. Restoring an execution-capable multi-LeRobot bridge needs
the original validated implementation (or a separately validated replacement)
and is recorded as technical debt.

## Normalization and actions

The stored dataset remains the `data.common` contract: two RGB uint8 images,
7D absolute state, 7D next-frame absolute action, and canonical task text.
OpenPI converts joint action dimensions 0-5 to deltas relative to the current
state, leaves gripper dimension 6 absolute in the raw controller convention,
and pads state/actions to the Pi0.5 action dimension of 32. Pi0.5 uses quantile
normalization.

Configs say whether statistics must be computed from exactly the declared
dataset set, loaded from a named precomputed asset, or preserved from a resume
checkpoint. The wrapper never silently recomputes or substitutes statistics.
