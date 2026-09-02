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
  --dataset-path real_xarm_pi05_20260703=/path/to/real_dataset
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

The command resolves every declared source independently and then delegates to
OpenPI's existing model, transforms, optimizer, distributed execution, and
checkpoint loop. The repository supplies only the deterministic named-source
loader. A/B/C therefore remain configuration-only experiments:

| Config | Mixing policy |
| --- | --- |
| `pi05_xarm_real50_sim50_stratified` | exactly 8 real + 8 sim frames in each global batch |
| `pi05_xarm_real1_sim10_stratified` | deterministic 1:10 weighted sample stream |
| `pi05_xarm_full_real_full_sim_trajectory_shuffle` | deterministic globally shuffled whole trajectories |

For example, launch A with separate physical dataset roots:

```bash
python -m training.cli preflight pi05_xarm_real50_sim50_stratified \
  --dataset-path real_xarm_pi05_20260703=/datasets/real \
  --dataset-path sim_mujoco_stable_v3_1x=/datasets/sim_v3

python -m training.cli train pi05_xarm_real50_sim50_stratified \
  --exp-name xarm_real50_sim50 \
  --dataset-path real_xarm_pi05_20260703=/datasets/real \
  --dataset-path sim_mujoco_stable_v3_1x=/datasets/sim_v3 \
  --assets-base-dir /work/assets \
  --checkpoint-base-dir /work/checkpoints \
  --execute
```

`--execute` is mandatory. `--dataset-path ID=PATH` can be repeated in any
order; no source is inferred from another source's parent directory.
`project_resolved_config.json` and the resolved paths are recorded under
`CHECKPOINT_BASE/_project_metadata/`, so they never pre-create OpenPI's run
directory. To replace a previously computed normalization asset intentionally,
add `--recompute-norm`.

## Normalization and actions

The stored dataset remains the `data.common` contract: two RGB uint8 images,
7D absolute state, 7D next-frame absolute action, and canonical task text.
OpenPI converts joint action dimensions 0-5 to deltas relative to the current
state, leaves gripper dimension 6 absolute in the raw controller convention,
and pads state/actions to the Pi0.5 action dimension of 32. Pi0.5 uses quantile
normalization.

For `compute_from_datasets`, the bridge computes statistics once over each
selected physical frame in the declared pool, rather than replaying a
ratio-biased training stream. It writes a manifest alongside the OpenPI asset;
an existing asset is reused only when that manifest matches the selected paths,
metadata hashes, episode selection, and state/action semantics. Precomputed
and resume-checkpoint assets are never silently replaced.
