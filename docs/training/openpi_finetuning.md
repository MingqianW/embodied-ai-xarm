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

## User-supplied historical configuration

On 2026-10-03 the user supplied the training-time xArm configuration snippet.
Its `XARM_CONFIG_SNIPPET` payload exactly matches the existing
[`legacy_openpi_xarm_config_snippet.py`](../../training/openpi/legacy_openpi_xarm_config_snippet.py).
The payload is preserved without adding a second configuration implementation.
Its SHA-256, computed over the UTF-8 string value, is
`d7c9f396dc7b0532500db6c5ec80b5ce3b7df1b19f9dd464cc3b04cc5e87477d`.

Inspect the original snippet from the repository root without importing OpenPI
or loading a checkpoint:

```bash
python -m training.openpi.legacy_openpi_xarm_config_snippet
```

The supplied source contains `pi05_xarm_full_finetune`, `pi05_xarm`, and
`pi05_xarm_colab_smoke`. Its historical `pi05_xarm` entry uses the dataset
placeholder `local/xarm_pi05_data`, 20,001 training steps, a 5,000-step save
interval, and EMA 0.999. The matching project registry identity is
`pi05_xarm_legacy_snippet_20001`; the current unsuffixed `pi05_xarm` still means
the separately audited 198-episode configuration. This source confirmation
does not change either registry entry.

For checkpoint inference, retain the supplied model and action-transform
semantics, and resolve the normalization asset from the actual checkpoint.
The HF `MingqianW/xarm-pi05-20260703` asset is
`local/xarm_pi05_20260703`, which differs from the snippet's dataset placeholder.
Do not rename checkpoint assets or recompute normalization to match that
placeholder. The snippet is historical evidence, not a registered upstream
config or a validated policy-server launch command; checkpoint restore and
server compatibility still require validation in the deployment environment.

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

For a new experiment, do not encode a real:sim sampling target by limiting or
repeating a dataset. Keep the desired physical episode selections on their
individual `DatasetSpec` objects, then use the explicit new API:

```python
MixingStrategy.real_sim_weighted_sampling(
    real_sampling_weight=1.0,
    sim_sampling_weight=10.0,
)
```

This means approximately 1/11 real and 10/11 sim loader samples, independent
of selected real/sim dataset sizes. For exact deterministic source counts in a
global batch, instead use `MixingStrategy.fixed_batch_composition(...)` with
counts that sum to the configured batch size. See
[`MIXING_AUDIT.md`](MIXING_AUDIT.md) for legacy A/B/C semantics, normalization
weighting, step semantics, and size-vs-ratio examples.

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

At training startup the loader prints selected episode/frame counts by source,
the mixing strategy and requested weighted probabilities, normalization
semantics, batch size, and total steps. It also reports observed real/sim
fractions from DataLoader-delivered samples at the log interval and at exit.
