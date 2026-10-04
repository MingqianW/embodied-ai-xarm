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

For inference of the public `MingqianW/xarm-pi05-20260703` export, the explicit
`training.openpi.inference` boundary resolves `pi05_xarm_hf_20260703`. It uses
the operator-supplied original `pi05_xarm` snippet's model/transform fields,
represented by the existing historical registry variant, and explicitly binds
its template normalization identity to `local/xarm_pi05_20260703` inside the
checkpoint. This does not identify it as the later Delta run of the same name,
establish its actual training step/task coverage, or launch training. Other
evaluation config names continue through the upstream registry. See the
[FarmShare deployment guide](../commands/farmshare_sim_evaluation.md) for
restoration and real inference verification status.

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

## Bounded real/sim feasibility configuration

`pi05_xarm_real_sim_feasibility_10steps` is an independent full Pi0.5 experiment
from the declared `gs://openpi-assets/checkpoints/pi05_base/params`, with fresh
optimizer/EMA/step, batch 4, FSDP across 4 devices, and at most 10 updates. It
disables W&B and saves the final upstream checkpoint (manager directory `9`,
training state step `10`). It does not continue the evaluated HF fine-tune.

The real source is fixed at dataset revision
`1bbb721baef1e152515a33a50fa4552686877ce1`: 158 training episodes and 40
task-stratified held-out episodes. The canonical paired-v1 pool has 46 training
members from scene index 0 and 46 held-out members from scene index 1. Explicit
`EpisodeSelection.episode_indices` retain original IDs; absent/empty selections
fail. Real scene group IDs are unavailable, so physical-scene independence of
the real split cannot be asserted. LeRobot `task_index` plus its task catalog is
accepted by preflight and converted to prompt by the existing loader.

The domain draw uses `P(sim)=0.9`, `P(real)=0.1`, then uniform valid training
frames within that domain, with replacement and independent of dataset sizes.
This follows the mixture definition in [the paper's section III-A and appendix
VIII-G](https://arxiv.org/html/2503.24361v2); 0.9 is a candidate for this xArm
smoke, not a demonstrated optimum. Uniform frames do not imply uniform tasks.

The fresh train-only quantile asset records revision, explicit selections,
metadata hashes and selected parquet hashes in normalization manifest v2.
State anchors enter statistics once; their transformed 10-step action chunks
use upstream RunningStats semantics. Domain exposure weights do not reweight
this physical-pool statistic. A bounded `max_frames` computation is recorded
and cannot be silently reused as a full-pool computation.

The existing cluster `training` workflow now accepts optional `dataset_paths`
as semicolon-separated `ID=PATH` values, passing each as `--dataset-path`.
Keep DataLoader-delivered counts separate from optimizer-used samples: upstream
fetches the next batch after the final update. No feasibility or real-robot
performance claim follows from configuration alone; actual execution evidence
is required.

## Mixed training from the user-supplied pi05_xarm config

`pi05_xarm_real_sim_alpha09` derives its model, base-weight initialization and
optimization settings from the original user-supplied `pi05_xarm` snippet:
batch 16, 20,001 updates, save interval 5,000, AdamW clip 1.0, EMA 0.999,
W&B enabled and upstream-default cosine LR (warmup 1,000, peak `2.5e-5`,
decay steps 30,000, final `2.5e-6`). The snippet's commented `5e-5` schedule
is not active. FSDP remains the original default of one device.

The registered historical unsuffixed `pi05_xarm` is a different audited run
with 30,001 updates and save interval 10,000; its identity is preserved.
The mixed experiment reuses the pinned real158/sim46 training selections and
domain probabilities real0.1/sim0.9, with a new train-only normalization asset
`xarm_pi05_real_sim_alpha09_trainonly_v1`. Dataset size does not set the ratio.
No additional scene-group annotation is required for this experiment.

Inspect it with `python -m training.cli show pi05_xarm_real_sim_alpha09`.
This is a prepared full-training configuration, not an executed run. The
completed ten-update smoke and its artifacts are unchanged. GPU allocation
must be reviewed for batch16 and the chosen device topology; the earlier
four-L40S smoke resource request is not a measurement of this configuration.
Final training publication to Hugging Face includes the matching checkpoint,
normalization, resolved configuration and provenance.

The [FarmShare feasibility report](FARMSHARE_COTRAINING_FEASIBILITY.md) records
the completed 2026-10-04 run: accepted 92-episode generation, independent
train-only statistics, verified alpha sampling, ten actual full-model updates,
checkpoint save/restore, measured resources, estimates and remaining limits.
