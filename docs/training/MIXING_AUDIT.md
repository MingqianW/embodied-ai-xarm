# Mixed training data audit

## Scope and conclusion

This audit covers the project-owned path from `ExperimentConfig` through
`training.openpi.mixed_loader` to the upstream OpenPI DataLoader. Dataset
definition and training exposure are separate:

- `DatasetSpec` and its `EpisodeSelection` select the physical trajectories
  that exist in an experiment. `EpisodeSelection.limit` is a dataset-size
  control; it is never derived from a mixing strategy.
- `MixingStrategy` selects source identities only after those physical pools
  have been built.

The old bridge did not truncate or duplicate data to implement A/B mixing.
The refactor makes this rule explicit for new weighted experiments, adds
float-weight normalization, startup/consumption reporting, and regression
coverage.

## Behavior before this change

| Configuration | Physical dataset selection | Training exposure |
| --- | --- | --- |
| A: `pi05_xarm_real50_sim50_stratified` | full real plus stable-v3 sim pools | exactly 8 real and 8 sim positions in every global batch |
| B: `pi05_xarm_real1_sim10_stratified` | full real plus stable-v4 10x sim pools | legacy deterministic schedule of one real then ten sim positions |
| C: `pi05_xarm_full_real_full_sim_trajectory_shuffle` | same physical pool as B | whole trajectories shuffled globally; source frequency follows physical frame counts |

`_load_datasets` constructs one `DatasetFrameRef` per selected stored frame.
`_MixedStreamDataset` is a virtual random-access dataset; its length is based
on requested optimizer steps and batch size, not on the sum of source lengths.
`TorchDataLoader` receives `shuffle=False`, because the project stream owns
the ordering. Its distributed sampler partitions that virtual stream; it does
not rebuild data according to source size.

The only intentional subset in the checked-in registry is the historical
`SIM_RED_BLOCK_EP198` `EpisodeSelection("first_by_episode_index", 198, ...)`.
That is a physical episode limit, not a 1:1 mixing mechanism.

The old integer `weighted_sample_stream` represented weights as a repeating
source-label schedule. It was size-independent but did not provide a clear
float-weight API, normalized probabilities, or runtime observed-fraction
reporting. The historic names remain because they identify existing model and
checkpoint evidence; they are not recommended for new experiments.

## Current API

New real/simulation experiments should use:

```python
from training.mixing.strategies import MixingStrategy

mixing = MixingStrategy.real_sim_weighted_sampling(
    real_sampling_weight=1.0,
    sim_sampling_weight=10.0,
)
```

This is `weighted_domain_sampling`. Its probabilities are:

```text
P(real) = real_sampling_weight / (real_sampling_weight + sim_sampling_weight)
P(sim)  = sim_sampling_weight  / (real_sampling_weight + sim_sampling_weight)
```

Weights may be integer or floating point. `1, 10` and `0.090909, 0.909091`
therefore express the same approximately 1:10 domain probability. After a
domain is selected, a frame is selected virtually from that domain's selected
physical pool. The implementation keeps each pool once and does not create a
repeated concatenated dataset.

For a deterministic composition requirement, use:

```python
MixingStrategy.fixed_batch_composition({"real": 8, "sim": 8})
```

The counts must add exactly to the global batch size. This is deliberately a
different strategy from weighted domain sampling. `per_batch()` and
`sample_ratio()` remain compatibility helpers and issue `DeprecationWarning`;
historical A/B registry entries preserve their original semantics.

## Dataset-size controls

The registry is Python dataclasses, not a YAML training-config parser. A
physical subset is declared on the relevant `DatasetSpec`, for example:

```python
DatasetSpec(
    "sim_subset",
    "local/xarm_sim",
    "sim",
    ALL_TASKS,
    selection=EpisodeSelection("first_by_episode_index", 5_000, "sim_max_episodes"),
)
```

`EpisodeSelection.limit` is the current explicit equivalent of a
`sim_max_episodes` or `real_max_episodes` field. It must be changed only when
the desired number of unique usable episodes changes. It is independent of
`real_sampling_weight` and `sim_sampling_weight`.

## Normalization

For `compute_from_datasets`, normalization uses the selected physical pool:
every selected frame from every declared dataset appears once. It does **not**
replay the training sampling stream, so sampling weights do not oversample a
small domain in normalization statistics. The consequence is explicit:
normalization is weighted by selected physical frame counts, not by requested
training exposure. `mixed_normalization_manifest.json` records this as
`uniform_selected_physical_frames` and records the selected datasets and
episode-selection metadata.

Precomputed and checkpoint-preserved normalization assets are unchanged.

## Step and observation semantics

Training remains step-based. The virtual stream reserves
`(num_train_steps + 2) * batch_size` positions when OpenPI does not explicitly
provide a finite batch count; the two extra batches are loader headroom.
Neither that virtual length nor optimizer-step count depends on combined real
+ simulation data length. There is no source-size-defined epoch.

At loader startup, the bridge prints selected episode/frame counts by source,
strategy, requested weighted probabilities (where applicable), normalization
semantics, global batch size, and total optimizer steps. It also keeps
cross-worker counters of DataLoader-delivered real/sim samples, logs them at
the OpenPI log interval, and prints final observed fractions when training
returns or fails. Prefetch means these are DataLoader-delivered counts; they
are the closest project-owned measurement without replacing the upstream
optimizer loop.

## Example experiments

Assume a real `DatasetSpec` selecting 100 episodes and a sim `DatasetSpec`
selecting 1,000 episodes:

```python
# Same physical 100 + 1,000 episode pools; approximately 50% real exposure.
MixingStrategy.real_sim_weighted_sampling(
    real_sampling_weight=1.0, sim_sampling_weight=1.0
)

# Same physical 100 + 1,000 episode pools; approximately 1/11 real exposure.
MixingStrategy.real_sim_weighted_sampling(
    real_sampling_weight=1.0, sim_sampling_weight=10.0
)

# Change only the physical sim selection from 1,000 to 5,000 episodes.
# Keep the same 1:10 mixing object, so the source probability is unchanged.
from dataclasses import replace
sim_5k = replace(
    sim,
    selection=EpisodeSelection("first_by_episode_index", 5_000, "sim_max_episodes"),
)
```

`DatasetSpec` is frozen, so `replace` creates a separate physical dataset
definition. It does not alter the mixing object.
