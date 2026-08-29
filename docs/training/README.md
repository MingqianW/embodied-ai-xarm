# Training architecture

## Ownership

`training/` consumes canonical LeRobot outputs produced by `data.real` and
`data.sim`. It does not import collection, simulation, evaluation, or
diagnostics implementations. `data.common` remains the sole owner of field
names, dimensions, ordering, units, prompts, and record validation.

The pipeline is:

```text
DatasetSpec(s) + EpisodeSelection
                |
                v
        DatasetSet (physical composition)
                +
        MixingStrategy (training exposure)
                |
                v
       xArm OpenPI data adapter
                |
                v
   upstream OpenPI optimizer/checkpoints
```

Core responsibilities:

- `datasets/`: immutable repository/path/source/task/episode identities and a
  provenance-stripping OpenPI fixture adapter.
- `mixing/`: deterministic source schedules and whole-trajectory shuffling.
- `normalization.py`: selection of computed, precomputed, or preserved
  normalization assets. OpenPI still computes/applies the statistics.
- `configs/`: resolved model, data, checkpoint, optimizer, LR, EMA, and output
  interval semantics.
- `openpi/`: the small xArm transform/config bridge and lazy upstream import.
- `validation/`: local metadata checks, synthetic sampler gates, OpenPI config
  construction, and the real LeRobot batch smoke test.
- `cli.py`: inspection/preflight and explicit delegation to OpenPI.

## Four independent data variables

These values are never treated as synonyms:

1. Dataset composition is the count and selection of stored trajectories in
   each `DatasetSpec`.
2. Sampling ratio is the frequency of source identities in the sample stream.
3. Per-batch composition enforces source counts separately in every batch.
4. Global trajectory shuffle permutes whole episodes, preserves frame order,
   and accepts the natural source composition of the physical pool.

Independent source cyclers use a stable seed and reshuffle each source at
exhaustion. The 1:10 strategy is a repeated sample-level schedule; with batch
size 16, individual batches naturally have different counts. It is not
misrepresented as an impossible exact 1:10 per-batch split.

## Historical experiment registry

| Name | Physical datasets | Training exposure | Normalization |
|---|---|---|---|
| `pi05_xarm_full_finetune` | 50-episode v1 real | real-only | compute for dataset |
| `pi05_xarm_legacy_snippet_20001` | generic historical real template | real-only | compute for dataset |
| `pi05_xarm_v2_warm_start_20260703` | 150-episode five-task v2 real | real-only parameter warm-start | named v2 dataset asset |
| `pi05_xarm` | latest audited 198-episode six-task real | real-only base initialization | named 20260703 dataset asset |
| `pi05_xarm_colab_smoke` | historical real | real-only LoRA smoke | compute for dataset |
| `pi05_xarm_real50_sim50_stratified` (A) | real + stable-v3 sim | exactly 8 real + 8 sim per batch | `xarm_pi05_real_v3sim_1x` |
| `pi05_xarm_real1_sim10_stratified` (B) | real + 10x stable-v4 sim | repeated 1-real/10-sim sample schedule | `xarm_pi05_real_v4sim_10x` |
| `pi05_xarm_full_real_full_sim_trajectory_shuffle` (C) | same pool as B | global trajectory shuffle; no source quota | `xarm_pi05_real_v4sim_10x` |
| `pi05_xarm_d_simonly_v3_1x` (D) | stable-v3 sim | sim-only | fresh sim-only asset identity |
| `pi05_xarm_real_sim_50_50_continue` | 198 real + first 198 successful sim episodes | exactly 8 real + 8 sim per batch; true state resume | preserve checkpoint real-data asset |

A/B/C checkpoint identities and data/mixing/normalization semantics are
authoritative in the evaluation model specs. Their original external OpenPI
source was not committed, so the repository does not claim byte-level config
equivalence or silently invent a runnable multi-dataset bridge. The resolved
config records this evidence limitation and preflight reports it.
Compatibility-default optimization fields are visibly listed as unverified and
cannot make these configs launch-ready. D is likewise execution-disabled
because its tracked model spec does not contain the original optimizer config.

The name `pi05_xarm` was historically reused. The registry does not conflate
the old 20,001-step snippet, the tracker-recorded 150-episode warm-start, and
the later audited 198-episode Delta config; each has a separate identity above,
and the unsuffixed name denotes the latest audited configuration.

## Preserved model and optimization semantics

The xArm base model is Pi0.5 with `action_dim=32`, `action_horizon=10`, and
`discrete_state_input=false`. Base initialization uses
`gs://openpi-assets/checkpoints/pi05_base/params`. The historical standard
optimizer is AdamW (`b1=.9`, `b2=.95`, `eps=1e-8`, weight decay `1e-10`, global
norm clip `1.0`) with a 1,000-step warmup and cosine schedule from `2.5e-5` to
`2.5e-6` over 30,000 steps. Configs explicitly retain their EMA and step/save
differences; the LoRA smoke schedule and no-EMA behavior remain distinct.

Checkpoint modes distinguish base-weight initialization, parameter-only warm
start, and full state resume. A resume definition must restore optimizer, EMA,
and scalar step together. The prior 30,000-to-50,000 continuation evidence is
kept under `docs/experiments/migrations/training/`; it is historical evidence,
not a generic current launcher.

## Outputs and cluster boundary

OpenPI owns checkpoint contents, optimizer state, metrics, and W&B behavior.
The project wrapper writes the resolved project config into the run directory.
Root ignore rules exclude checkpoints, logs, W&B state, and generated data.
Slurm files remain under `slurm/` for Phase 7; Phase 6 changed only stale code
paths required by the reorganization and added no cluster architecture.
