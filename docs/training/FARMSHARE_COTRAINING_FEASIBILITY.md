# FarmShare xArm π0.5 mixed-data feasibility evidence

Verified on 2026-10-04, on branch `deployment/farmshare-sim-evaluation`.
Scientific generation/training source: `30f3859c275e7f8f4d13affadece371749c73caa`.
External, unchanged OpenPI: `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
Use the [training guide](openpi_finetuning.md), the
[paired generation runbook](../simulation_data/PAIRED_TRAJECTORY_V1_RUNBOOK.md),
and [FarmShare templates](../../cluster/farmshare/README.md).

The bounded experiment completed real backward passes and ten optimizer
updates, then saved and restored the full model/Adam/EMA/step state. This
establishes execution feasibility for this configuration; it does not establish
policy quality, a suitable full-training schedule, or real-robot improvement.
No full training, large generation, alpha sweep, or robot execution was run.

## Frozen evaluation baseline and independent starting point

The preceding evaluation completed before this experiment began: fixed HF model
`MingqianW/xarm-pi05-20260703` revision
`c308df02100a3f2b67512cc7ae61e28f16ee9263`, 120 valid episodes, zero invalid,
five successes. Its scientific source was `4d48a779c2db11317526af69daac63d94cb76531`.
The baseline inventory's 172 result/video files retained identical SHA256,
size and modification time; the original archive checksums also passed in CPU
job **1776771**. Baseline weights and normalization were not training inputs.

Experiment `pi05_xarm_real_sim_feasibility_10steps` starts from the declared
`gs://openpi-assets/checkpoints/pi05_base/params`. Twenty downloaded objects
(12,441,721,931 bytes) were pinned by GCS object generations and verified using
CRC32C, available MD5, and SHA256. Actual restoration used this base cache.
Initial step and both optimizer counters were zero; all 6,706,867,744 floating
optimizer entries were finite and zero, and every EMA value equalled its initial
parameter. The full model contains 3,353,433,872 parameters; no LoRA or freeze
filter was added. Model settings remain π0.5, 32 internal action dimensions,
10-step horizon, and `discrete_state_input=False`.

## Data, generation repair and splitting

Real data: `MingqianW/xarm_pi05_20260703`, fixed revision
`1bbb721baef1e152515a33a50fa4552686877ce1`. All 204 files/15,064,162,091 bytes
passed pinned integrity checks. CPU job **1776373** audited all 198 episodes,
22,618 frames and 45,236 decoded RGB images. Both RGB streams are 480×640;
state/actions are finite 7D values, timestamps are 10 Hz, and task-index/catalog
text resolves to exact canonical prompts. Every nonterminal action equals the
next observed state. The final action has no following recorded observation
and can only be checked for schema/finite values. Gripper values 211–843 fit
the canonical SDK-raw contract; source metadata does not independently document
their physical units.

Real split uses task-stratified SHA256 episode ordering with seed 20261004:
158 train episodes/18,075 frames and 40 held-out episodes/4,543 frames.
The explicit episode IDs are part of the experiment and normalization manifest.
The source provides no scene-group identifiers; physical-scene independence of
the real split remains unverified. Do not describe it as a scene-disjoint split.

The canonical bounded paired plan produced **92 accepted episodes**, **12 complete
scene groups**, **9,434 frames**, six tasks, and zero failed attempts in job
**1776439**. Conversion **1776572** and all-image raw/converted auditing
**1776731** passed with `READY_FOR_TRAINING`. Both generation and evaluation
resolve `clean_wide_v4` through the canonical simulation stack.
For every task, scene 0 is training and scene 1 is validation: 46 episodes each,
4,881 training frames and 4,553 held-out frames. Scene groups and generated
scene/trajectory seeds are disjoint; the saved common layout fingerprints and
seeds do not overlap the frozen 120-episode evaluation.

An earlier bounded run, **1776368**, exhausted all four retries for one Place
member: initialization contact/height/drift checks passed, but post-release speed
was 0.012144 m/s, exceeding the unchanged 0.01 m/s limit. The complete rejected
run, including its 91 accepted episodes and 12 failed attempts, was preserved.
The task-owned `direct_place` generator v2 changes only its oracle release
height from 0.125 m to 0.095 m. All four originally exhausted trajectories
pass the original checks in focused GPU regression **1776431**. Reset, assets,
four-bar gripper, collision geometry, randomization ranges, seeds, retry limit,
acceptance and evaluation scoring were not changed. Pepper remains a free body
with contact-based holding/release and no weld, body swap or repositioning aid.

## Normalization and sampling

CPU job **1776731** computed new statistics over all **22,956 training anchors**
using canonical normalization. Each selected anchor contributes once, with its
10-step action chunk processed by upstream statistics semantics. Held-out
episodes do not enter statistics. This physical pool is approximately 78.74%
real and 21.26% sim by anchor count; it is distinct from training exposure.

Asset: `xarm_real_sim_feasibility_20261004_trainonly_v1`.
`norm_stats.json` SHA256:
`770833694cd1ced59561db61a497fcefc090c4966980742f584c765a1387c086`.
The independent v2 manifest records revisions, selected IDs, metadata/parquet
hashes and `max_frames=null`. The saved checkpoint contains byte-identical norm
statistics; its associated manifest is retained with project metadata. A new
data composition requires fresh statistics and its own traceability manifest.

[Paper §III-A and appendix VIII-G](https://arxiv.org/html/2503.24361v2) guide
the candidate domain probability: `P(sim)=0.9`, `P(real)=0.1`, followed by
uniform sampling over valid selected frames within the chosen domain, with
replacement. Dataset sizes do not set alpha. The appendix's 1:9 example is
task/model-specific evidence, not proof of an xArm optimum. The implementation
uses existing `training.mixing` and the mixed loader; upstream π0.5 model, loss,
optimizer and loop remain unchanged.

The deterministic 100,000-draw audit observed 90,012 sim and 9,988 real draws;
within-domain ordinal bins passed the uniformity check. The actual ten updates
used 36 sim/4 real frames. The upstream loop also fetched the next unused batch,
so DataLoader-delivered totals were 40 sim/4 real (44 total), not 40 updates.

| Canonical task | Real training frames | Sim training frames | Optimizer samples |
| --- | ---: | ---: | ---: |
| pick up the largest block | 2,545 | 914 | 9 |
| pick up the blue block | 2,617 | 844 | 6 |
| pick up the red block | 2,732 | 827 | 7 |
| pick up the red pepper | 3,717 | 901 | 9 |
| pick up the smallest block | 2,743 | 894 | 6 |
| place the red pepper in the ring | 3,721 | 501 | 3 |

Real and sim OpenPI interface checks **1776442/1776731** verified two active
RGB views plus the masked empty third slot, 224×224 preprocessing, 7D→32D zero
padding, six joint deltas relative to observation state, absolute raw gripper,
10-step actions, finite normalized batches and canonical prompts.

## Actual training and resources

Training job **1776761** completed on oat-04: **4 L40S, 48 CPU, 180 GiB RAM**,
batch **4**, upstream FSDP **4**, workers **0**, EMA **0.999**, ten updates and
2-hour requested limit. Wall time, including extensive state evidence audits,
initialization, compilation, save and restore, was **9m38s**. Slurm sampled
MaxRSS was **151,651,596 KiB (~144.63 GiB)** for the batch process.

All ten losses/gradient norms were finite with nonzero gradients. Selected
action/time parameter fingerprints changed. The saved manager directory is
`9`, containing actual TrainState step **10**, both optimizer counters **10**,
nonzero finite moments, and EMA. The unchanged upstream restoration API loaded
the full checkpoint on the same four-device topology; selected model/EMA
fingerprints, optimizer audit, step and normalization matched. Restoration
added **zero optimizer updates**. An interrupted training-loop continuation or
a different device topology was not tested.

Synchronized training computation: first cold step **64.807 s**; nine subsequent
steps **0.8125–0.8137 s**, median **0.8130 s**. These timings exclude fetching the
next batch and checkpoint/evidence work. Per-device JAX allocator peak live
bytes were **26,989,086,208 (~25.14 GiB)**. One-second NVML samples peaked at
**33,519 MiB (~32.73 GiB)** on GPU 0 and **33,503 MiB** on the other devices.
NVML includes retained allocator/context memory and is a sampled peak; it is
not the same measurement as peak live JAX buffers.

At the same batch/configuration, 20,000/30,000 steps require approximately
**4.52/6.78 hours of steady computation**, or **18.1/27.1 GPU-hours**. A planning
allowance of 25–50% yields roughly **5.65–6.78 / 8.47–10.16 hours**, before queue
time and initial setup; that allowance is an assumption, not a long-run
measurement. Checkpoint IO, input stalls, validation and long-run variability
need separate budgeting. Batch 16 or larger configurations were not measured.
Batch 4 has fewer examples per step than the historical batch-16 configuration;
the same number of steps is not equivalent data exposure or a validated schedule.

Model+Adam+EMA arrays contain approximately **53.65 GB** before storage encoding.
The actual saved checkpoint occupies **44,769,077,565 bytes (~44.77 GB)**;
total measured experiment payload is approximately **91.63 GB (~85.33 GiB)**,
including the dataset/base/Arrow caches and preserved rejected-run evidence.
Account for real parquet and Arrow cache, base weights, 1.29 GB raw/0.79 GB
converted sim data, rejected-run evidence and each retained checkpoint. Every
additional retained full-state checkpoint can cost another approximately 54 GB.
Verify personal quota and preservation requirements before any full run.

## Evidence and execution boundaries

Runtime products are outside source packages. The control root is
`~/xarm-work/cotraining-feasibility-20261004`; the payload root is
`/scratch/users/$USER/cotraining-feasibility-20261004`, accessible through
the FarmShare personal scratch alias. Important control evidence:
`training-observed-1776761.json`, `optimizer-step-ledger.json`,
`training-gates.json`, `mixing-audit.json`, `split-leakage-audit.json`,
`real-full-frame-audit.json`, `preservation-audit.json`,
`feasibility-resources.json`, `checkpoint-integrity.json`, and
`integrity-generated_raw.json` / `integrity-converted_sim.json`.
`evidence/place-release-human-review.mp4` is a diagnostic rendering, not a
recorded human acceptance. The private collector scripts and logs are preserved
as runtime evidence; they observe the canonical CLI and upstream functions.

The canonical launch uses site-reviewed resources and explicit dataset roots:

```bash
python -m cluster.cli submit training \
  --param config=pi05_xarm_real_sim_feasibility_10steps \
  --param exp_name=NEW_INDEPENDENT_EXPERIMENT_NAME \
  --param "dataset_paths=real_hf_20260703_train=$REAL_DATA;sim_paired_v1_train=$SIM_DATA"
```

This command creates an independent experiment and performs ten updates; it was
already executed for this authorization. It is documentation, not authorization
to repeat it or exceed the cumulative ten-update limit. For independent later
experiments use fresh optimizer/EMA/step and base weights, and recompute
normalization when composition changes. Restore the matching checkpoint stats
for inference rather than any previously evaluated model's stats.

Official FarmShare [policy](https://docs.farmshare.stanford.edu/policy/),
[Slurm](https://docs.farmshare.stanford.edu/slurm/) and
[transfer guidance](https://docs.farmshare.stanford.edu/transfer/) were consulted
on 2026-10-04. Eligibility was explicitly confirmed as unsponsored research
with a full-service SUNet ID. Compute, rendering, downloads, full-frame audits,
normalization and training ran in allocations. Requests used account `operator`
and verified normal/normal or gpu/gpu partition/QoS. The CPU/RAM limit adjusted
8 CPU/32 GiB generation/audit requests to 10 allocated CPUs; the training
48 CPU/180 GiB request was unchanged. The observed GPU QoS permits four GPUs
per user and partition maximum wall time is two days.

A home quota failure during download required relocating only this experiment's
payload to personal scratch and revalidating all bytes; two damaged parquet
files were replaced from the pinned HF revision. Global filesystem free space
does not prove personal quota, retention or backup. Neither scratch retention
nor an independent backup has been established. CPU focused checks passed
**76 tests**; the four physical Place regressions ran separately in GPU Slurm.

The next useful work would establish real-data scene grouping, enlarge the
number of independently held-out sim scenes, and test a suitable batch size,
training schedule and validation metrics before a full run or alpha comparison.
Those experiments require separate authorization. Any eventual real-robot
performance claim requires actual authorized hardware evaluation.
