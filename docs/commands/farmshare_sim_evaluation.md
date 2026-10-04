# FarmShare simulation evaluation deployment

## Verified deployment

This guide records the actual FarmShare deployment of
`MingqianW/xarm-pi05-20260703`, model ID `HF_REAL_20260703`. Scientific execution
uses source commit `4d48a779c2db11317526af69daac63d94cb76531` on the independent
`deployment/farmshare-sim-evaluation` branch. This guide is published afterward;
a newer documentation commit is not the source identity of existing results.
The confirmed starting source was `refactor/reorganize-repository` at
`924f4e5771d76f336e085aba3b90ca67cabc7b53`, not `main`.

Fixed-source smoke 1775501 and formal 1775518 passed independent acceptance;
actual matching-source resume 1775551 completed successfully. The CPU proof
verified identical SHA-256, size and mtime for all 120 raw result JSONs and
52 retained videos. All 120 formal episodes are valid, with no infrastructure
invalids. Raw/summary/per-task/macro/provenance/exact-prompt checks agree;
52 videos cover all 13 observed task/category combinations and pass frame-count
and endpoint decode. Smoke retained all 48 videos and has 12 valid/0 invalid,
0 success; its normal failures were 9 NoLift,1 PartialLift and 2 PlaceOutsideRing.

| Stage | Slurm job | Actual completion | Allocation |
| --- | --- | --- | --- |
| Fixed-source 12 smoke | 1775501 | COMPLETED 0:0,3 m 54 s,oat-02 | 16 CPU,60 G,1 GPU,45 min limit |
| Smoke acceptance | 1775515 | COMPLETED 0:0,4 s,wheat-03 | 4 CPU,8 G,one 2 CPU task,0 GPU |
| Fixed-source 120 formal | 1775518 | COMPLETED 0:0,31 m 39 s,oat-02 | 16 CPU,60 G,1 GPU,1 h limit |
| Formal acceptance/before snapshot | 1775549 | COMPLETED 0:0,6 s,wheat-03 | 4 CPU,8 G,one 2 CPU task,0 GPU |
| Actual guarded resume | 1775551 | COMPLETED 0:0,43 s,oat-02 | 16 CPU,60 G,1 GPU,15 min limit |
| Unchanged-file proof/archive/readback | 1775552 | COMPLETED 0:0,24 s,wheat-03 | 4 CPU,8 G,one 2 CPU task,0 GPU |

All three GPU submissions had Requeue 0 successfully set immediately. Smoke,
formal and resume batch peak RSS were 19,160,404 K,18,781,512 K and 19,430,488 K.
Five-second GPU sampling observed 8824 MiB over 46 samples for smoke and 8764 MiB
over 379 samples for formal; these are sampled maxima, not absolute peaks.
The 1 h budget preserves the successful 16 CPU/60 G/one GPU resource shape and
covers both the earlier 29 m 10 formal and current 31 m 39 actual run.

| Canonical task | Success / valid | Success rate |
| --- | --- | --- |
| red_pepper (Pick) | 0/20 | 0% |
| blue_block (Pick) | 0/20 | 0% |
| red_block (Pick) | 0/20 | 0% |
| smallest_block (Pick) | 0/20 | 0% |
| largest_block (Pick) | 0/20 | 0% |
| place_red_pepper_in_ring | 5/20 | 25% |

Six-task macro, valid overall and all-attempt success rates are each 4.1667%
(5/120). Invalid count 0, invalid reasons{}; there are no unresolved
infrastructure-invalid episodes. Normal failures: 72 PICK_NO_MEANINGFUL_LIFT,
26 PICK_PARTIAL_LIFT,2 PICK_DROPPED_AFTER_LIFT,
14 PLACE_RELEASED_OUTSIDE_RING,1 PLACE_STABLE_BUT_NOT_SUSTAINED.
Original Place initialization validation passed on all 20 policy episodes;
no initialization anomaly was converted into a model task failure.

The six task prompts, canonical `clean_wide_v4` randomization, seeds, safety,
reset acceptance, scoring and action semantics remain unchanged. Formal seeds
are 50018–50037 for each task; smoke uses 50018–50019. Each prediction has a
10-step horizon, executes five actions, and allows at most 50 predictions.
Actions last 0.1 seconds; the physics timestep is 0.002 seconds. Target
protocols inherit the canonical v3 protocol and override only output roots.
A/B/C/D specifications and the original scientific protocols are preserved.

## Official policy and actual site

The operator confirmed unsponsored research and a full-service SUNet ID.
These official pages were read and archived on 2026-10-03
(America/Los_Angeles):

- [Introduction](https://docs.farmshare.stanford.edu/)
- [Policy](https://docs.farmshare.stanford.edu/policy/)
- [Connecting](https://docs.farmshare.stanford.edu/connecting/)
- [Slurm](https://docs.farmshare.stanford.edu/slurm/)
- [Software](https://docs.farmshare.stanford.edu/software/)
- [Transfer](https://docs.farmshare.stanford.edu/transfer/)

Authentication uses Stanford's normal mechanism. Model restoration, GPU
initialization, warm-up, inference, physics, rendering and substantial audits
run in Slurm allocations. The service and client share one allocated node and
localhost; no persistent model service runs on a rice login node.

Read-only audit commands actually exercised:

```bash
hostname -f
id
git status --short --branch
git submodule status
sinfo -o '%P %a %l %D %G'
scontrol show partition
scontrol show node oat-01
sacctmgr -n -P show assoc where user="$USER" \
  format=Cluster,Account,User,Partition,QOS,DefaultQOS
sacctmgr show qos \
  format=name%16,maxsubmitjobspu,maxjobspu,mintres%20,maxtrespu%45,maxwall
squeue -u "$USER" -o '%i %P %j %T %M %l %D %R'
module spider python
module spider cuda
df -h "$HOME" /scratch/users/"$USER"
```

The actual account was `operator`, Slurm 26.05.4, CPU partition/QoS
`normal/normal` and GPU `gpu/gpu`. Observed partitions also included `bigmem`;
the documentation's `interactive` example did not match this site. GPU limits
included two days, four GPUs per node, one-GPU minimum and four GPUs per user,
and maximum memory 4000 MB per CPU. Actual accounting matters: a two-CPU/8 G
request allocated four CPUs, while its explicit one-task step used two.
Standalone `srun` must specify `--nodes=1 --ntasks=1`; omitting task count
previously ran two read-only auditors and made one exclusive report write fail.

Available modules included Python 3.12.3 and CUDA 12.9.0, differing from older
software-page examples. No CUDA module was loaded merely to follow an example.
Allocated NVIDIA L40S reported 46,068 MiB and driver 595.91.07. Site inventory,
GPU snapshots and sampled maxima are distinct from absolute peak measurements.
Recheck resource/association/module facts for a later deployment.

Home and scratch are NFS. Filesystem available space is not user quota; no
quota executable or retention/backup guarantee was established. Archive copies
on this same home filesystem are verified snapshots, not independent backups.
Use the documented DTN/Globus mechanism for a separately authorized large
transfer; no transfer to an external destination was performed here.

## Versions and actual paths

| Item | Fixed identity or actual path |
| --- | --- |
| Scientific source | `4d48a779c2db11317526af69daac63d94cb76531` |
| Source checkout | `/home/users/mw27/projects/embodied-ai-xarm` |
| External unmodified OpenPI | `/home/users/mw27/projects/openpi`, `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| Runtime root | `/home/users/mw27/xarm-work/farmshare-20261003` |
| Model interpreter | `$XARM_WORK_ROOT/envs/openpi/bin/python`, Python 3.11.15 |
| Submission/offline audit interpreter | `$XARM_WORK_ROOT/envs/audit/bin/python`, Python 3.12.3 |
| HF revision | `c308df02100a3f2b67512cc7ae61e28f16ee9263` |
| Snapshot | `$XARM_WORK_ROOT/checkpoints/huggingface/xarm-pi05-20260703/c308df02100a3f2b67512cc7ae61e28f16ee9263` |
| Rebuildable caches | `$XARM_WORK_ROOT/caches/{openpi,jax,uv}` |
| Environment freeze | `$XARM_WORK_ROOT/openpi-environment-freeze.txt` |
| Smoke identity | `hf_real_20260703_resume_smoke_v3` |
| Formal identity | `hf_real_20260703_resume_formal_v3` |
| Results | `$XARM_WORK_ROOT/mujoco_outputs/policy_evaluation/IDENTITY` |
| Logs | `$XARM_WORK_ROOT/logs/{cluster,policy_service}` |
| Recovery records | `$XARM_WORK_ROOT/state.json`, `STAGES.md` |

Pinned runtime versions: JAX/JAXlib 0.5.3, Flax 0.10.2, Orbax-checkpoint
0.11.13, NumPy 1.26.4, ml-dtypes 0.4.1, TensorStore 0.1.74, PyTorch 2.7.1,
MuJoCo 3.3.7, OpenCV headless 4.11.0.86, huggingface-hub 0.32.3,
pytest 8.4.2 and uv 0.11.28. Installation job 1774160 completed 4 m 42 s and its
202-package dependency check passed. The frozen upstream lock includes MuJoCo
2.3.7 through unused Aloha dependencies; this scene needs MuJoCo >=3.2. The
existing CPU environment workflow exports upstream dependencies, omits the
unused gym-aloha/dm-control/mujoco packages, installs the pinned client, and
adds the project runtime versions. OpenPI source and lock remain unchanged;
imports use `OPENPI_ROOT/src`. The source submodule remains uninitialized at
the same pin; no `third_party/` contents were modified.

Model card: `Real-robot-only`. The checkpoint contents are at HF repository
root, with no proven numeric manager step. The model spec explicitly uses
`manager_step: null`. Snapshot job 1774189 verified all 23 files and read all 106
OCDBT keys/12,441,508,165 bytes. That integrity report does not claim model
restoration. Actual allocated restoration passed separately.

The operator supplied the original xArm config snippet and named `pi05_xarn`,
interpreted as its `pi05_xarm` entry. The explicit inference config
`pi05_xarm_hf_20260703` matches full Pi0.5, internal action dimension 32,
horizon 10, `discrete_state_input=false`, Libero inputs/outputs, six joint
DeltaActions/AbsoluteActions and absolute SDK raw gripper. Six joints are
radians; the legacy snippet's gripper_mm comment does not authorize a scale
conversion. Two RGB images pass 224×224 preprocessing; final actions are 10×7.
The target's own quantile normalization asset is
`local/xarm_pi05_20260703`, SHA-256
`63a3d8456b509600fde0e9a3546fa2c01145cb0c3527d46ead649b4c35b37fc4`.
No A/B/C normalization was substituted. Original manager/training step and
training task coverage remain unknown; config-name reuse does not prove a
connection to a later Delta training run.

## Changes and evidence

Scheduling stays in the existing `cluster.cli` and workflow/job owners.
[FarmShare resources](../../cluster/farmshare/resources.json) and
[environment template](../../cluster/farmshare/environment.example.sh) are
site templates; measured private overrides selected 16 CPU/60 G/one GPU,
45 min smoke,1 h formal and 15 min resume. Existing DeltaAI defaults and command
resolution remain covered by focused cluster regression tests.

The [model spec](../../configs/evaluation/sim/models/HF_REAL_20260703.json)
and inference adapter live in their canonical owners. Root-exported
checkpoint handling, nonexistent provenance code-path repair, discarded-video
path cleanup and strict parameter restore have focused regression evidence.
Service identity is derived only after actual strict loading and finite
warm-up. The thin [request RNG adapter](../../policy_runtime/openpi_request_rng.py)
strips transport metadata and passes seeded JAX Gaussian noise through pinned
OpenPI's public inference hook. GPU verification 1774739 restored twice, obtained
finite 10×7 actions, rejected missing seeds, changed actions for different seeds,
and verified bitwise same-seed equality after an intervening seed and restart.
No third-party service/source was patched. Allocation-local readiness is bounded;
owned subprocesses are terminated in cleanup on success or error.

Independent Place repair commit
`25f1582bbd7aafc7ec2e8e164143c76c0192bd38` changes canonical initialization
under the current four-bar/contact geometry. Diagnosis recorded zero contacts
for the old kinematic held pepper on all 20 seeds, and failure of a same-pose
free-body transfer. A finite model-free contact/height/TCP-drift/collision
probe selected raw 450 and one TCP-relative pose `[0,0,-0.040]`m. The existing
free pepper is positioned once after arm randomization, then undergoes the
existing 500 normal physics steps. It remains free throughout original grasp
validation and policy execution. There is no weld/fixed attachment, continuous
repositioning, observation override, raw-command lock or release teleport.
Opening preserves pose/momentum and the same body falls physically.

Initial arm pose, ring position, geometry, solver/friction, action semantics,
scoring, all acceptance thresholds, tasks/seeds and randomization ranges stay
unchanged. Both generation configurations and the canonical object declaration
resolve the same setting. Existing simulation and evaluation owners are reused.
Evidence `place-repair-invariants.json` records unchanged scientific-file hashes;
`place-physical-shared-summary.json` from 1775264 records zero compiled physics
differences for both v4/paired generation plans versus formal Place.

The no-assistance focused physical regression first checked 50018/50019,
including raw/observation behavior, unchanged pose/momentum on release and
physical fall/contact loss; all 20 formal seeds passed original grasp checks.
`evidence/place-repair-canonical-20/summary.json`: minimum 2 gripper contacts,
minimum 0.1052584 m height above table, maximum relative drift 0.0000631967 m,
no table or forbidden collision. Allocated rendering 1775233 passed 13 affected
regressions including 120 scene resets; one known inventory baseline failure
was deselected. Its two 3 s review clips show 2 s free-body contact hold then 1 s
opening/falling, with quantitative reports and all four cameras. These clips
use Place `25f1582` plus the then-uncommitted EGL delta later published 1 eb; physics
was unchanged. They are model-free previews, not learned-policy demonstrations.
No human judgment has been received.

EGL repair commit `1eb9a929ff6d0d6dd1e655a9d4b1f4b3eb488aa5` addresses a
Slurm global GPU index differing from EGL enumeration under remapping. The
canonical service matches the sole allocated visible CUDA GPU's hardware UUID
to EGL using the public [EGL_NV_device_cuda query](https://registry.khronos.org/EGL/extensions/NV/EGL_NV_device_cuda.txt),
without initializing displays on unallocated GPUs. Missing/ambiguous identity
fails before rendering. Actual UUID/index selections are recorded in load
reports; monitoring selects that hardware UUID. Fourteen service unit checks
and allocated rendering passed, including global index 2/3 versus EGL 0.

Strict JSON-resume repair commit
`4d48a779c2db11317526af69daac63d94cb76531` serializes protocol tasks as a
JSON list. Strict full provenance equality is retained. The filesystem
roundtrip test reproduced the prior tuple/list mismatch, then 34 focused
provenance/output-inheritance/cluster tests passed. Serialized scientific
values, RNG, model and physical settings are unchanged. The valid first 120
at source `1eb9a929` are preserved; changing source provenance requires new output
identities rather than mixing or rewriting old results.

## Reproduce the verified execution

Freeze source at the scientific commit above, with a clean checkout at the
same absolute source path. Use the same OpenPI revision, model snapshot,
normalization, environment and absolute result paths. Even a documentation
commit or moved path changes strict provenance; resume is allowed only when
all recorded identities match. Do not switch or edit source during a job.
For a later resume from the published guide branch, select the recorded source
in a clean checkout before submitting, for example:

```bash
git status --short
git switch --detach 4d48a779c2db11317526af69daac63d94cb76531
```

This checkout instruction is a prerequisite; the actual run itself used the
migration branch while its HEAD was that same scientific commit.
The private wrappers/resources below are archived runtime operators; wrappers
only enforce allocation/source identity and sample assigned-GPU memory.
They call canonical implementations rather than duplicating simulation or
scoring. The following are the actual submission command forms, with the
recorded runtime environment:

```bash
source /home/users/mw27/xarm-work/farmshare-20261003/audit-environment.sh
cd "$XARM_REPOSITORY"
export XARM_PYTHON="$XARM_WORK_ROOT/place-monitored-python"
export XARM_SLURM_RESOURCE_CONFIG="$XARM_WORK_ROOT/farmshare-resources.json"
"$XARM_WORK_ROOT/envs/audit/bin/python" -m cluster.cli submit formal-sim-evaluation \
  --param model_spec=configs/evaluation/sim/models/HF_REAL_20260703.json \
  --param protocol=configs/evaluation/sim/protocols/hf_real_20260703_resume_smoke_v3.json \
  --param host=127.0.0.1 --param port=18005 --param start_server=true

# Only after independent smoke acceptance:
export XARM_SLURM_RESOURCE_CONFIG="$XARM_WORK_ROOT/farmshare-formal-resources.json"
"$XARM_WORK_ROOT/envs/audit/bin/python" -m cluster.cli submit formal-sim-evaluation \
  --param model_spec=configs/evaluation/sim/models/HF_REAL_20260703.json \
  --param protocol=configs/evaluation/sim/protocols/hf_real_20260703_resume_formal_v3.json \
  --param host=127.0.0.1 --param port=18005 --param start_server=true

# Completed-result resume: same scientific source, inputs and output identity.
export XARM_SLURM_RESOURCE_CONFIG="$XARM_WORK_ROOT/farmshare-resume-resources.json"
"$XARM_WORK_ROOT/envs/audit/bin/python" -m cluster.cli submit formal-sim-evaluation \
  --param model_spec=configs/evaluation/sim/models/HF_REAL_20260703.json \
  --param protocol=configs/evaluation/sim/protocols/hf_real_20260703_resume_formal_v3.json \
  --param host=127.0.0.1 --param port=18005 --param start_server=true \
  --param resume=true
```

Capture each submission once and immediately run
`scontrol update JobId=SUBMITTED_ID Requeue=0`; query that ID with
`squeue`/`sacct` instead of submitting again. Preserve source throughout formal,
resume and its evidence archive. First-run commands above were executed on
fresh output roots. Those roots now contain results: repeating first-run
commands must fail rather than overwrite. Use the actual guarded resume form
for existing results. A separate fresh experiment needs a new output-only
inherited protocol and a new accepted-smoke gate; an arbitrary CLI output-root
change must not evade recorded protocol identity. A new source commit needs
its own output identity and validation; do not resume across versions.

The existing CPU setup/checkpoint/preflight commands were also exercised:
`cluster.cli submit openpi-inference-environment`,
`cluster.cli submit checkpoint-snapshot` and
`cluster.cli submit formal-sim-evaluation --param dry_run=true`. For CPU-only
work use `normal/normal` and a zero-GPU resource override; inspect resolved
commands before submitting. Actual configuration, exact invocation and evidence
are preserved in `logs/cluster/runs`, state and submission logs. Model
verification uses the same formal workflow with `verification_report` and an
exclusive new evidence filename. File presence/preflight never substitutes for
real restoration/inference.

Focused commands verified during this deployment include:

```bash
python -m pytest -q tests/evaluation_sim/test_provenance.py \
  tests/evaluation_sim/test_protocol_inheritance.py tests/cluster
# Physics/rendering tests below require the recorded Slurm allocations:
python -m pytest -q tests/evaluation_sim/test_place_physical_reset.py
```

Read-only acceptance uses canonical protocol/result/summary/video owners, checks
the exact task/seed/prompt matrix, full provenance, all raw versus summary
metrics, and retained video frame counts plus first/last decode. It rejects
unresolved infrastructure-invalid episodes. Smoke retains all 48 videos; formal
validates canonical category representatives rather than claiming all 120 episode
videos are retained. CPU helpers use explicit one-task allocations:

```bash
srun --account=operator --partition=normal --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  --output=NEW_EXCLUSIVE_LOG \
  "$XARM_WORK_ROOT/resume-smoke-acceptance.sh"
# After 120 completes: formal acceptance plus before-resume SHA/size/mtime snapshot.
srun --account=operator --partition=normal --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  --output=NEW_EXCLUSIVE_LOG \
  "$XARM_WORK_ROOT/formal-acceptance-and-resume-before.sh"
# Only after actual matching-source GPU resume COMPLETED0:0:
srun --account=operator --partition=normal --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:15:00 \
  --output=NEW_EXCLUSIVE_LOG \
  "$XARM_WORK_ROOT/finalize-runtime-evidence.sh" COMPLETED_RESUME_JOB_ID
```

These helpers create exclusive reports. For a later independent audit copy the
read-only operator and choose new report filenames; do not overwrite original
evidence or blindly rerun finalization. The completed-result resume proof
compares all 120 result JSONs and retained clips before/after for identical
SHA-256, size and mtime, and checks actual Slurm completion. No forced mid-run
interruption was injected; this proves reuse of existing completed episodes,
not every possible failure-recovery scenario.

## Evidence, preserved failures and limits

All paths below are relative to the runtime root in the table above:

- Raw formal results: `mujoco_outputs/policy_evaluation/hf_real_20260703_resume_formal_v3/models/HF_REAL_20260703/tasks/TASK/seed_SEED/result.json`.
- Formal summary and video indexes: same model root's `summary.json`, `video_index.json/.csv`, `representative_video_index.json/.csv`; root `summaries/comparison.json` and video indexes.
- Smoke products: corresponding `hf_real_20260703_resume_smoke_v3` root; all four video views retained per episode.
- Acceptance: `evidence/resume-fixed-smoke-acceptance.json`, `resume-fixed-formal-acceptance.json`.
- Actual resume proof: `evidence/resume-fixed-formal-resume-proof.json`; before snapshot `resume-fixed-formal-resume-before.json`,172 files, inventory SHA-256 `6ebe14477b9cef587911c7846d4b93a56a7cdd938c9ddcb874e84052919b0548`.
- Formal full provenance SHA-256: `39c5092cb58692eadc27edee95d10c5464675a016b989e268bb00f15bea0e2b3`.
- Allocation/service records: `logs/cluster/runs/formal-sim-evaluation/JOB.json`, `logs/cluster/slurm/xarm-formal-sim-evaluation-HF_REAL_20260703-JOB.out/.err`, `logs/policy_service/JOB/load-0.json` and `server-0.log`.
- GPU samples: `evidence/gpu-monitor-JOB.csv`.
- Model-free human review: `evidence/place-human-review-1775233/seed_50018/combined.mp4` and `seed_50019/combined.mp4`,3 s each.
- Actual Place automatic-success representative: formal model root `representative_videos/place_red_pepper_in_ring/SUCCESS/seed_50022/combined.mp4`,61 frames; failure representatives are 50018 OutsideRing and 50020 StableButNotSustained,750 frames each. Automatic labels are not human judgments.
- Verified scientific archive: `archives/accepted-formal-20261004T065509Z`,778 runtime files/189414330 uncompressed bytes. `file-manifest.json` records every runtime member; five products in `SHA256SUMS` passed check and all three gzip streams passed in 1775552. Report `evidence/final-evidence-archive.json`; readback log `evidence/final-runtime-evidence.log`.

After guide publication, `completion-audit-and-sidecar.sh PUBLISHED_GUIDE_COMMIT`
checks clean/published Git identity, unchanged third-party/scientific owners,
actual Slurm completion, all accepted reports, model/RNG/physical evidence,
archive checksums/streams and documentation links. It creates an exclusive
`post-publication/` sidecar below the archive with the final guide, audit report
and revisions plus separately checked hashes. This keeps the original five
scientific archive products unchanged. Its reports are
`evidence/completion-audit.json` and `post-publication-archive-readback.json`.

The source archive pins scientific `4d48a77`; guide publication happens afterward.
The runtime archive includes raw results, summaries, video indexes/retained
videos, both preserved `1eb9a929` and current `4d48a77` runs, human preview clips, environment
freeze, official-page snapshots, logs, evidence and private operators. It
excludes model weights, environments, caches and credentials. Each runtime
member has SHA-256/size and is read back from the compressed archive; product
checksums and gzip stream checks are separate real CPU checks. No existing
data was deleted. Preserve the fixed HF snapshot separately or redownload the
same immutable revision using its verified file integrity evidence.

Earlier failures remain visible. Smoke 1774743 had 10 valid pick failures and
2 Place reset-invalid episodes, correctly rejected before 120 submission. Its
187-file rejected-evidence archive was verified by 1774762 and 1774768 at
`archives/rejected-smoke-20261004T014607Z`. The bounded Place repair then
produced accepted smoke 1775235 and formal job `1775252` at source `1eb9a929`:
120 valid/0 invalid/5 success (Place 5/20; all five Pick tasks 0/20),52 category videos.
Actual resume 1775480 failed after 43 s on the JSON tuple/list defect, without
new episode inference;1775495 proved all 172 existing result/video files and
mtimes preserved. New-source `4d48a77` runs use separate output identities.

Other retained attempts include official-wheel timeout/NFS install timeout,
wrong operator test/protocol filenames, an optional attach monitor missing
batch-GPU metadata, wrong global-index EGL gate failure before restoration,
and the two-task read-only auditor invocation. Each was diagnosed before
necessary bounded continuation; none is silently counted as a policy failure.

Three unrelated baseline defects are documented with unchanged assertions:
`simulation/tools/teleoperate_pick.py` imports a higher-level data module;
config inventory expects only v4 although paired-v1 exists in the baseline;
the MJCF hash fixture expects `e7d09d0d...` while baseline/current XML hash is
`ac657567...` (full proof `evidence/baseline-mjcf-hash-fixture.json`). The broad
optional real-data fixture was not exercised. An earlier unmarked login-node
OpenGL test failed without a context; subsequent affected physics/rendering
ran in allocations and passed. No baseline tests were weakened or removed.

Scientific success rate is not a deployment acceptance threshold. The HF card's
Real-robot-only label and unknown original task coverage limit interpretation
of these six simulation tasks. Neither automated sim success nor these videos
establishes real-robot performance. No real hardware, training/retraining,
merge or force-push occurred. No human-review decision has been fabricated.
