# FarmShare simulation evaluation deployment

## Verification status

Deployment is **not complete**. The actual Linux environment, fixed HF snapshot,
strict model restoration, service identity, request RNG, and EGL observation
checks passed. Smoke produced all 12 episodes but failed infrastructure
acceptance on both Place resets. No formal 120-episode job was submitted.

The operator confirmed unsponsored research and a full-service SUNet ID.
CPU installation `1774160` passed all six steps in 4m42s on `wheat-01`, with
202 compatible packages; the environment freeze is in the external runtime.
Earlier wheel-timeout/NFS-install failures are retained in its state/logs.
Snapshot `1774189` verified all 23 files and read all 106 OCDBT logical keys
(12,441,508,165 bytes). CPU preflight `1774370` passed in 1m02s; action-transform
probe `1774384` verified quantile normalization, six-joint delta/absolute
round-trip, and unchanged raw gripper. Physics audit `1774385` found zero
compiled differences across all six generation/evaluation tasks.

Allocated environment `1774383` reported ready on `oat-02`: Python 3.11.15,
MuJoCo 3.3.7, x86_64, NVIDIA L40S (46,068 MiB), driver 595.91.07, EGL render,
and MP4 encoding. Seven native regressions passed, including all 120 fixed-seed
scene resets. The overall job failed at an existing MJCF hash fixture: the
unchanged source baseline and current XML both hash to `ac657567...`, while
the test expects `e7d09d0d...`. The original assertion remains intact.

Model verification `1774739` completed in 2m03s, 16 CPU/60G/one GPU, peak RSS
31,681,640K. It restored the specified checkpoint twice with exact parameter
key/shape checking and warmed up finite 10x7 final actions. Same-seed requests
were bitwise equal after an intervening different seed and across restart;
different seeds changed actions, missing seeds were rejected, actual-load
identity matched the evaluator, and both real simulation camera images passed.
Live GPU snapshots were 8,661/8,717 MiB; these are observations, not peak claims.
An optional attaching monitor step lacked the batch GPU environment and failed;
verification itself completed successfully. Subsequent monitoring runs in the
batch wrapper, selecting only `SLURM_JOB_GPUS`.

Smoke `1774743` completed its 12-result matrix in 3m35s on `oat-02`,
16 CPU/60G/one GPU, peak RSS 18,930,128K. All ten pick episodes were valid
model failures (nine `PICK_NO_MEANINGFUL_LIFT`, one `PICK_PARTIAL_LIFT`);
both Place episodes were infrastructure-invalid before model inference.
Overall success is 0/12, valid success is 0/10, and the six-task valid macro
is undefined because Place has no valid denominator. Read-only audit
`1774750` verified raw/summary/per-task/macro/provenance agreement and all 48
video files (frame counts and endpoint decode), then correctly rejected the
two reset failures. Its earlier single-frame seeking failure was an auditor
issue; for one frame, the decoded first frame is also the last frame.
GPU monitoring sampled every five seconds and observed a maximum 8,824 MiB.
These are smoke findings, not a completed formal benchmark or human review.

CPU diagnosis `1774748` reproduced the Place reset conflict on all 20 formal
seeds. The canonical configuration enables a kinematic `held_red_pepper`,
while v3 validation requires real fingertip contacts and the runbook describes
a free-body grasp. The unchanged fixture has zero contacts on every check.
Using the existing transfer at exactly the same pose also fails all 20 seeds:
the free pepper falls below the required height or contacts the table. No pose
was tuned and no threshold, protocol, model, or scene asset was changed.
Progress requires the previously validated free-body initialization (especially
`initial_tcp_to_object`) or an explicit scientific correction to this conflict.
Preserve these failed results; a source/reset correction needs a new isolated
output identity and cannot resume them with changed provenance.

Actual evaluation source was `9a2981c63f09a8d892b6b079d24802a5df4f6629`.
Runtime evidence, raw results, summaries, video indexes, and job logs are under
`$HOME/xarm-work/farmshare-20261003`; continue from `state.json`/`STAGES.md`.

Source baseline: `refactor/reorganize-repository` at
`924f4e5771d76f336e085aba3b90ca67cabc7b53`; do not substitute `main`.
OpenPI submodule pin and unmodified external checkout:
`15a9616a00943ada6c20a0f158e3adb39df2ccac`.
Public target `MingqianW/xarm-pi05-20260703`, fixed Hugging Face revision
`c308df02100a3f2b67512cc7ae61e28f16ee9263`.

The target model card says `Real-robot-only`. Its repository has checkpoint
contents at the root, no numeric manager directory or original training
config, and an embedded
`assets/local/xarm_pi05_20260703/norm_stats.json`. Its normalization SHA-256 is
`63a3d8456b509600fde0e9a3546fa2c01145cb0c3527d46ead649b4c35b37fc4`.
The operator subsequently supplied the original xArm config snippet and named
`pi05_xarn`, interpreted as its `pi05_xarm` entry. Its inference fields specify
Pi0.5 full model, internal action dimension 32, horizon 10,
`discrete_state_input=false`, Libero input/output transforms, a six-joint delta
mask, and absolute raw gripper. This matches the inference fields of the
tracked historical snippet, not proof that the model is the later Delta run
that reused the same name. Resolve the snippet's template dataset asset ID to
the target checkpoint's own embedded asset explicitly, and verify the actual
parameters/transforms/normalization in an allocation. Do not infer original
training task coverage or manager step. The separate full snapshot integrity report supplies file/OCDBT evidence;
strict parameter/configuration restoration has now passed in job `1774739`.

## Official policy and actual cluster checks

These official pages were read and archived on 2026-10-03
(America/Los_Angeles):

- [Introduction](https://docs.farmshare.stanford.edu/)
- [Policy](https://docs.farmshare.stanford.edu/policy/)
- [Connecting](https://docs.farmshare.stanford.edu/connecting/)
- [Slurm](https://docs.farmshare.stanford.edu/slurm/)
- [Software](https://docs.farmshare.stanford.edu/software/)
- [Transfer](https://docs.farmshare.stanford.edu/transfer/)

Before any job, confirm full-service SUNet eligibility and that this project is
coursework or unsponsored research. Complete authentication through Stanford's
normal mechanism; never put passwords, tokens, or private keys in configuration.
Keep model restoration, GPU initialization, warm-up, inference, rendering, and
substantial simulation inside Slurm. Do not run a persistent model server on a
rice login node or connect directly to an unallocated compute node.

Read-only commands verified in the initial audit:

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

Actual partitions were `normal`, `bigmem`, and `gpu`; the documentation's
`interactive` example did not match the current partition list. The observed
GPU partition had a two-day limit, four GPUs per node, and maximum memory of
4000 MB per requested CPU. Slurm can increase allocated CPUs to satisfy a
GiB memory request, so retain actual accounting as well as requested values.
Node features reported L40S/48GB; this is scheduler
inventory, not an allocated driver or inference measurement. Actual QoS limits
also differed from the documentation's example. Available modules included
Python 3.12.3 and CUDA 12.9.0, differing from the software page's older module
examples. Recheck before each deployment; do not load a CUDA module solely
because it appears in an example.

Filesystem available space is not user quota. The `quota` executable was not
available, and these pages did not establish a storage retention/backup
guarantee. Keep source and required scientific results in separately named
durable user directories; put rebuildable environments, model snapshots, and
caches outside source. Explicitly establish an archive destination and verify
copies before relying on it. Prefer the documented DTN/Globus mechanism for
large transfers. The model snapshot workflow runs transfers on a scheduled
compute node and verifies downloaded data before any model restoration.

## Paths and scheduler templates

Set absolute deployment locations outside the repository:

```bash
export XARM_REPOSITORY="$HOME/projects/embodied-ai-xarm"
export OPENPI_ROOT="$HOME/projects/openpi"
export XARM_WORK_ROOT="$HOME/xarm-work/farmshare-RUN_ID"
export XARM_PYTHON="$XARM_WORK_ROOT/envs/openpi/bin/python"
# Set XARM_SLURM_ACCOUNT from the verified association, not an example account.
source "$XARM_REPOSITORY/cluster/farmshare/environment.example.sh"
cd "$XARM_REPOSITORY"
```

Keep a machine-specific environment file and `state.json` under the runtime
root, recording stage status, evidence paths, source/model/OpenPI revisions,
actual environment freeze, pending gates, and all job IDs. Source, external
OpenPI, environments, checkpoint snapshots, caches, logs, smoke, formal results,
and archives must be distinct paths. Reuse completed downloads/environments;
do not submit again merely because a previous turn ended.

The [resource overrides](../../cluster/farmshare/resources.json) and
[environment template](../../cluster/farmshare/environment.example.sh) change
only site scheduling. Inspect them through the existing CLI, using an existing
audit interpreter if the full model environment is not ready:

```bash
python -m cluster.cli show environment-check
python -m cluster.cli submit environment-check --dry-run
python -m cluster.cli show formal-sim-evaluation \
  --param model_spec=/absolute/path/to/verified-model.json --param host=127.0.0.1
```

These command-resolution checks have been exercised; they do not prove that a
job can run. The unmodified OpenPI lock pins JAX/JAXlib 0.5.3, Flax 0.10.2,
Orbax 0.11.13, NumPy 1.26.4, TensorStore 0.1.74, and ml-dtypes 0.4.1. The
separate initial audit environment used Python 3.12.3 and MuJoCo 3.3.7 for
offline checks. The installed model environment uses Python 3.11.15 and those pinned
OpenPI dependencies; GPU compatibility was verified on the allocated L40S described above. The pinned upstream lock includes MuJoCo
2.3.7 through the unused Aloha simulator, while this project's scene requires
MuJoCo >=3.2. The `openpi-inference-environment` workflow exports the frozen
upstream dependencies, omits the unrelated `gym-aloha`/`dm-control`/`mujoco`
packages, installs the pinned client, and adds MuJoCo 3.3.7, OpenCV headless
4.11.0.86, and pytest 8.4.2. OpenPI source/lock remain unchanged; its source is
loaded from `OPENPI_ROOT/src` rather than installed as a package with Aloha
dependencies. The workflow checks dependencies and saves a freeze file.
Use a CPU partition/QoS for this CPU-only setup job:

```bash
XARM_SLURM_PARTITION=normal XARM_SLURM_QOS=normal \
  python -m cluster.cli submit openpi-inference-environment --dry-run \
  --param uv=/absolute/path/to/verified/uv
```

Remove `--dry-run` only after the site eligibility/resource checks. Record
compatibility and driver evidence inside an allocation before claiming
readiness. No model is imported during installation.

## Model and evaluation gates

Use the canonical [model contract](../../evaluation/common/models.py) and keep
the target's new model spec in `configs/evaluation/sim/models/`; preserve A/B/C/D.
Root-exported checkpoints explicitly use `manager_step: null`, so an unknown
manager/training step is not fabricated. The original normalization asset is
mandatory. After downloading the fixed revision, verify every snapshot file
and manifest-referenced OCDBT parameter data, then restore and infer on a
compute node. File preflight alone is insufficient.

The pinned upstream service lacks the evaluator's required
`formal_evaluation_provenance` and `request_rng_required` boundary. The thin
[`policy_runtime.openpi_request_rng`](../../policy_runtime/openpi_request_rng.py)
adapter uses the pinned JAX Pi0/Pi0.5 public explicit-noise hook. Offline tests
verify transport stripping, seed propagation, repeat/restart behavior with a
fixture, and canonical final-action checks. Real GPU sampling and service
identity passed in job `1774739`. The service integration must strip request RNG metadata before model
transforms, use the request seed in real sampling, reject missing seeds, return
finite final 10x7 actions, and derive identity from the actually loaded
checkpoint/config/assets. Do not advertise required metadata without proving
the behavior. Keep third-party source unchanged.

Follow the [canonical scientific protocol](../formal_xarm_model_evaluation.md):
six exact tasks/prompts; named `clean_wide_v4` scene; horizon 10, execute 5,
at most 50 predictions; 0.1-second actions and 0.002-second physics; formal
seeds 50018–50037. Success/safety/reset/randomization criteria remain unchanged.
Smoke uses the canonical two-seed all-video protocol, twelve episodes, and a
separate output identity. Formal evaluation uses 120 episodes and category
representative retention. Use isolated target output identities without copying
scene randomization ranges or overwriting the existing A/B/C configurations.

Required sequence: narrow tests and config checks; checkpoint/norm/provenance
preflight; allocated GPU inference/RNG/EGL verification; closed-loop smoke and
infrastructure/video acceptance; measured-resource formal submission; raw-result
and summary/video-coverage validation. Keep service and client on one allocated
node initially. Verify readiness, warm-up, bounded timeouts, failures, and
cleanup before launching episodes. Record every job ID and avoid duplicates.

Current offline commands include:

```bash
python -m pytest tests/evaluation_sim/test_provenance.py -q
python -m pytest tests/evaluation_common/test_models.py -q
python -m pytest tests/cluster -q
```

Three pre-existing regressions were reproduced or checked against the untouched source baseline:
`test_simulation_does_not_depend_on_higher_layers` flags a data import in
`simulation/tools/teleoperate_pick.py`; the protocol-regression config inventory
test expects only the stable-v4 YAML despite the paired-v1 YAML in the source;
the model-contract test expects a different MJCF SHA-256 from the XML committed
at the same baseline. The byte-identical XML and stale expected constant are
recorded in `evidence/baseline-mjcf-hash-fixture.json`.
Tests have not been weakened to hide these failures. A broad test invocation
also attempted an unmarked rendering test on the login node and failed for lack
of OpenGL context; it did not use a GPU. Review test bodies before grouping
them. Allocated camera/joint/builder and all-120-scene-reset checks passed; the
original failing MJCF hash assertion was retained and reported. The broader
optional real-data fixture was not exercised.

Only resume with exactly matching provenance and the canonical evaluator's
`--resume`. Actual model restarts and request reproducibility passed. Full evaluation
rerun/resume remains unverified because the required smoke acceptance failed.
Use the source commit recorded by each result when checking resume; a later
documentation commit also changes the Git identity used by provenance. Never fabricate human-review decisions; sim performance alone
does not establish real-robot performance.

## Target commands and acceptance gates

After the CPU environment workflow passes its dependency check, inspect and
submit the fixed snapshot workflow using `XARM_PYTHON` from that environment:

```bash
XARM_SLURM_PARTITION=normal XARM_SLURM_QOS=normal \
  python -m cluster.cli submit checkpoint-snapshot --dry-run
```

The snapshot report verifies Hugging Face sizes/content hashes and reads every
OCDBT key/value. It is not evidence of a restored model. The target spec is
[`HF_REAL_20260703.json`](../../configs/evaluation/sim/models/HF_REAL_20260703.json),
using the explicit inference-only config `pi05_xarm_hf_20260703` and its own
normalization asset. Numeric manager step remains unknown.

The following command forms are implemented and covered by offline resolution
tests; model verification and smoke were actually executed as recorded above.
Run preflight first through the
same scheduler workflow, with a private
resource override for `formal-sim-evaluation` specifying zero GPUs, sufficient
CPU/memory, and a bounded time limit. OpenPI imports may be substantial, so
this deployment schedules even the file/configuration preflight:

```bash
XARM_SLURM_PARTITION=normal XARM_SLURM_QOS=normal \
XARM_SLURM_RESOURCE_CONFIG=/absolute/path/to/cpu-preflight-resources.json \
  python -m cluster.cli submit formal-sim-evaluation \
  --param dry_run=true \
  --param model_spec=configs/evaluation/sim/models/HF_REAL_20260703.json \
  --param protocol=configs/evaluation/sim/protocols/hf_real_20260703_smoke_v3.json \
  --param host=127.0.0.1

python -m cluster.cli submit formal-sim-evaluation --dry-run \
  --param model_spec=configs/evaluation/sim/models/HF_REAL_20260703.json \
  --param protocol=configs/evaluation/sim/protocols/hf_real_20260703_smoke_v3.json \
  --param host=127.0.0.1 --param port=18005 --param start_server=true \
  --param verification_report="$XARM_WORK_ROOT/evidence/gpu-model-verification.json"
```

Verification restores the real model twice, compares repeated seeds exactly
with an intervening different seed, rejects missing seeds, and renders both
canonical policy images through EGL. Metadata is derived after restoration
and a finite 10x7 warm-up. It never consumes prepared identity JSON.
Service logs and restoration reports live under
`$XARM_WORK_ROOT/logs/policy_service/SLURM_JOB_ID/`.

Use a new verification report path on a repeat, because report/service evidence
is created exclusively. After real verification passes, omit `verification_report` for twelve
all-video smoke episodes. Accept raw result validity, observation/action
contracts, scoring, provenance, and videos before choosing measured resources
and changing the protocol to `hf_real_20260703_formal_v3.json` for 120 episodes.
Both target protocols inherit the canonical v3 scientific fields and override
only output roots; scientific overrides/nested inheritance are rejected.
`resume=true` maps to the existing evaluator's guarded `--resume`.
