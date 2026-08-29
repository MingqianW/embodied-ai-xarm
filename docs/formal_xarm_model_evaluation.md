# Formal xArm π0.5 MuJoCo evaluation

`python -m evaluation.sim.cli` is the only formal entry point for
the new A/B/C π0.5 comparison. It is protocol-driven, requires an explicit
model specification, and writes provenance-checked isolated outputs.

## Protocol

The active immutable protocol is
`configs/evaluation/sim/protocols/formal_xarm_pi05_eval_v2.json`:

- six canonical prompts: red pepper, blue block, red block, smallest block,
  largest block, and red-pepper placement in the ring;
- fixed seeds `50000..50019` for every task and every model;
- action horizon 10, execute first 5 actions, acquire a fresh observation, and
  repeat for at most 50 policy calls (250 executed targets);
- simulator action duration 0.1 s and expected MuJoCo timestep 0.002 s;
- calibrated base/wrist 640×480 RGB observations, resized/padded through the
  existing 224×224 policy preprocessing path;
- `video_policy=category_representative`: record each rollout temporarily, then
  retain exactly the lowest-seed video for every observed
  `(model, task, category)` key; category is `SUCCESS`, a diagnosed failure
  category, or `INVALID`;
- invalid episodes are saved and make the formal command fail.

The evaluator gives the learned policy only base RGB, wrist RGB, the canonical
7D state `[joint1..joint6, gripper]`, and the exact prompt. Simulator ground
truth is used only for reset, scoring, and diagnostics.

## Model selection and provenance

The model specifications are explicit JSON files:

- `configs/evaluation/sim/models/A.json`
- `configs/evaluation/sim/models/B.json`
- `configs/evaluation/sim/models/C.json`

Each names the training config, checkpoint root, manager step `15000`, and
embedded norm asset. A uses `xarm_pi05_real_v3sim_1x`; B and C both use
`xarm_pi05_real_v4sim_10x`.

Before a server starts, the CLI verifies the selected manager directory,
`params`, embedded `assets/<norm_asset>/norm_stats.json`, and the normal OpenPI
normalization loader. The formal server receives a compact provenance document
and the evaluator refuses a connection whose protocol/model identity differs.

Each `result.json` contains the model specification, resolved manager path,
normalization asset, OpenPI and embodied-ai-xarm commits, protocol hash, and
camera/task/XML paths plus hashes. `--resume` is allowed only when that
provenance exactly matches the existing model output.

## Deterministic policy sampling

Each inference request derives its JAX seed using BLAKE2s from the protocol
salt, task ID, evaluation seed, and policy-step index. The seed is sent as
transport metadata and removed before transforms/model input. Therefore an
earlier failure, model order, process restart, or partial resume cannot alter
the stochastic samples assigned to a later episode.

The server must use `--require-request-rng`; the evaluator verifies this in its
metadata handshake.

## Safety and scoring

All 10×7 predicted values must have the correct shape and be finite. Joint and
gripper safety clipping/rejection applies only to the five actions that will
actually execute. Results record every executed action, clipped-action count
and fraction, per-dimension counts, and invalid-reason taxonomy.

Pick success remains target-specific: the target must first be at least 5 cm
above its post-settle reference height for three successive policy checks. This
is only a provisional success. The runner then continues for three additional
c5 policy checks (about 1.5 s), monitors target height at every MuJoCo physics
step during that hold, and accepts success only when the target remains at or
above 5 cm without a downward movement greater than 5 mm from the hold-stage
peak. A detected slip resets the provisional success; a later stable re-grasp
may still succeed before the episode horizon. Diagnostics include initial,
final, and maximum height, confirmation counts, hold checks, and maximum
post-success downward slip.

Placement starts from the existing physically simulated free-body grasp. Before
policy control, the formal runner holds the reset target for ten 0.1 s checks
and rejects the episode if the pepper loses gripper contact, touches the table,
drifts more than 5 mm relative to the TCP, becomes non-finite, or has a
forbidden collision. No constraint or privileged assistance remains after this
reset validation.

Placement success requires all of:

1. requested opening plus zero pepper/finger contacts and at least 4.5 cm
   pepper-to-TCP separation;
2. geometry-aware containment: XY center distance ≤ `0.053 - 0.022 - 0.002 =
   0.029 m`;
3. pepper center near table height (0.005–0.040 m above table);
4. linear speed ≤ 0.01 m/s and angular speed ≤ 0.25 rad/s;
5. three consecutive policy checks.

The result schema reports every release, containment, height, and stability
diagnostic even for failures.

## Failure diagnosis

Current formal results use `xarm-formal-episode-v2`. Every valid unsuccessful
episode has a machine-readable `episode.failure_category`,
`episode.failure_stage`, `episode.failure_reason`, and compact
`failure_diagnostics` evidence. Invalid runtime episodes remain separate:
they retain `valid=false` and `invalid_reason`, with no task-failure category.

Pick failures are classified from tracked lift/contact/hold history as
`PICK_NO_MEANINGFUL_LIFT`, `PICK_PARTIAL_LIFT`,
`PICK_REACHED_HEIGHT_BUT_NOT_SUSTAINED`, `PICK_DROPPED_AFTER_LIFT`, or
`PICK_TIMEOUT_OTHER`. The 5 mm meaningful-lift boundary is diagnostic only;
the formal 5 cm lift criterion, three initial confirmations, three
post-success hold confirmations, and 5 mm post-success-slip tolerance are
recorded in the protocol and episode evidence.

Placement failures are classified from the existing release, containment,
height, stability, and sustained-success components as `PLACE_NOT_RELEASED`,
`PLACE_RELEASED_OUTSIDE_RING`, `PLACE_RELEASED_WRONG_HEIGHT`,
`PLACE_RELEASED_IN_RING_BUT_UNSTABLE`, `PLACE_STABLE_BUT_NOT_SUSTAINED`, or
`PLACE_TIMEOUT_OTHER`. No success threshold is changed by diagnosis.

`summary.json` and `summaries/comparison.json` include category counts and
rates among valid failures. Category-aware recording writes a separate,
authoritative `representative_video_index.json` and
`representative_video_index.csv` below each model root. Their exact selection
policy is `lowest_seed_per_model_task_category`: one representative for every
observed category, not one generic failure video per task.

The rollout recorder writes an episode-local `temporary_video/` bundle first.
Only after `result.json` is durable does retention classify the outcome. A
selected bundle is moved to `representative_videos/<task>/<category>/seed_<n>/`;
an unselected temporary bundle is deleted only after classification. If a
lower-seed representative appears later (for example after resume or
out-of-order execution), its bundle and index are finalized before the older
bundle is removed. This video bookkeeping never changes automated success,
failure diagnosis, task metrics, or seeds.

Validate coverage after a category-representative evaluation:

```bash
cd /u/mw89/repos/embodied-ai-xarm
/u/mw89/repos/openpi/.venv/bin/python \
  evaluation/sim/tools/validate_category_video_coverage.py \
  --evaluation-root /work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/pi05_abc_15000_six_task_stable_hold_v2
```

It prints every observed category for each model/task and exits nonzero with
`CATEGORY VIDEO COVERAGE INCOMPLETE` if a representative bundle is missing.

Historical v1 result files can be reclassified without simulation and without
modifying them:

```bash
python evaluation/sim/tools/reclassify_failures.py \
  --root /work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/pi05_abc_15000_smoke_v1
```

The command writes schema-v2 copies, summaries, and video indexes below the
source root's `derived/failure_diagnosis_v1/` tree. It refuses a non-empty
derived target unless `--overwrite-derived` is explicit.

## Blinded human video review

Human review is an independent artifact layer. It never changes automated
`result.json`, success labels, failure categories, or numerical metrics.

1. Build a deterministic full-review manifest after videos are available:

   ```bash
   cd /u/mw89/repos/embodied-ai-xarm
   /u/mw89/repos/openpi/.venv/bin/python evaluation/sim/tools/build_human_review_manifest.py \
     --evaluation-root /work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/pi05_abc_15000_six_task_stable_hold_v2 \
     --review-seed 20260808 \
     --mode full
   ```

   The command writes a private manifest containing model/automated metadata,
   a reviewer-safe manifest containing only anonymized IDs and prompts, and a
   coverage report with every missing-video episode. Full review includes only
   videos that exist; do not describe it as a complete 360-episode human study
   unless `coverage_complete` is true.

2. Start the localhost-only review UI. The reviewer sees only review ID, task
   prompt, and one video; model identity and automated labels are not returned
   by its API.

   ```bash
   /u/mw89/repos/openpi/.venv/bin/python evaluation/sim/tools/review_human_videos.py \
     --review-root /work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/pi05_abc_15000_six_task_stable_hold_v2/human_review/full_seed_20260808
   ```

   If the browser is remote, use an SSH tunnel to `127.0.0.1:8765`; do not bind
   the review service to a public interface. Reviewers select `SUCCESS`,
   `FAILURE`, or `UNCERTAIN`, optionally choose a failure subtype, add notes,
   and continue automatically. Decisions are saved immediately to
   `human_review.csv`; restarting the same command resumes at the first
   undecided item. Existing decisions require
   `--allow-overwrite-decisions` to change.

3. After all reviewable items are decided, unblind and summarize:

   ```bash
   /u/mw89/repos/openpi/.venv/bin/python evaluation/sim/tools/summarize_human_review.py \
     --review-root /work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/pi05_abc_15000_six_task_stable_hold_v2/human_review/full_seed_20260808
   ```

   This writes `human_review_unblinded.csv`, JSON/Markdown summaries, and a
   paired A/B/C task-seed table. Reports include per-model/per-task human
   success rates, automated-vs-human agreement, Cohen's kappa as a secondary
   statistic, disagreement video paths, and paired A/B/C win/loss transitions.

`--mode representative` reads the representative-video indexes directly and
selects exactly one automated `SUCCESS`, failure category, or `INVALID` example
per model × task. It is qualitative only; the full blinded review is the
scientific comparison. `--mode full` reads every available per-episode video;
use the explicit all-video protocol below if complete 360-episode review is
required.

## Output layout and summaries

```text
<output-root>/
  protocol.json
  models/<A|B|C>/model_config.json
  models/<A|B|C>/tasks/<task>/seed_<seed>/result.json
  models/<A|B|C>/representative_videos/<task>/<category>/seed_<seed>/combined.mp4
  models/<A|B|C>/representative_video_index.json
  models/<A|B|C>/representative_video_index.csv
  models/<A|B|C>/summary.json
  summaries/comparison.json
```

Success rate is reported over valid episodes and over all attempted episodes;
invalid rate and reasons are always retained. Macro success is the unweighted
mean of the six per-task valid-episode success rates. The summary reads the
stable result schema rather than guessing historical field names.

## Dry run, compute smoke, and formal launch

Use this login-safe check after the manager checkpoint exists:

```bash
cd /u/mw89/repos/embodied-ai-xarm
/u/mw89/repos/openpi/.venv/bin/python -m evaluation.sim.cli \
  --model-spec configs/evaluation/sim/models/A.json \
  --protocol configs/evaluation/sim/protocols/formal_xarm_pi05_eval_v2.json \
  --output-root /work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/pi05_abc_15000_six_task_stable_hold_v2 \
  --openpi-root /u/mw89/repos/openpi \
  --dry-run
```

Then validate comparison-level invariants, including byte-identical embedded
Norm BC in B/C:

```bash
cd /u/mw89/repos/embodied-ai-xarm
/u/mw89/repos/openpi/.venv/bin/python evaluation/sim/tools/validate_abc_evaluation.py \
  --model-spec configs/evaluation/sim/models/A.json \
  --model-spec configs/evaluation/sim/models/B.json \
  --model-spec configs/evaluation/sim/models/C.json \
  --openpi-root /u/mw89/repos/openpi
```

The repository-owned Slurm entry point is the `formal-sim-evaluation` workflow
in `cluster/`. It is intentionally a thin client of an already-running,
provenance-configured OpenPI server. Starting that upstream server remains a
manual external OpenPI operation because this repository does not own a
validated canonical server launcher. Supply its reachable host explicitly;
localhost is not assumed across separately scheduled jobs.

The checked-in smoke protocol uses the same task, camera, safety, c5/i50, and
RNG semantics but two seeds and a separate output root. Generate and inspect the
Slurm command only after the server and static checks are ready:

```bash
python -m cluster.cli submit formal-sim-evaluation --dry-run \
  --param model_spec=configs/evaluation/sim/models/A.json \
  --param protocol=configs/evaluation/sim/protocols/formal_xarm_pi05_eval_smoke_v2.json \
  --param host=POLICY_SERVER_HOST
```

Do not modify the formal protocol or reuse smoke output roots.

For a complete blinded-review collection, use the explicit all-video protocol
and its separate output root. It preserves every episode video rather than
category representatives:

```bash
python -m cluster.cli submit formal-sim-evaluation --dry-run \
  --param model_spec=configs/evaluation/sim/models/A.json \
  --param protocol=configs/evaluation/sim/protocols/formal_xarm_pi05_eval_video_all_v2.json \
  --param host=POLICY_SERVER_HOST
```

This is a distinct output identity; do not mix it with the
category-representative formal output.

The v1 formal, smoke, and all-video protocol files are retained for historical
results that used the former three-check lift rule without a post-success hold.
Do not resume or combine those outputs with v2.

## Historical entry points — do not use for new A/B/C evaluation

- `slurm/pi05_xarm_abc_six_task_eval.sbatch`
- `slurm/pi05_xarm_abc_six_task_smoke.sbatch`
- `slurm/pi05_xarm_abc_six_task_smoke_v2.sbatch`
- `evaluation/sim/legacy/evaluate_remote_policy_automatic.py`
- `evaluation/sim/legacy/run_remote_policy_closed_loop.py`
- historical `summarize_xarm_abc*_evaluation.py` scripts

The former Slurm launchers are available only through Git history. The Python
legacy modules remain for result reproducibility. They include old checkpoint
identities, output layouts, and/or legacy result schemas and are not valid
formal launchers for the new A/B/C models.
