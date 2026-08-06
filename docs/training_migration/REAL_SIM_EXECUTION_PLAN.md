# π0.5 xArm Real/Simulation 50/50 Continuation Execution Plan

## Scope and gates

Run all GPU, EGL, collection, conversion, OpenPI batch, restore, training, and
evaluation work through Slurm on one GH200. Stop production training if a
required environment, semantic, mixed-batch, restore, source-immutability, or
baseline-serving gate fails.

## Focused changes

OpenPI files to change:

- `src/openpi/training/config.py`: add the mixed continuation data/config and,
  only if required, a loader-independent inference config.
- `src/openpi/training/data_loader.py`: add deterministic, step-derived 8-real
  plus 8-simulation LeRobot batching with independent per-source shuffles and
  cycling.
- `src/openpi/training/checkpoints.py` and `scripts/train.py`: separate the
  immutable restore source from the new output manager, detect a completed
  target, resume the new output, and save safely on `USR1`.
- Focused adjacent test files and small verification/manifest scripts under
  `scripts/` as required.

Files to create:

- `slurm/train_pi05_xarm_real_sim_50_50_continue.sbatch` plus durable staged
  submission helpers.
- Focused OpenPI tests for exact composition, deterministic reconstruction,
  pytree equality, true resume, source/output separation, timeout saving, and
  completion detection.
- The requested reports and status documents under
  `/work/nvme/bfmk/mw89/logs/openpi_real_sim_50_50`,
  this `docs/training_migration` directory, and the new run root.

Existing MuJoCo collection, conversion, strict validation, comparison, policy
runtime, and evaluation code will be reused. The shared
`fine_tune/xarm_lerobot_writer.py` remains the only LeRobot writer.

## Mixed-loader design

Construct the real and simulation LeRobot datasets independently with the same
xArm transforms and the normalization asset
`local/xarm_pi05_20260703`. For global batch step `s`, derive each source epoch
and offset from `seed`, `s`, source length, and the fixed per-source batch size
of eight. Each source epoch has its own deterministic permutation. Gather
exactly eight examples from each source, cycling and reshuffling independently
at exhaustion, then collate and optionally apply a deterministic within-batch
permutation. Reconstructing at the same restored global step therefore yields
the same subsequent sample identities without mutable iterator state or
physical duplication.

The real dataset metadata contains exactly 198 episodes. The training
simulation dataset therefore converts the first 198 successful raw episodes in
deterministic episode-index order. The conversion manifest records this
selection and strict validation requires real and simulation episode counts to
be equal. The complete 300-episode raw simulation pool is retained unchanged
for provenance; the additional 102 raw episodes are not exposed to training.

## Checkpoint separation

Treat source manager directory `.../xarm_pi05_20260703_run1/30000` as
read-only. An empty new output root restores that explicit source; a partial
new root restores its latest valid manager step; a valid manager directory
`50000` is verified and exits successfully. Restore raw parameters, Adam state
and count, EMA parameters, and scalar step. Record source hashes before and
after all smoke/training work. Never copy into or write under the source run.

## Slurm stages

1. Environment/EGL/scene/reset/gripper/collision gates; bounded dataset
   discovery; fixed and randomized oracle gates; resumable collection only if
   no valid dataset exists; conversion and strict validation.
2. Real-only pytree capture, distribution report, exact 8+8 loader checks,
   checkpoint-30000 restore verification, baseline serving/evaluation, and
   forward/backward/update throughput preflight in a temporary work directory.
3. Resumable production continuation, using one or an `afterok` chain of
   12-hour jobs based on measured throughput, stopping exactly at manager
   directory `50000`.
4. Fresh-process final checkpoint and policy-serving verification.
5. Same-node baseline-versus-final MuJoCo evaluation on an identical fixed
   seed list, followed by the run manifest and evaluation-ready commands.

All persistent data, caches, logs, and checkpoints use explicit paths under
`/work/nvme/bfmk/mw89`; W&B is online and policy traffic is localhost-only.

## Evaluation design

Serve baseline manager directory `30000` and final manager directory `50000`
with identical model, observation, postprocessing, real normalization, safety,
randomization, seed list, and control settings. Run at least 20 seeds when the
measured allocation runtime permits. Record automatic success/failure/invalid
outcomes, termination reasons, steps, latency, first-action magnitude,
clipping, and representative videos.

## Rollback

No upstream push or source-checkpoint mutation is permitted. Code rollback is
the removal/reversal of only this branch's focused diff, using the saved
pre-change status, patch, and hashes in
`/work/nvme/bfmk/mw89/logs/openpi_real_sim_50_50/pre_change`. Data and run
outputs are versioned and never overwrite existing results. Jobs can be
cancelled explicitly by ID; cancellation does not alter the source checkpoint.
Persistent outputs are retained for inspection and are not deleted without
separate approval.
