# xArm formal-evaluation slip diagnosis

This workflow diagnoses object motion relative to the gripper without changing
formal task scoring or policy observations. Historical evaluation outputs are
read-only. Every diagnostic run must use a new root below
`/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics`.

## What is measured

Set `XARM_SLIP_TRACE=1` to enable physics-cadence tracing in the formal episode
runner. The recorder writes `slip_trace.csv` only at episode completion, using
a temporary file followed by an atomic rename.

The reference grasp offset is the first physics sample with at least one
target–finger contact. With

```text
relative_offset = tcp_position - object_position
```

the diagnostics are

```text
relative_3d_drift = norm(relative_offset - reference_relative_offset)
relative_downward_slip = max(0, relative_offset_z - reference_relative_offset_z)
```

A positive relative-Z increase therefore means that the object moved downward
relative to the TCP. Samples before the first target–finger contact have blank
drift/slip fields. The CSV also contains raw/clamped gripper commands, the
simulator gripper target, measured gripper state, per-finger target contacts,
left/right fingertip–table contacts, target–table contact, contact normal force,
penetration distance, target velocity, and finite-difference TCP vertical
velocity. None of these values is added to the policy observation.

## Diagnostic-only post-success continuation

The diagnostic entry point uses the historical formal v1 protocol and its
unchanged RNG salt. At the first original v1 success, it freezes the original
success, validity, metrics, policy-step count, executed-action count, and
safety accounting. Policy execution then continues for the requested duration
(default 2 seconds) only to collect physical evidence. An error during this
extra continuation is recorded as a diagnostic error and cannot retroactively
change the frozen task outcome.

The policy RNG remains derived from `(v1 protocol salt, task, seed,
policy_step)`. Subsetting the protocol or selecting c1/c2/c5 does not introduce
a global RNG stream. A c5 run should therefore reproduce the historical prefix
up to the original success point, subject to normal hardware/runtime
determinism.

## Primary c5 reproduction

Do not submit without explicit approval. The Slurm wrapper uses one GH200 and
starts an identity-checked localhost OpenPI server.

```bash
cd /u/mw89/repos/embodied-ai-xarm
OUTPUT_ROOT=/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics/B_red_block_seed50000_c5 \
MODEL_SPEC=/u/mw89/repos/embodied-ai-xarm/configs/evaluation/sim/models/B.json \
TASK=red_block \
SEED=50000 \
EXECUTE_CHUNK_STEPS=5 \
POST_SUCCESS_SECONDS=2.0 \
sbatch slurm/xarm_eval/diagnose_slip.sbatch
```

Use separate `..._c2` and `..._c1` output roots with
`EXECUTE_CHUNK_STEPS=2` and `1`. Never reuse or resume one diagnostic root for
a different setting.

## Data-only analysis

After all three jobs complete:

```bash
cd /u/mw89/repos/embodied-ai-xarm
/u/mw89/repos/openpi/.venv/bin/python sim_mujoco/scripts/analyze_xarm_slip_trace.py \
  --trace c5=/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics/B_red_block_seed50000_c5/models/B/tasks/red_block/seed_50000/slip_trace.csv \
  --trace c2=/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics/B_red_block_seed50000_c2/models/B/tasks/red_block/seed_50000/slip_trace.csv \
  --trace c1=/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics/B_red_block_seed50000_c1/models/B/tasks/red_block/seed_50000/slip_trace.csv \
  --output-dir /work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics/B_red_block_seed50000_comparison
```

The analysis produces one plot per trace plus `slip_analysis.json` and
`slip_analysis.md`. It reports contact ordering, table-contact force and
penetration, relative slip immediately after table contact, contact
persistence, gripper-command/actual-state changes, TCP-relative drift, and
world-frame object drop. The v2 analysis schema also recomputes relative slip
from five reference events (first any-finger contact, first both-finger
contact, first 5 mm world lift, first 5 cm world lift, and the original v1
success) and reports each contiguous fingertip-table contact event separately.

## Completed c5 baseline

Slurm job `2905639` completed the B/red_block/seed-50000 c5 reproduction. The
first 241 decoded combined-video frames are pixel-identical to the historical
representative; the diagnostic video has 60 additional frames for the two
second continuation. The frozen episode, metrics, and safety records also
match the historical result.

The trace proves real TCP-relative motion: the object loses all finger contact
at 10.642 s and falls to the table while the TCP remains airborne. Even using
the original v1-success sample as the reference, maximum subsequent downward
slip is 0.1150 m. Before v1 termination, the first-contact reference already
shows 0.0147 m maximum downward slip. The commanded and measured gripper state
both move toward greater closure, not opening.

Three fingertip-table contact events were observed. The two substantial events
(2.235 N and 0.567 N) occur before the first target-finger contact. A third
weak event (0.0625 N) overlaps the grasp but does not coincide with an abrupt
relative-slip jump. This establishes correlation with an aggressive approach,
not causality; c2/c1 remain required.

Versioned baseline analysis artifacts are under
`/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics/B_red_block_seed50000_c5_analysis_v2`.

## Interpretation order

1. Confirm the c5 video and frozen v1 metrics reproduce the historical
   B/red_block/50000 episode before causal interpretation.
2. Establish whether TCP-relative slip is nonzero and when it begins.
3. Compare its onset with left/right fingertip–table contact, normal force,
   penetration, gripper commands, actual gripper state, and TCP motion.
4. Compare c5/c2/c1 with all other settings unchanged.
5. Only after the baseline mechanism is established, design a friction
   sensitivity test. The current MJCF has no finger-object-only contact pair:
   scaling a finger geom also changes finger-table friction, while scaling the
   target changes target-table friction. Such a test must name and account for
   that confound.
6. Compare with the scripted oracle using the same seed/environment and the
   same trace quantities. The oracle's accepted training trajectories already
   undergo a separate 2-second stable-grasp verification.

No conclusion about friction, table collision, model under-training, or a
success-criterion change is justified before these controlled traces exist.
