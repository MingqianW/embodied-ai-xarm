# Evaluation architecture

`evaluation/` is the canonical owner of policy-evaluation identities, episode
outcomes, evidence, review, and reporting. It deliberately does not provide a
single backend runner: MuJoCo and a physical xArm have different reset,
measurement, and safety responsibilities.

## Shared contracts

`evaluation/common/` owns backend identity, run/episode identity, the normalized
result view, model/checkpoint specifications, human-review decisions, and
content-addressed provenance helpers. Task IDs and exact prompts are not copied
there; `data.common.task_identity` remains their only registry.

The common result is a normalized view over backend-native documents. Formal
simulation JSON remains compatible with the established v1/v2 schemas, while
new records explicitly include `backend: sim`. Real records use their own
schema and explicitly include `backend: real`, `automatic_success: null`, and
the source of any eventual human outcome.

## Formal simulation evaluation

`evaluation/sim/` owns the protocol, deterministic request seeds, MuJoCo episode
runner, success measurement, failure diagnosis, slip evidence, artifacts,
provenance, representative-video selection, blinded review integration, and
summaries. Its supported entrypoint is:

```bash
python -m evaluation.sim.cli --help
```

Formal protocols and model selections live under
`configs/evaluation/sim/protocols/` and `configs/evaluation/sim/models/`.
Pre-formal evaluators remain under `evaluation/sim/legacy/` because their
human-label and output semantics are not interchangeable with the formal
six-task protocol.

Simulation measurement may inspect MuJoCo object state and contacts. Those
implementations stay in `evaluation.sim`; a task prompt is shared, but its
measurement backend is not.

## Real-robot boundary

`evaluation/real/run_policy.py` is the cleaned-up location of the existing
operator-driven hardware runtime. Importing it is offline-safe. Running it
requires `--allow-hardware`, site-owned OpenPI and `real_world` packages, and a
second interactive authorization confirming operator presence, a clear
workspace, and emergency-stop access.

The repository can currently acquire two RGB observations, infer a policy
chunk, delta-limit joint targets, clamp the gripper command, execute a
receding-horizon rollout through the external xArm wrapper, and log raw images,
state, actions, and metadata. It cannot automatically perceive objects or
determine real task success. Real result records therefore remain unreviewed
until a shared human-review decision is attached. See
`evaluation/real/README.md` for the precise capability and dependency boundary.

Phase validation must use only offline result creation, pure safety-gate tests,
and mocks. It must never connect to or move a physical robot.

## Adjacent ownership

- `policy_runtime/` owns policy transport, observations/actions, reusable action
  safety, and logging primitives. Evaluation result ownership remains entirely
  under `evaluation/`.
- `simulation/` owns MuJoCo models, reset/runtime, observations, and robot
  control.
- `diagnostics/` owns maintained physics, camera, environment, and gripper
  checks; operator-facing evaluation utilities live under
  `evaluation/sim/tools/`.
