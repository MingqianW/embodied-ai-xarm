# Repository architecture

This document is the current ownership and dependency reference for the
project. `REPOSITORY_REORGANIZATION_PLAN.md` records migration history; it is
not an operating architecture.

## Subsystems

| Subsystem | Owns | Does not own |
| --- | --- | --- |
| `policy_runtime` | Backend-independent policy schemas, image preprocessing, transport, action decoding/safety, recording, and episode logging | MuJoCo state, evaluation outcomes, policy-server deployment |
| `simulation` | MuJoCo assets/configuration, scene reset, observations, robot/control mappings, physics-facing runtime and trace instrumentation, development tools | Dataset schemas, training, result classification, cluster submission |
| `data.common` | Task identity, shared 7D records/schema, validation, and the LeRobot writer | Simulator implementation or training sampling |
| `data.real` | Offline discovery and conversion of externally collected raw data | Hardware acquisition/control |
| `data.sim` | Generation plans, oracle, recording, acceptance, audit, conversion, and simulation-dataset paths | Core simulation physics |
| `training` | Dataset selection, normalization, mixing, OpenPI configuration adaptation, experiment identity, and preflight | OpenPI implementation or evaluation |
| `evaluation.common` | Model identity, shared result views, provenance, and review contracts | Backend measurement |
| `evaluation.sim` | Formal deterministic protocol, simulation measurement, failure diagnosis, evidence, videos, and reports | Policy-server launch or real outcomes |
| `evaluation.real` | Explicitly authorized operator runtime and honest unreviewed/human-reviewed result boundary | Automatic real success perception |
| `diagnostics` | Maintained measurements of camera, physics, environment, gripper, and real/simulation behavior | Production control or orchestration |
| `cluster` | DeltaAI resources, environment, dependencies, submission, logs, and provenance | Scientific, dataset, evaluation, or training behavior |
| `tools` | Thin cross-cutting dataset inspection/export/preparation/upload operations | Canonical business logic |
| `configs` | Operator-selected data and evaluation inputs | Package-internal simulation or training definitions |
| `third_party` | External OpenPI and xArm ROS sources/assets | Project-owned application code |

`policy_runtime` remains top-level because its transport, schemas,
preprocessing, and safety primitives are shared by simulation and real/evaluation
runtimes without owning either backend. Evaluation result helpers are not
re-exported from it.

## Dependency direction

```text
data.sim ──> simulation ──> policy_runtime
    └──────> data.common <── data.real
training ──> data.common
training ──> external OpenPI

evaluation.sim  ──> simulation / policy_runtime / data.common / evaluation.common
evaluation.real ──> policy_runtime / data.common / evaluation.common

diagnostics / tools ──> canonical packages they inspect
cluster ──> canonical CLIs through subprocesses only
```

More precisely:

- `simulation` may depend on `policy_runtime`, but never on data, evaluation,
  training, diagnostics, or cluster.
- `data.common` and `data.real` are simulator-independent.
- `data.sim` may depend on `simulation` and `data.common`.
- `training` may depend on `data.common` and its own OpenPI boundary, never on
  evaluation, diagnostics, or cluster.
- Evaluation may depend on shared data contracts, `policy_runtime`, and the
  relevant backend.
- Diagnostics and tools may depend on canonical packages they inspect.
- `cluster` resolves canonical CLIs and executes them as subprocesses;
  canonical packages never import `cluster`.
- `third_party` is a leaf source boundary. Project-owned modules do not live
  inside it, and its tests are not part of normal project discovery.

## Stable contracts and single owners

| Concept | Canonical owner |
| --- | --- |
| Task IDs and prompts | `data.common.task_identity` |
| 7D state/action schema | `data.common.schema` and `data.common.records` |
| LeRobot serialization | `data.common.lerobot_writer` |
| MuJoCo model and configuration | `simulation.resources` and `simulation/config/` |
| Gripper conversion | `simulation.robot.gripper_mapping` |
| Physics-cadence trace instrumentation | `simulation.instrumentation.trace` |
| Simulation data plans | `data.sim.generation.plans` plus `configs/data/sim/generation/` |
| Training normalization and mixing | `training.normalization` and `training.mixing` |
| Model/checkpoint identity | `evaluation.common.models` |
| Formal result contract | `evaluation.sim.result_contract` |
| Physics consistency measurement | `diagnostics.simulation.physics.consistency` |
| Cluster resources | `cluster.workflows` |

Legacy modules under `data.sim.generation.legacy` and
`evaluation.sim.legacy` retain validated pre-formal formats or behavior that
current conversion/evaluation code still exercises. They are not aliases for
removed top-level packages and may not become competing registries.

## Configuration and runtime products

Package-internal immutable configuration stays beside its owner:
`simulation/config/`, `training/configs/`, diagnostic baselines, and cluster
deployment defaults. Operator-selected JSON/YAML lives under `configs/`.

Runtime datasets, evaluation outputs, videos, checkpoints, reports, logs, and
caches belong outside source packages. `MUJOCO_OUTPUT_ROOT`,
`MUJOCO_DATASET_ROOT`, and `XARM_WORK_ROOT` select external roots; ignored
`outputs/` and `datasets/` paths are local fallbacks.

## External boundaries

The repository intentionally does not own the physical xArm collector/driver,
RealSense acquisition, a complete OpenPI environment, policy-server launch,
remote datasets/checkpoints, Hugging Face credentials, or the DeltaAI
scheduler. Real-hardware execution requires the explicit gates documented in
`evaluation/real/README.md`; missing external capabilities must not be
simulated or inferred.
