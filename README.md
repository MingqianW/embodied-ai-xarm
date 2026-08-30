# Embodied AI xArm

This repository provides the project-owned simulation, data, training,
evaluation, diagnostics, and cluster orchestration needed to develop OpenPI
policies for an xArm 6. It keeps physical-robot control and external services
behind explicit safety and deployment boundaries.

## Architecture

| Path | Responsibility |
| --- | --- |
| `simulation/` | Canonical MuJoCo model, scene configuration, reset, observations, robot control, and development tools |
| `data/common/` | Shared task identity, state/action records, validation, and LeRobot writing |
| `data/real/` | Offline discovery, validation, and conversion of externally collected real demonstrations |
| `data/sim/` | Simulation-data generation, acceptance, audit, conversion, and versioned plans |
| `policy_runtime/` | Backend-independent policy transport, schemas, preprocessing, action safety, logging, and recording |
| `evaluation/` | Shared result/model contracts, formal simulation evaluation, and the safety-first real evaluation boundary |
| `training/` | Dataset identities, normalization, mixing strategies, OpenPI adapter, experiment registry, and preflight |
| `diagnostics/` | Maintained camera, physics, environment, gripper, and real/simulation consistency checks |
| `cluster/` | DeltaAI/Slurm resource declarations, submission, dependencies, logs, and provenance |
| `tools/` | Thin cross-cutting dataset exporters, inspection, preparation, and upload helpers |
| `configs/` | Operator-selected data and formal-evaluation configuration |
| `tests/` | Project-owned unit, regression, architecture, and lightweight integration tests |
| `third_party/` | Vendored or submodule dependencies; not project-owned application code |

See [the repository architecture](docs/architecture/REPOSITORY_ARCHITECTURE.md)
for dependency direction and ownership details.

## Canonical data flow

```text
external real collector ──> data.real ──┐
                                        ├──> data.common contract ──> training
simulation ──> data.sim ────────────────┘

simulation + policy_runtime ──> evaluation.sim ──> result/review artifacts
external robot/runtime ───────> evaluation.real ──> human-reviewed results
```

`data.common.task_identity` is the task registry. `data.common.schema` and
`data.common.records` define the shared 7D state/action boundary. Generated
data and evaluation output are not source code and should live outside the
checkout or under an ignored runtime root.

## Simulation and simulation data

The canonical model is
`simulation/assets/xarm6/xarm6_pick_scene.xml`; camera, gripper, and task
configuration live under `simulation/config/`.

Useful checks and tools include:

```bash
python -m diagnostics.simulation.environment.headless_render --help
python -m simulation.tools.render_task_scenes --help
python -m simulation.tools.teleoperate_pick --help
python -m data.sim.generation.cli --help
python -m data.sim.generation.tools.validate_scripted_oracle --help
```

The versioned generation plans live in `configs/data/sim/generation/`. The
DeltaAI workflow is documented in
[`docs/simulation_data/DELTA_AI_RUNBOOK.md`](docs/simulation_data/DELTA_AI_RUNBOOK.md).

## Real data boundary

This repository does not own the physical xArm/RealSense collection stack.
`data.real` reads an existing raw dataset without commanding hardware. Set the
raw location in an ignored `configs/data/real/xarm_data_config.json`, pass the
relevant CLI override, or use the ignored default `datasets/real/raw`. Existing
ignored data at `fine_tune/data/xarm_pi05_data/raw` is detected as a deprecated
read-only compatibility fallback; it is never moved automatically. Dataset task
renaming previews by default and requires `--apply` to write changes.

The existing operator-controlled hardware runtime is isolated under
`evaluation/real/`. It requires explicit hardware authorization and operator
presence. Automatic real-world task success is not claimed; results remain
unreviewed until supported perception or human review supplies an outcome.

## Training

`training/` defines project-owned experiment identity and preprocessing, then
delegates execution to an external OpenPI checkout:

```bash
python -m training.cli list
python -m training.cli show pi05_xarm
python -m training.cli preflight pi05_xarm --help
```

Training does not run implicitly. Dataset paths, OpenPI availability, assets,
checkpoints, and launch support must pass preflight first. The established 7D
state/action convention, six-joint delta normalization, absolute raw gripper,
10-step horizon, and real/simulation mixing strategies are regression tested.

## Evaluation

Formal deterministic MuJoCo evaluation is exposed through:

```bash
python -m evaluation.sim.cli --help
```

Protocols and model specifications live under `configs/evaluation/sim/`.
Formal evaluation requires a separately started, verified OpenPI policy server;
this repository does not fabricate a server launcher. See
[`docs/formal_xarm_model_evaluation.md`](docs/formal_xarm_model_evaluation.md)
and [`evaluation/real/README.md`](evaluation/real/README.md).

## Diagnostics and DeltaAI

Diagnostics are import-safe and organized by measured subsystem. Start with:

```bash
python -m diagnostics.simulation.environment.check --help
python -m diagnostics.simulation.physics.consistency --help
python -m diagnostics.simulation.camera.cli --help
```

`cluster/` is the only maintained Slurm orchestration layer:

```bash
python -m cluster.cli list
python -m cluster.cli show WORKFLOW --param NAME=VALUE
python -m cluster.cli submit WORKFLOW --dry-run --param NAME=VALUE
```

Only `submit` without `--dry-run` calls `sbatch`. Consult
[`cluster/README.md`](cluster/README.md) for environment variables, resource
profiles, output ownership, dependencies, and provenance.

## Configuration and generated outputs

- Simulation runtime configuration: `simulation/config/`
- Data/operator configuration: `configs/data/`
- Evaluation models and protocols: `configs/evaluation/`
- Training experiment definitions: `training/configs/`
- Cluster deployment defaults: `cluster/config.py`

Use `MUJOCO_OUTPUT_ROOT`, `MUJOCO_DATASET_ROOT`, and `XARM_WORK_ROOT` for
external runtime storage. Local fallbacks use ignored `outputs/` and
`datasets/` roots. Checkpoints, logs, videos, caches, diagnostic products, and
machine-specific configuration are ignored by Git.

## Development and testing

Use Python 3.11 for the validated MuJoCo workflow. The root environment files
describe the physical data-collection dependencies; the DeltaAI MuJoCo groups
are documented in `environment/mujoco_deltaai_requirements.txt`. OpenPI client
code is supplied by the `third_party/openpi` submodule.

Clone dependencies with:

```bash
git submodule update --init --recursive
```

Run project-owned tests from the repository root:

```bash
python -m pytest
```

Test discovery is restricted to `tests/`; vendored suites under `third_party/`
are not collected. The project is currently used directly from its repository
root rather than through a published wheel.

## External dependencies

The physical collector/driver, RealSense acquisition, complete OpenPI runtime,
policy-server launch procedure, remote datasets, checkpoints, Hugging Face
credentials, and DeltaAI scheduler environment are intentionally external.
Their absence must be reported or handled by preflight—not replaced with
invented repository functionality.
