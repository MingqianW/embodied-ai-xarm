# Embodied AI xArm

Tools for collecting xArm 6 demonstrations, validating and converting datasets,
fine-tuning OpenPI policies, and reproducing the task in MuJoCo or Isaac Sim.

## Repository Layout

- `data_collection/` and `scripts/`: real-robot collection and dataset tools.
- `docs/`: task specifications and workflow documentation.
- `fine_tune/`: raw-data checks, LeRobot conversion, and OpenPI helpers.
- `sim_mujoco/`: xArm 6 scene generation, camera calibration, and tests.
- `sim_isaac/`: local Isaac Sim asset preparation, scene/cameras, policy
  runners, evaluation, and diagnostics.
- `evaluation/`: shared contracts, formal MuJoCo evaluation, and the explicit
  safety/review boundary for the existing real-robot runtime.
- `policy_runtime/`: simulator-independent OpenPI observations, transport,
  action safety, logs, and recording primitives.
- `third_party/openpi/`: OpenPI Git submodule.
- `third_party/xarm_ros2/`: vendored xArm ROS 2 descriptions and meshes.

## Clone

Clone with submodules so the OpenPI dependency is available:

```bash
git clone --recurse-submodules <repository-url>
```

For an existing clone:

```bash
git submodule update --init --recursive
```

## Environments

The root data-collection environment is described by `environment.yml` and
`requirements.txt`. MuJoCo calibration has a small pinned dependency set in
`sim_mujoco/constraints.txt`.

Raw demonstrations, converted datasets, local environments, calibration
renders, and machine-specific path configuration are intentionally excluded
from Git. Fine-tuning scripts default to
`fine_tune/data/xarm_pi05_data/raw`; pass `--raw-root` or create an ignored
`configs/data/real/xarm_data_config.json` to use another location.

See `docs/step1_data_collection.md`, `docs/step2_openpi_finetuning.md`,
`simulation/calibration/README.md`, and
`docs/ISAAC_SIM_SETUP.md` for the detailed workflows.
