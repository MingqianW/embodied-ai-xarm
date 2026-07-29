# DeltaAI MuJoCo Migration Handoff

This document is self-contained and may be pasted into ChatGPT to continue
the migration without re-auditing the Windows laptop.

## 1. Repository

- Owner-confirmed GitHub URL and configured local `origin`:
  `https://github.com/MingqianW/embodied-ai-xarm.git`
- Branch: `main`
- Pre-migration-commit HEAD:
  `33104d957122a1830a2723d6e5e6f07036fb5a83`
- `main` is 3 commits ahead of `origin/main`.
- Proposed migration commit message:
  `Prepare MuJoCo workflow for DeltaAI migration`
- Migration commit: the containing commit with message
  `Prepare MuJoCo workflow for DeltaAI migration`; use `git rev-parse HEAD`
  after cloning for its exact self-referential SHA.
- Scope decision: the owner confirmed that unrelated Isaac Sim files and the
  stray `1.0.5` file must remain outside this commit.
- Push verification: run `git ls-remote origin main` and require it to equal
  `git rev-parse HEAD`. The completion response accompanying this handoff
  records the verified SHA.

Do not force-push or rewrite history.

## 2. Current local environment

- OS: Microsoft Windows NT 10.0.26200.0; registry reports Windows 10 Home
  25H2, build 26200.8875.
- Python executable: `D:/miniconda/envs/mujoco-pi/python.exe`
- Python: 3.11.15, AMD64
- MuJoCo: 3.10.0
- NumPy: 2.4.6
- OpenCV packages: `opencv-python` 4.10.0.84 and
  `opencv-python-headless` 5.0.0.93
- Pillow: 12.3.0
- imageio: 2.37.4
- imageio-ffmpeg: 0.6.0
- LeRobot: 0.1.0
- datasets: 3.6.0
- huggingface-hub: 1.24.0
- PyYAML: 6.0.3
- OpenPI client: 0.1.0 from the local OpenPI checkout
- pytest: not installed; local validation used `unittest`.

## 3. DeltaAI target

The following values were supplied by the owner and were not verified from
the Windows machine:

- account: `bfmk-dtai-gh`
- interactive partition: `ghx4-interactive`
- batch partition: `ghx4`
- architecture: Linux aarch64
- large-data root: `/work/nvme/bfmk/mw89/`
- expected repository: `~/repos/embodied-ai-xarm`
- OpenPI repository: `~/repos/openpi`
- checkpoint:
  `/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/xarm_pi05_20260703_run1/30000`

## 4. Active MuJoCo implementation

- Active scene XML:
  `sim_mujoco/assets/xarm6/xarm6_pick_scene.xml`
- Arm MJCF source generator:
  `sim_mujoco/scripts/generate_xarm6_mjcf.py`
- Task scene generator:
  `sim_mujoco/scripts/build_xarm6_pick_scene.py`
- Active camera config:
  `sim_mujoco/config/camera_calibration.yaml`
- Baseline camera config:
  `sim_mujoco/calibration/baseline_camera_calibration.yaml`
- Camera names: `overview_camera`, `base_camera`, `wrist_camera`
- Active task config: `sim_mujoco/config/task_scenes.yaml`
- State/action ordering:
  `[joint1, joint2, joint3, joint4, joint5, joint6, gripper_raw]`
- Arm mapping: identity, radians, named joints. The kinematic audit found
  identity-plus-frame-offset validation error of 0.270 mm and 0.177 degrees;
  the previous apparent TCP discrepancy was a flange/TCP-frame label issue,
  not a joint permutation/sign issue.
- Gripper mapping: official xArm four-bar linkage model, raw closed/open
  50/845, driver maximum 850 units, 1000 units/radian, simulation slide
  0.006–0.047 m.
- Physics timestep: 0.002 s.
- Oracle action timestep: 0.1 s (10 Hz, 50 physics steps/action).
- Remote closed-loop control duration: 0.02 s/action by default.
- Remote action chunk: server output `(10,7)`; default execution is one
  action before re-observation, configurable from 1 through 10.
- Native images: RGB `uint8`, 640×480.
- Policy images: RGB `uint8`, 224×224 using resize-with-pad.
- Policy observation keys:
  `observation/image`, `observation/wrist_image`, `observation/state`,
  `prompt`.

Active tasks, exact configured prompts, and success criteria:

| Task | Prompt | Success |
| --- | --- | --- |
| `red_block` | `pick_up_the_red_block` | 5 cm sustained lift, 3 policy steps |
| `blue_block` | `pick_up_the_blue_block` | 5 cm sustained lift, 3 policy steps |
| `largest_block` | `pick up the largest block` | 5 cm sustained lift, 3 policy steps |
| `smallest_block` | `pick_up_the_smallest_block` | 5 cm sustained lift, 3 policy steps |
| `red_pepper` | `pick_up_the_red_pepper` | 5 cm sustained lift, 3 policy steps |
| `place_red_pepper_in_ring` | `place_the_red_pepper_in_the_ring` | released, within 5.2 cm XY and 9 cm above table, sustained 3 steps |

The red-block oracle recorder deliberately uses the real-training prompt
`pick up the red block`; do not normalize or change prompt semantics during
migration.

## 5. Required code and assets

The complete 101-file hashed list is in:

- `docs/mujoco_migration/required_files_manifest.json`
- `docs/mujoco_migration/required_files_manifest.md`

Critical paths:

- `.gitmodules`
- `environment/mujoco_deltaai_requirements.txt`
- `environment/mujoco_deltaai_environment.md`
- `scripts/check_deltaai_mujoco_environment.py`
- `sim_mujoco/paths.py`
- `sim_mujoco/environment.py`
- `sim_mujoco/remote_policy_observation.py`
- `sim_mujoco/remote_policy_control.py`
- `sim_mujoco/remote_policy_evaluation.py`
- `sim_mujoco/task_scenes.py`
- `sim_mujoco/config/camera_calibration.yaml`
- `sim_mujoco/config/task_scenes.yaml`
- `sim_mujoco/assets/xarm6/xarm6_pick_scene.xml`
- `third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/link_base.stl`
  and `link1.stl` through `link6.stl`
- `policy_runtime/`
- `sim_mujoco/data_collection/`
- `sim_mujoco/scripts/collect_oracle_data.py`
- `sim_mujoco/scripts/convert_mujoco_to_lerobot.py`
- `sim_mujoco/scripts/validate_mujoco_lerobot_dataset.py`
- `sim_mujoco/scripts/prepare_mujoco_hf_ready.py`
- `sim_mujoco/scripts/upload_mujoco_dataset_to_hf.py`
- `sim_mujoco/scripts/run_remote_policy_closed_loop.py`
- `sim_mujoco/scripts/smoke_test_headless_render.py`
- `fine_tune/xarm_lerobot_writer.py`
- `tests/`

## 6. Current status

| Feature | Status | Evidence |
| --- | --- | --- |
| Headless-capable render path | complete locally | both policy cameras rendered; Linux EGL unverified |
| Task reset | complete | all six tasks compile/reset in tests |
| Camera views | complete locally | shapes/dtypes pass; real/sim visual domain gap remains |
| Collision validation | complete for encoded rules | every task starts without forbidden contact; link/table/self-collision tests pass |
| Scripted oracle | complete for red-block lift | current 1/1; historical fixed 10/10 and randomized 20/20 |
| Multi-task oracle | partial | collection code supports six tasks; migration validation only exercised red block |
| Raw data collection | complete locally | one 72-sample episode recorded |
| LeRobot conversion | complete locally | canonical v2.1, 1 episode/72 frames |
| Dataset validation | complete locally | exact schema, RGB, finite values, `(10,7)` horizon |
| Hugging Face dry run | complete locally | 11 files, no upload |
| Remote policy evaluation | implemented, unverified in this migration run | server intentionally not required/contacted |
| Linux aarch64 EGL | unverified | DeltaAI-gated |
| Slurm batch collection | missing | command/script must be generated on DeltaAI |

## 7. Known problems and limitations

- The local tree contains unrelated untracked Isaac Sim work. Do not include
  it in the focused MuJoCo migration commit; the owner explicitly confirmed
  this exclusion.
- The OpenPI submodule is configured but locally uninitialized according to
  Git, even though a local checkout is present. DeltaAI must run
  `git submodule update --init third_party/openpi` if using that copy.
- Linux aarch64 wheel availability is unverified, particularly for OpenCV,
  PyArrow/LeRobot transitive dependencies, and imageio-ffmpeg.
- EGL and NVIDIA driver visibility are unverified.
- MP4 codecs differ by node. Recorders try MP4, then AVI where applicable,
  then PNG frame sequences; downstream jobs must accept the recorded path
  from metadata rather than assuming `.mp4`.
- The first local dataset validation hit an unwritable default Hugging Face
  cache. DeltaAI must set `HF_HOME`, `HF_DATASETS_CACHE`, and `HF_HUB_CACHE`
  to writable scratch.
- Real/sim camera appearance, lighting, texture, optics, and visual gripper
  geometry still have a substantial domain gap. Calibration establishes
  camera semantics but does not remove that gap.
- Collision geometry is simplified; contact tolerances are task-specific and
  should be rechecked after Linux/MuJoCo version changes.
- The current action rates intentionally differ: oracle data is 10 Hz,
  while remote closed-loop action holding defaults to 20 ms. Do not change
  this merely for portability.
- The original recorder for some historical TCP-bearing real CSV files is not
  present. The kinematic audit corroborates the identity mapping, but those
  CSV fields should not be relabeled as raw encoders.
- Live remote π0.5 output, policy metadata, normalization compatibility, and
  same-node localhost throughput were not tested in this migration run.
- No real Hugging Face upload was performed.

## 8. Exact local commands

From repository root:

```powershell
$python = 'D:\miniconda\envs\mujoco-pi\python.exe'

& $python scripts/check_deltaai_mujoco_environment.py
& $python sim_mujoco/scripts/smoke_test_headless_render.py --task red_block --seed 0

& $python sim_mujoco/scripts/test_scripted_oracle.py `
  --task red_block --episodes 10 --seed-start 0 `
  --object-xy-range 0 --object-yaw-range-deg 0 --joint-noise 0

& $python sim_mujoco/scripts/collect_oracle_data.py `
  --episodes 3 --seed-start 1000 --task red_block `
  --object-xy-range 0.01 --object-yaw-range-deg 5 --joint-noise 0.005 `
  --headless

& $python sim_mujoco/scripts/convert_mujoco_to_lerobot.py

$env:HF_HOME = "$PWD\sim_mujoco\output\hf_cache"
$env:HF_DATASETS_CACHE = "$env:HF_HOME\datasets"
$env:HF_HUB_CACHE = "$env:HF_HOME\hub"
& $python sim_mujoco/scripts/validate_mujoco_lerobot_dataset.py `
  --skip-openpi-batch

& $python sim_mujoco/scripts/prepare_mujoco_hf_ready.py
& $python sim_mujoco/scripts/upload_mujoco_dataset_to_hf.py `
  --private --dry-run

& $python sim_mujoco/scripts/run_remote_policy_closed_loop.py `
  --host 127.0.0.1 --port 18000 --task red_block `
  --headless --max-policy-steps 80
```

## 9. Proposed DeltaAI sequence

1. Stage A — clone the confirmed repository URL, checkout the exact confirmed
   commit, initialize submodules, and compare SHA256 hashes.
2. Stage B — create a fresh Python 3.11 Linux aarch64 environment.
3. Stage C — install native EGL/driver prerequisites and the grouped Python
   dependencies.
4. Stage D — export scratch/cache paths and run
   `scripts/check_deltaai_mujoco_environment.py --require-egl`.
5. Stage E — run one headless dual-camera render.
6. Stage F — run one fixed scripted-oracle episode.
7. Stage G — run ten fixed scripted-oracle episodes and require 10/10.
8. Stage H — record three raw episodes under NVMe scratch.
9. Stage I — convert and validate the three episodes; retain reports.
10. Stage J — submit a restartable Slurm batch collection job with explicit
    output root, logs, wall time, and signal handling.
11. Stage K — prepare and dry-run the Hugging Face upload, inspect the file
    manifest, then request explicit authorization for any real upload.
12. Stage L — allocate one node, start π0.5 bound to localhost, wait for its
    health check, then run MuJoCo against `127.0.0.1:18000`.

## 10. DeltaAI commands ChatGPT must generate next

Generate commands only after confirming DeltaAI site module/package names:

- Slurm interactive allocation for `ghx4-interactive` and account
  `bfmk-dtai-gh`;
- Python 3.11 environment creation and activation;
- Linux aarch64 dependency installation;
- native EGL/OpenGL diagnostic commands;
- environment-variable exports for repository, output, datasets, OpenPI,
  checkpoint, and Hugging Face caches;
- exact-commit clone/submodule/hash verification;
- one EGL render smoke test;
- one and ten fixed-oracle runs;
- three-episode raw collection, conversion, and validation;
- `sbatch` script for resumable long oracle collection on `ghx4`;
- `squeue`, `sacct`, log-tail, cancellation, and resume commands;
- HF-ready preparation, dry run, authentication, and separately authorized
  upload;
- same-node OpenPI server and MuJoCo localhost launch with cleanup/traps.

## 11. Files not pushed

The following are intentionally local-only and must not be committed:

- `fine_tune/data/` raw real-robot datasets, including approximately 24 GiB
  `raw.tar` and 22 GiB `raw.zip`;
- `sim_mujoco/output/` generated episodes, conversions, reports, videos, and
  the 2.5 GiB local output tree;
- generated calibration images/logs/overlays/contact sheets;
- `.cache/`, Hugging Face caches, virtual environments, Conda/UV caches;
- checkpoints, WandB runs, videos, arrays, Parquet files, and archives;
- `.env*` and any credentials/tokens.

Required runtime meshes and small YAML/XML assets are normal Git files. No
required file exceeds GitHub's normal size limit; Git LFS is not needed.
Large datasets should be regenerated on DeltaAI or transferred separately to
NVMe/Hugging Face, never through this Git repository.

## 12. Verification hashes

```text
9f6d8ef3dc3be7ed8996dfc8513742e6257322efaafe6268536b639f4e80edd4  sim_mujoco/assets/xarm6/xarm6_pick_scene.xml
0ec030ef4bc28d9155deb663a82c1132d2ff6f1bf16af6ef6a8a68b5089350f4  sim_mujoco/config/camera_calibration.yaml
6a341b24975c2ee5413b12f7d6495eafd32b938e898d1db092e922f15ecd6b1e  sim_mujoco/config/task_scenes.yaml
17fa963e887bef5feb03577ebfc2d92020721f0ce17d916307971d2ecad548c4  environment/mujoco_deltaai_requirements.txt
0cfd0dc591c718c2b93e11dea11d1e4b52ec835f59fcdf6f2250421ee022effe  sim_mujoco/scripts/collect_oracle_data.py
c9a9a89f766976ec9f788059d31902d2d3452f0cb5adbca63f5302642c44bf52  sim_mujoco/scripts/convert_mujoco_to_lerobot.py
```
