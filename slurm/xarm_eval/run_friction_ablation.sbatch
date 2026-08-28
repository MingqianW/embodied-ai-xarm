#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-friction-ab
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/openpi_real_sim/xarm-friction-ab-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/openpi_real_sim/xarm-friction-ab-%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
RUN_ID=${XARM_FRICTION_AB_RUN_ID:-split_pad_friction_ab_v1_20260814}
OUTPUT_ROOT=/work/nvme/bfmk/mw89/runs/embodied-ai-xarm/gripper_slip/friction_ablation/${RUN_ID}
CAMERA_CONFIG=${REPOSITORY}/sim_mujoco/config/diagnostics/legacy_split_pad_camera_calibration.yaml

cd "${REPOSITORY}"
test -x "${PYTHON}"
test -f "${CAMERA_CONFIG}"
test ! -e "${OUTPUT_ROOT}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export XARM_CAMERA_CONFIG_PATH="${CAMERA_CONFIG}"

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true
nvidia-smi

FILES=(
  sim_mujoco/gripper_mapping.py
  sim_mujoco/remote_policy_observation.py
  sim_mujoco/data_collection/conversions.py
  sim_mujoco/environment.py
  sim_mujoco/task_scenes.py
  sim_mujoco/gripper_slip_diagnostics.py
  sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py
  sim_mujoco/scripts/run_friction_ablation.py
  sim_mujoco/scripts/analyze_friction_ablation.py
  tests/test_friction_ablation.py
)
"${PYTHON}" -m compileall -q "${FILES[@]}"
"${PYTHON}" -m ruff format --check "${FILES[@]}"
"${PYTHON}" -m ruff check "${FILES[@]}"
"${PYTHON}" -m pytest -q tests/test_friction_ablation.py
"${PYTHON}" sim_mujoco/scripts/run_friction_ablation.py --validate-only
"${PYTHON}" sim_mujoco/scripts/run_friction_ablation.py \
  --output-root "${OUTPUT_ROOT}"
"${PYTHON}" sim_mujoco/scripts/analyze_friction_ablation.py \
  --run-root "${OUTPUT_ROOT}"

date
