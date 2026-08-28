#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-slip-data
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_gripper_data_%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_gripper_data_%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
PERSISTENT_ROOT=/work/nvme/bfmk/mw89
DATA_RUN_NAME=${XARM_GRIPPER_DATA_RUN_ID:-real_sim_${SLURM_JOB_ID}}
DATA_OUTPUT=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/${DATA_RUN_NAME}
REAL_DATASET=${PERSISTENT_ROOT}/datasets/lerobot/local/xarm_pi05_20260703
SIM_DATASET=${PERSISTENT_ROOT}/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v4_10x_real

if [[ -e "${DATA_OUTPUT}" ]]; then
  echo "Refusing existing output: ${DATA_OUTPUT}" >&2
  exit 3
fi
test -d "${REAL_DATASET}"
test -d "${SIM_DATASET}"
mkdir -p "${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip"
mkdir -p "${PERSISTENT_ROOT}/caches/matplotlib"
export MPLCONFIGDIR=${PERSISTENT_ROOT}/caches/matplotlib

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
test -n "${SLURM_JOB_ID:-}"
scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | head -n 40
echo "REAL_DATASET=${REAL_DATASET}"
echo "SIM_DATASET=${SIM_DATASET}"
echo "DATA_OUTPUT=${DATA_OUTPUT}"
cd "${REPOSITORY}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true
nvidia-smi

"${PYTHON}" sim_mujoco/scripts/analyze_real_sim_gripper_trajectories.py \
  --real-root "${REAL_DATASET}" \
  --sim-root "${SIM_DATASET}" \
  --output-dir "${DATA_OUTPUT}" \
  --gripper-q01 206.2904512664795 \
  --gripper-q99 844.8720790008545

test -s "${DATA_OUTPUT}/results.json"
test -s "${DATA_OUTPUT}/episode_metrics.csv"
echo "Validated dataset comparison output: ${DATA_OUTPUT}"
date
