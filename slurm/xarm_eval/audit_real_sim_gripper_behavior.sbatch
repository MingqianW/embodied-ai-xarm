#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-gripper-audit
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm-gripper-audit-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm-gripper-audit-%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
PERSISTENT_ROOT=/work/nvme/bfmk/mw89
RUN_ID=${XARM_GRIPPER_AUDIT_RUN_ID:-real_sim_behavior_${SLURM_JOB_ID}}
OUTPUT_DIR=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_behavior_audit/${RUN_ID}
REAL_DATASET=${PERSISTENT_ROOT}/datasets/lerobot/local/xarm_pi05_20260703
SIM_DATASET=${PERSISTENT_ROOT}/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v4_10x_real
MENAGERIE_VALIDATION=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/menagerie_gripper_v1_retry2_20260814

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing existing output: ${OUTPUT_DIR}" >&2
  exit 3
fi
test -d "${REAL_DATASET}"
test -d "${SIM_DATASET}"
test -d "${MENAGERIE_VALIDATION}"
mkdir -p "${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_behavior_audit"
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
echo "MENAGERIE_VALIDATION=${MENAGERIE_VALIDATION}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd "${REPOSITORY}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true
nvidia-smi

"${PYTHON}" sim_mujoco/scripts/audit_real_sim_gripper_behavior.py \
  --real-root "${REAL_DATASET}" \
  --sim-root "${SIM_DATASET}" \
  --menagerie-validation-root "${MENAGERIE_VALIDATION}" \
  --output-dir "${OUTPUT_DIR}"

test -s "${OUTPUT_DIR}/results.json"
test -s "${OUTPUT_DIR}/motion_segments.csv"
test -s "${OUTPUT_DIR}/pick_hold_metrics.csv"
test -s "${OUTPUT_DIR}/frame_trajectories.csv"
echo "Validated read-only gripper behavior audit: ${OUTPUT_DIR}"
date
