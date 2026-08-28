#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-menagerie-analysis
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm-menagerie-analysis-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm-menagerie-analysis-%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
SOURCE_ROOT=/work/nvme/bfmk/mw89/runs/embodied-ai-xarm/gripper_slip/menagerie_gripper_v1_retry2_20260814
GRASP_ROOT=${SOURCE_ROOT}/grasp_slip
OUTPUT_DIR=${GRASP_ROOT}/analysis_v2
OLD_BASELINE_ROOT=/work/nvme/bfmk/mw89/runs/embodied-ai-xarm/gripper_slip/baseline_20260812_v1

test -x "${PYTHON}"
test -s "${GRASP_ROOT}/results.json"
test -s "${OLD_BASELINE_ROOT}/results.json"
test ! -e "${OUTPUT_DIR}"
jq -e '.status == "complete" and (.trials | length) == 20' \
  "${GRASP_ROOT}/results.json" >/dev/null

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "SOURCE_ROOT=${SOURCE_ROOT}"
scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | head -n 40

cd "${REPOSITORY}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true

"${PYTHON}" -m compileall -q \
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py
"${PYTHON}" -m ruff format --check \
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py
"${PYTHON}" -m ruff check \
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py
"${PYTHON}" sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py \
  --suite-root "${GRASP_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --comparison-root "${OLD_BASELINE_ROOT}"

test -s "${OUTPUT_DIR}/summary.json"
test -s "${OUTPUT_DIR}/trial_summary.csv"
echo "Completed Menagerie analysis: ${OUTPUT_DIR}"
date
