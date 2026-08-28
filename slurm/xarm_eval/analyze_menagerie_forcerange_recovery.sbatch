#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-menagerie-analyze
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm-menagerie-analyze-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm-menagerie-analyze-%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
PERSISTENT_ROOT=/work/nvme/bfmk/mw89
RUN_ROOT=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/menagerie_forcerange_v1_20260814
GRASP_ANALYSIS=${RUN_ROOT}/grasp_slip/analysis_v2
FINAL_ANALYSIS=${RUN_ROOT}/analysis

test -x "${PYTHON}"
test -s "${RUN_ROOT}/grasp_slip/results.json"
if [[ -e "${GRASP_ANALYSIS}" ]]; then
  echo "Refusing existing recovery analysis: ${GRASP_ANALYSIS}" >&2
  exit 3
fi
if [[ -e "${FINAL_ANALYSIS}" ]]; then
  echo "Refusing existing final analysis: ${FINAL_ANALYSIS}" >&2
  exit 3
fi
mkdir -p "${PERSISTENT_ROOT}/logs"
mkdir -p "${PERSISTENT_ROOT}/caches/matplotlib"
export MPLCONFIGDIR=${PERSISTENT_ROOT}/caches/matplotlib

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "RUN_ROOT=${RUN_ROOT}"
test -n "${SLURM_JOB_ID:-}"
scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | head -n 40
nvidia-smi

cd "${REPOSITORY}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true

FILES=(
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py
  sim_mujoco/scripts/analyze_menagerie_forcerange_tuning.py
)
"${PYTHON}" -m compileall -q "${FILES[@]}"
"${PYTHON}" -m ruff format --check "${FILES[@]}"
"${PYTHON}" -m ruff check "${FILES[@]}"

"${PYTHON}" sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py \
  --suite-root "${RUN_ROOT}/grasp_slip" \
  --output-dir "${GRASP_ANALYSIS}"
"${PYTHON}" sim_mujoco/scripts/analyze_menagerie_forcerange_tuning.py \
  --run-root "${RUN_ROOT}" \
  --grasp-analysis-name analysis_v2 \
  --output-dir "${FINAL_ANALYSIS}"

test -s "${GRASP_ANALYSIS}/summary.json"
test -s "${FINAL_ANALYSIS}/parameter_results.json"
jq -e '.candidates | length == 5' \
  "${FINAL_ANALYSIS}/parameter_results.json" >/dev/null

echo "Recovered analysis without rerunning simulation: ${FINAL_ANALYSIS}"
date
