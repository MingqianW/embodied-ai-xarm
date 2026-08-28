#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-grip-width
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_grip_force_width_%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_grip_force_width_%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
PERSISTENT_ROOT=/work/nvme/bfmk/mw89
RUN_NAME=${XARM_GRIP_FORCE_RUN_ID:-grip_force_vs_width_${SLURM_JOB_ID}}
OUTPUT_ROOT=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/${RUN_NAME}

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing existing output: ${OUTPUT_ROOT}" >&2
  exit 3
fi

mkdir -p "${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip"

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
test -n "${SLURM_JOB_ID:-}"
scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | head -n 40

cd "${REPOSITORY}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true
nvidia-smi 2>/dev/null || true

FILES=(
  sim_mujoco/scripts/run_grip_force_vs_width.py
  sim_mujoco/scripts/analyze_grip_force_vs_width.py
  tests/test_grip_force_vs_width.py
)
"${PYTHON}" -m compileall -q "${FILES[@]}"
"${PYTHON}" -m ruff format --check "${FILES[@]}"
"${PYTHON}" -m ruff check "${FILES[@]}"
"${PYTHON}" -m pytest -q tests/test_grip_force_vs_width.py
"${PYTHON}" sim_mujoco/scripts/run_grip_force_vs_width.py --validate-only

"${PYTHON}" sim_mujoco/scripts/run_grip_force_vs_width.py \
  --output-root "${OUTPUT_ROOT}"
"${PYTHON}" sim_mujoco/scripts/analyze_grip_force_vs_width.py \
  --suite-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/analysis"

test -s "${OUTPUT_ROOT}/manifest.json"
test -s "${OUTPUT_ROOT}/model_validation.json"
test -s "${OUTPUT_ROOT}/results.json"
test -s "${OUTPUT_ROOT}/analysis/steady_state_summary.json"
test -s "${OUTPUT_ROOT}/analysis/steady_state_summary.csv"
test -s "${OUTPUT_ROOT}/analysis/acceptance.json"
test -s "${OUTPUT_ROOT}/analysis/summary.md"
test -s "${OUTPUT_ROOT}/analysis/force_chain.csv"
test -s "${OUTPUT_ROOT}/analysis/force_chain_vs_width.png"
jq -e '.status == "complete" and .all_widths_valid == true and .force_metrics_authorized == true' \
  "${OUTPUT_ROOT}/results.json" >/dev/null
for WIDTH_MM in 25 30 35 40 45 50 55; do
  jq -e '.passed == true' \
    "${OUTPUT_ROOT}/trials/width_${WIDTH_MM}mm/validity.json" >/dev/null
done
echo "Validated grip-force width diagnostic: ${OUTPUT_ROOT}"
date
