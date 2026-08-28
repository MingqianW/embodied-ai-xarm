#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-menagerie-tune
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm-menagerie-tune-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm-menagerie-tune-%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
PERSISTENT_ROOT=/work/nvme/bfmk/mw89
RUN_NAME=${XARM_MENAGERIE_TUNING_RUN_ID:-menagerie_forcerange_${SLURM_JOB_ID}}
OUTPUT_ROOT=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/${RUN_NAME}
BASELINE_GRASP_ROOT=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/menagerie_gripper_v1_retry2_20260814/grasp_slip
FORCE_LIMITS=(1 1.5 2 3 5)

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing existing output: ${OUTPUT_ROOT}" >&2
  exit 3
fi
test -x "${PYTHON}"
test -s "${BASELINE_GRASP_ROOT}/results.json"
mkdir -p "${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip"
mkdir -p "${PERSISTENT_ROOT}/logs"
mkdir -p "${PERSISTENT_ROOT}/caches/matplotlib"
mkdir -p "${OUTPUT_ROOT}/force_width"
export MPLCONFIGDIR=${PERSISTENT_ROOT}/caches/matplotlib
export MUJOCO_GL=egl

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
test -n "${SLURM_JOB_ID:-}"
scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | head -n 40
nvidia-smi

cd "${REPOSITORY}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true

FILES=(
  simulation/robot/control.py
  sim_mujoco/data_collection/oracle_controller.py
  evaluation/sim/episode_runner.py
  evaluation/sim/legacy/run_remote_policy_closed_loop.py
  sim_mujoco/scripts/run_grip_force_vs_width.py
  sim_mujoco/scripts/analyze_grip_force_vs_width.py
  sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py
  sim_mujoco/scripts/analyze_menagerie_forcerange_tuning.py
  tests/test_grip_force_vs_width.py
  tests/test_remote_policy_pipeline.py
  tests/test_menagerie_forcerange_tuning.py
)
"${PYTHON}" -m compileall -q "${FILES[@]}"
"${PYTHON}" -m ruff format --check "${FILES[@]}"
"${PYTHON}" -m ruff check "${FILES[@]}"
git diff --check
"${PYTHON}" -m pytest -q \
  tests/test_grip_force_vs_width.py \
  tests/test_remote_policy_pipeline.py \
  tests/test_menagerie_forcerange_tuning.py \
  tests/test_formal_xarm_evaluation.py

for FORCE_LIMIT in "${FORCE_LIMITS[@]}"; do
  FORCE_LABEL=pm${FORCE_LIMIT}
  FORCE_ROOT=${OUTPUT_ROOT}/force_width/${FORCE_LABEL}
  set +e
  "${PYTHON}" sim_mujoco/scripts/run_grip_force_vs_width.py \
    --output-root "${FORCE_ROOT}" \
    --gripper-force-limit "${FORCE_LIMIT}" \
    --closing-rate-raw-per-s 244 \
    --opening-rate-raw-per-s 220
  FORCE_STATUS=$?
  set -e
  if [[ "${FORCE_STATUS}" -eq 0 ]]; then
    "${PYTHON}" sim_mujoco/scripts/analyze_grip_force_vs_width.py \
      --suite-root "${FORCE_ROOT}" \
      --output-dir "${FORCE_ROOT}/analysis"
  else
    "${PYTHON}" sim_mujoco/scripts/analyze_grip_force_vs_width.py \
      --suite-root "${FORCE_ROOT}" \
      --output-dir "${FORCE_ROOT}/analysis" \
      --report-invalid-contact-outcomes
  fi
done

"${PYTHON}" sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py \
  --output-root "${OUTPUT_ROOT}/grasp_slip" \
  --suite menagerie_forcerange \
  --task smallest_block \
  --task largest_block \
  --task red_block \
  --task blue_block \
  --task red_pepper \
  --seed 50000 \
  --hold-seconds 2 \
  --prerequisite-root "${BASELINE_GRASP_ROOT}"
"${PYTHON}" sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py \
  --suite-root "${OUTPUT_ROOT}/grasp_slip" \
  --output-dir "${OUTPUT_ROOT}/grasp_slip/analysis"
"${PYTHON}" sim_mujoco/scripts/analyze_menagerie_forcerange_tuning.py \
  --run-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/analysis"

test -s "${OUTPUT_ROOT}/analysis/parameter_results.json"
jq -e '.schema_version == "xarm_menagerie_forcerange_tuning_v1"' \
  "${OUTPUT_ROOT}/analysis/parameter_results.json" >/dev/null
jq -e '.candidates | length == 5' \
  "${OUTPUT_ROOT}/analysis/parameter_results.json" >/dev/null

echo "Completed isolated Menagerie force-range tuning: ${OUTPUT_ROOT}"
date
