#!/usr/bin/env bash
#SBATCH --job-name=xarm-friction-opt
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/openpi_real_sim/xarm-friction-opt-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/openpi_real_sim/xarm-friction-opt-%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
RUN_ID=${XARM_FRICTION_SEARCH_RUN_ID:-friction_search_v1_20260814}
RUN_ROOT=/work/nvme/bfmk/mw89/runs/embodied-ai-xarm/gripper_slip/friction_search/${RUN_ID}
VARIANT_ROOT=${RUN_ROOT}/selected_policy_variant
OUTPUT_ROOT=/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics/friction_optimized_red_block_seed50000_v1_20260814
CAMERA_CONFIG=${REPOSITORY}/sim_mujoco/config/diagnostics/legacy_split_pad_camera_calibration.yaml

cd "${REPOSITORY}"
test -x "${PYTHON}"
test -s "${CAMERA_CONFIG}"
test ! -e "${RUN_ROOT}"
test ! -e "${OUTPUT_ROOT}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XARM_CAMERA_CONFIG_PATH="${CAMERA_CONFIG}"

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true
nvidia-smi

FILES=(
  sim_mujoco/scripts/run_friction_search.py
  sim_mujoco/scripts/analyze_friction_search.py
  sim_mujoco/scripts/prepare_friction_policy_variant.py
  sim_mujoco/scripts/diagnose_xarm_slip.py
  sim_mujoco/formal_evaluation/slip_trace.py
  tests/test_friction_ablation.py
)
"${PYTHON}" -m compileall -q "${FILES[@]}"
"${PYTHON}" -m ruff format --check "${FILES[@]}"
"${PYTHON}" -m ruff check "${FILES[@]}"
bash -n slurm/xarm_eval/diagnose_slip.sbatch
"${PYTHON}" sim_mujoco/scripts/run_friction_search.py --validate-only
"${PYTHON}" sim_mujoco/scripts/prepare_friction_policy_variant.py --validate-only

"${PYTHON}" sim_mujoco/scripts/run_friction_search.py \
  --output-root "${RUN_ROOT}"
"${PYTHON}" sim_mujoco/scripts/analyze_friction_search.py \
  --run-root "${RUN_ROOT}"

SELECTION=${RUN_ROOT}/analysis/selection.json
SELECTED_MU=$(jq -er '.selected_sliding_friction' "${SELECTION}")
SELECTED_ACCEPTED=$(jq -r '.selected_passes_all_acceptance_gates' "${SELECTION}")
echo "SELECTED_MU=${SELECTED_MU}"
echo "SELECTED_ACCEPTED=${SELECTED_ACCEPTED}"

"${PYTHON}" sim_mujoco/scripts/prepare_friction_policy_variant.py \
  --output-dir "${VARIANT_ROOT}" \
  --evaluation-output-root "${OUTPUT_ROOT}" \
  --lower-friction "${SELECTED_MU}" \
  --upper-friction "${SELECTED_MU}"

RUNTIME_MODEL=${VARIANT_ROOT}/split_pad_selected_friction.xml
RUNTIME_PROTOCOL=${VARIANT_ROOT}/formal_v2_runtime_friction.json
VARIANT_MANIFEST=${VARIANT_ROOT}/variant_manifest.json
export XARM_MUJOCO_MODEL_PATH="${RUNTIME_MODEL}"
export OUTPUT_ROOT
export PROTOCOL="${RUNTIME_PROTOCOL}"
export MODEL_SPEC=${REPOSITORY}/sim_mujoco/config/formal_models/B.json
export TASK=red_block
export SEED=50000
export EXECUTE_CHUNK_STEPS=5
export POST_SUCCESS_SECONDS=5.0
export FRICTION_VARIANT_MANIFEST="${VARIANT_MANIFEST}"

bash slurm/xarm_eval/diagnose_slip.sbatch

EPISODE=${OUTPUT_ROOT}/models/B/tasks/red_block/seed_50000
"${PYTHON}" sim_mujoco/scripts/analyze_xarm_slip_trace.py \
  --trace "selected_mu=${EPISODE}/slip_trace.csv" \
  --output-dir "${OUTPUT_ROOT}/analysis"

test -s "${EPISODE}/videos/combined.mp4"
test -s "${EPISODE}/result.json"
test -s "${OUTPUT_ROOT}/analysis/slip_analysis.json"
date
