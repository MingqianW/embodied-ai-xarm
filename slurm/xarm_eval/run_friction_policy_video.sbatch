#!/usr/bin/env bash
#SBATCH --job-name=xarm-friction-video
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/openpi_real_sim/xarm-friction-video-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/openpi_real_sim/xarm-friction-video-%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
PHASE1_ROOT=/work/nvme/bfmk/mw89/runs/embodied-ai-xarm/gripper_slip/friction_ablation/split_pad_friction_ab_v1_20260814
VARIANT_ROOT=${PHASE1_ROOT}/policy_variant_v1
OUTPUT_ROOT=/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/slip_diagnostics/split_pad_friction_red_block_seed50000_v1_20260814
RUNTIME_MODEL=${VARIANT_ROOT}/split_pad_menagerie_like_friction.xml
RUNTIME_PROTOCOL=${VARIANT_ROOT}/formal_v2_runtime_friction.json
VARIANT_MANIFEST=${VARIANT_ROOT}/variant_manifest.json
CAMERA_CONFIG=${REPOSITORY}/sim_mujoco/config/diagnostics/legacy_split_pad_camera_calibration.yaml

cd "${REPOSITORY}"
test -x "${PYTHON}"
test -s "${PHASE1_ROOT}/analysis/phase2_gate.json"
test ! -e "${VARIANT_ROOT}"
test ! -e "${OUTPUT_ROOT}"

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "VARIANT_ROOT=${VARIANT_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true
nvidia-smi

"${PYTHON}" -m compileall -q \
  sim_mujoco/scripts/prepare_friction_policy_variant.py \
  sim_mujoco/scripts/analyze_friction_ablation.py \
  sim_mujoco/scripts/diagnose_xarm_slip.py \
  sim_mujoco/formal_evaluation/slip_trace.py
"${PYTHON}" -m ruff format --check \
  sim_mujoco/scripts/prepare_friction_policy_variant.py \
  sim_mujoco/scripts/analyze_friction_ablation.py \
  sim_mujoco/scripts/diagnose_xarm_slip.py \
  sim_mujoco/formal_evaluation/slip_trace.py
"${PYTHON}" -m ruff check \
  sim_mujoco/scripts/prepare_friction_policy_variant.py \
  sim_mujoco/scripts/analyze_friction_ablation.py \
  sim_mujoco/scripts/diagnose_xarm_slip.py \
  sim_mujoco/formal_evaluation/slip_trace.py
"${PYTHON}" sim_mujoco/scripts/prepare_friction_policy_variant.py --validate-only
"${PYTHON}" sim_mujoco/scripts/prepare_friction_policy_variant.py \
  --output-dir "${VARIANT_ROOT}" \
  --evaluation-output-root "${OUTPUT_ROOT}"
"${PYTHON}" sim_mujoco/scripts/analyze_friction_ablation.py \
  --run-root "${PHASE1_ROOT}" \
  --output-dir "${PHASE1_ROOT}/analysis_v2"

export XARM_MUJOCO_MODEL_PATH="${RUNTIME_MODEL}"
export XARM_CAMERA_CONFIG_PATH="${CAMERA_CONFIG}"
export OUTPUT_ROOT
export PROTOCOL="${RUNTIME_PROTOCOL}"
export MODEL_SPEC=${REPOSITORY}/sim_mujoco/config/formal_models/B.json
export TASK=red_block
export SEED=50000
export EXECUTE_CHUNK_STEPS=5
export POST_SUCCESS_SECONDS=5.0
export FRICTION_VARIANT_MANIFEST="${VARIANT_MANIFEST}"

bash slurm/xarm_eval/diagnose_slip.sbatch

TRACE=${OUTPUT_ROOT}/models/B/tasks/red_block/seed_50000/slip_trace.csv
"${PYTHON}" sim_mujoco/scripts/analyze_xarm_slip_trace.py \
  --trace "friction_B=${TRACE}" \
  --output-dir "${OUTPUT_ROOT}/analysis"

test -s "${OUTPUT_ROOT}/models/B/tasks/red_block/seed_50000/videos/combined.mp4"
test -s "${OUTPUT_ROOT}/analysis/slip_analysis.json"
date
