#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-menagerie-gripper
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm-menagerie-gripper-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm-menagerie-gripper-%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
PERSISTENT_ROOT=/work/nvme/bfmk/mw89
RUN_NAME=${XARM_MENAGERIE_RUN_ID:-menagerie_gripper_${SLURM_JOB_ID}}
OUTPUT_ROOT=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/${RUN_NAME}
OLD_BASELINE_ROOT=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/baseline_20260812_v1

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing existing output: ${OUTPUT_ROOT}" >&2
  exit 3
fi
test -x "${PYTHON}"
test -s "${OLD_BASELINE_ROOT}/results.json"
mkdir -p "${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip"
mkdir -p "${PERSISTENT_ROOT}/logs"
mkdir -p "${PERSISTENT_ROOT}/caches/matplotlib"
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
  simulation/robot/gripper_mapping.py
  simulation/observation/policy.py
  simulation/robot/control.py
  simulation/environment.py
  simulation/scene
  sim_mujoco/data_collection/conversions.py
  simulation/tools/build_xarm6_pick_scene.py
  sim_mujoco/scripts/run_menagerie_gripper_validation.py
  sim_mujoco/scripts/run_grip_force_vs_width.py
  sim_mujoco/scripts/analyze_grip_force_vs_width.py
  sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py
  tests/simulation/test_gripper_integration.py
  tests/test_mujoco_data_conversions.py
  tests/simulation/test_gripper_motion.py
  tests/simulation/test_task_scenes.py
  tests/test_remote_policy_pipeline.py
  tests/test_grip_force_vs_width.py
)
"${PYTHON}" -m compileall -q "${FILES[@]}"
"${PYTHON}" -m ruff format --check "${FILES[@]}"
"${PYTHON}" -m ruff check "${FILES[@]}"
git diff --check

SCENE_SHA_BEFORE=$(sha256sum simulation/assets/xarm6/xarm6_pick_scene.xml | cut -d' ' -f1)
"${PYTHON}" -m simulation.tools.build_xarm6_pick_scene
SCENE_SHA_AFTER=$(sha256sum simulation/assets/xarm6/xarm6_pick_scene.xml | cut -d' ' -f1)
test "${SCENE_SHA_BEFORE}" = "${SCENE_SHA_AFTER}"

"${PYTHON}" -m pytest -q \
  tests/simulation/test_gripper_integration.py \
  tests/test_mujoco_data_conversions.py \
  tests/simulation/test_gripper_motion.py \
  tests/simulation/test_task_scenes.py \
  tests/test_remote_policy_pipeline.py \
  tests/test_grip_force_vs_width.py \
  tests/test_mujoco_collisions.py \
  tests/test_formal_xarm_evaluation.py

"${PYTHON}" sim_mujoco/scripts/run_menagerie_gripper_validation.py \
  --output-root "${OUTPUT_ROOT}/open_close"

set +e
"${PYTHON}" sim_mujoco/scripts/run_grip_force_vs_width.py \
  --output-root "${OUTPUT_ROOT}/force_width"
FORCE_SWEEP_STATUS=$?
set -e
if [[ "${FORCE_SWEEP_STATUS}" -eq 0 ]]; then
  "${PYTHON}" sim_mujoco/scripts/analyze_grip_force_vs_width.py \
    --suite-root "${OUTPUT_ROOT}/force_width" \
    --output-dir "${OUTPUT_ROOT}/force_width/analysis"
else
  "${PYTHON}" sim_mujoco/scripts/analyze_grip_force_vs_width.py \
    --suite-root "${OUTPUT_ROOT}/force_width" \
    --output-dir "${OUTPUT_ROOT}/force_width/analysis" \
    --report-invalid-contact-outcomes
fi

"${PYTHON}" sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py \
  --output-root "${OUTPUT_ROOT}/grasp_slip" \
  --suite baseline \
  --task smallest_block \
  --task largest_block \
  --task red_block \
  --task blue_block \
  --task red_pepper \
  --seed 50000 \
  --hold-seconds 2
"${PYTHON}" sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py \
  --suite-root "${OUTPUT_ROOT}/grasp_slip" \
  --output-dir "${OUTPUT_ROOT}/grasp_slip/analysis" \
  --comparison-root "${OLD_BASELINE_ROOT}"

test -s "${OUTPUT_ROOT}/open_close/open_close_validation.json"
test -s "${OUTPUT_ROOT}/force_width/analysis/steady_state_summary.json"
test -s "${OUTPUT_ROOT}/force_width/analysis/acceptance.json"
test -s "${OUTPUT_ROOT}/grasp_slip/results.json"
test -s "${OUTPUT_ROOT}/grasp_slip/analysis/summary.json"
jq -e '.measured_direction == "ctrl_0_open_ctrl_255_closed"' \
  "${OUTPUT_ROOT}/open_close/open_close_validation.json" >/dev/null
jq -e '
  (.status == "complete") or
  (.status == "failed" and .force_metrics_authorized == false and (.trials | length) == 7)
' "${OUTPUT_ROOT}/force_width/results.json" >/dev/null
jq -e '.status == "complete"' "${OUTPUT_ROOT}/grasp_slip/results.json" >/dev/null

echo "Validated Menagerie gripper run: ${OUTPUT_ROOT}"
date
