#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-slip-scripted
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_gripper_slip_%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_gripper_slip_%j.err

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
PERSISTENT_ROOT=/work/nvme/bfmk/mw89
SLIP_SUITE_NAME=${XARM_SLIP_SUITE:-baseline}
SLIP_RUN_NAME=${XARM_SLIP_RUN_ID:-${SLIP_SUITE_NAME}_${SLURM_JOB_ID}}
SLIP_OUTPUT_ROOT=${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip/${SLIP_RUN_NAME}
SLIP_PREREQUISITE=${XARM_SLIP_PREREQUISITE_ROOT:-}

case "${SLIP_SUITE_NAME}" in
  baseline|contact|realism|geometry_realism|force|friction|geometry|object|dynamics) ;;
  *)
    echo "Unsupported XARM_SLIP_SUITE=${SLIP_SUITE_NAME}" >&2
    exit 2
    ;;
esac
if [[ "${SLIP_SUITE_NAME}" != baseline && "${SLIP_SUITE_NAME}" != geometry_realism && -z "${SLIP_PREREQUISITE}" ]]; then
  echo "Non-baseline suites require XARM_SLIP_PREREQUISITE_ROOT" >&2
  exit 2
fi
if [[ -e "${SLIP_OUTPUT_ROOT}" ]]; then
  echo "Refusing existing output: ${SLIP_OUTPUT_ROOT}" >&2
  exit 3
fi

mkdir -p "${PERSISTENT_ROOT}/runs/embodied-ai-xarm/gripper_slip"
mkdir -p "${PERSISTENT_ROOT}/caches/matplotlib"
export MPLCONFIGDIR=${PERSISTENT_ROOT}/caches/matplotlib
export MUJOCO_GL=egl

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
test -n "${SLURM_JOB_ID:-}"
scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | head -n 40
echo "SLIP_SUITE_NAME=${SLIP_SUITE_NAME}"
echo "SLIP_OUTPUT_ROOT=${SLIP_OUTPUT_ROOT}"
echo "SLIP_PREREQUISITE=${SLIP_PREREQUISITE}"
cd "${REPOSITORY}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true
nvidia-smi

"${PYTHON}" -m compileall -q \
  sim_mujoco/gripper_slip_diagnostics.py \
  sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py \
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py \
  sim_mujoco/scripts/run_contact_model_realism_regression.py \
  sim_mujoco/scripts/analyze_contact_model_realism_regression.py \
  sim_mujoco/scripts/run_split_pad_geometry_experiment.py \
  sim_mujoco/scripts/analyze_split_pad_geometry_experiment.py \
  tests/test_gripper_slip_diagnostics.py \
  tests/test_split_pad_geometry_experiment.py
"${PYTHON}" -m ruff format --check \
  sim_mujoco/gripper_slip_diagnostics.py \
  sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py \
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py \
  sim_mujoco/scripts/run_contact_model_realism_regression.py \
  sim_mujoco/scripts/analyze_contact_model_realism_regression.py \
  sim_mujoco/scripts/run_split_pad_geometry_experiment.py \
  sim_mujoco/scripts/analyze_split_pad_geometry_experiment.py \
  tests/test_gripper_slip_diagnostics.py \
  tests/test_split_pad_geometry_experiment.py
"${PYTHON}" -m ruff check \
  sim_mujoco/gripper_slip_diagnostics.py \
  sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py \
  sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py \
  sim_mujoco/scripts/run_contact_model_realism_regression.py \
  sim_mujoco/scripts/analyze_contact_model_realism_regression.py \
  sim_mujoco/scripts/run_split_pad_geometry_experiment.py \
  sim_mujoco/scripts/analyze_split_pad_geometry_experiment.py \
  tests/test_gripper_slip_diagnostics.py \
  tests/test_split_pad_geometry_experiment.py
git diff --check

"${PYTHON}" -m pytest -q \
  tests/test_gripper_slip_diagnostics.py \
  tests/test_xarm_slip_trace.py \
  tests/test_split_pad_geometry_experiment.py \
  tests/test_formal_xarm_evaluation.py \
  tests/test_mujoco_collisions.py \
  tests/test_mujoco_scripted_oracle.py

if [[ "${SLIP_SUITE_NAME}" == realism ]]; then
  "${PYTHON}" sim_mujoco/scripts/run_contact_model_realism_regression.py \
    --output-root "${SLIP_OUTPUT_ROOT}" \
    --prerequisite-root "${SLIP_PREREQUISITE}" \
    --seed 50000 \
    --seed 50001 \
    --seed 50002 \
    --hold-seconds 5
  "${PYTHON}" sim_mujoco/scripts/analyze_contact_model_realism_regression.py \
    --suite-root "${SLIP_OUTPUT_ROOT}" \
    --output-dir "${SLIP_OUTPUT_ROOT}/analysis"
elif [[ "${SLIP_SUITE_NAME}" == geometry_realism ]]; then
  "${PYTHON}" sim_mujoco/scripts/run_split_pad_geometry_experiment.py \
    --output-root "${SLIP_OUTPUT_ROOT}" \
    --seed 50000 \
    --seed 50001 \
    --seed 50002 \
    --hold-seconds 5
  "${PYTHON}" sim_mujoco/scripts/analyze_split_pad_geometry_experiment.py \
    --suite-root "${SLIP_OUTPUT_ROOT}" \
    --output-dir "${SLIP_OUTPUT_ROOT}/analysis"
else
  RUNNER=(
    "${PYTHON}"
    sim_mujoco/scripts/run_scripted_gripper_slip_experiments.py
    --output-root "${SLIP_OUTPUT_ROOT}"
    --suite "${SLIP_SUITE_NAME}"
    --task red_block
    --task blue_block
    --task red_pepper
    --seed 50000
    --seed 50001
    --seed 50002
    --hold-seconds 5
  )
  if [[ -n "${SLIP_PREREQUISITE}" ]]; then
    RUNNER+=(--prerequisite-root "${SLIP_PREREQUISITE}")
  fi
  "${RUNNER[@]}"

  ANALYZER=(
    "${PYTHON}"
    sim_mujoco/scripts/analyze_scripted_gripper_slip_experiments.py
    --suite-root "${SLIP_OUTPUT_ROOT}"
    --output-dir "${SLIP_OUTPUT_ROOT}/analysis"
  )
  if [[ -n "${SLIP_PREREQUISITE}" ]]; then
    ANALYZER+=(--comparison-root "${SLIP_PREREQUISITE}")
  fi
  "${ANALYZER[@]}"
fi

test -s "${SLIP_OUTPUT_ROOT}/results.json"
if [[ "${SLIP_SUITE_NAME}" != realism && "${SLIP_SUITE_NAME}" != geometry_realism ]]; then
  test -s "${SLIP_OUTPUT_ROOT}/analysis/summary.csv"
fi
if [[ "${SLIP_SUITE_NAME}" == contact ]]; then
  test -s "${SLIP_OUTPUT_ROOT}/analysis/validation.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/contact_model_summary.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/contact_model_summary.csv"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/contact_model_by_object_command.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/contact_model_by_object_command.csv"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/metric_definitions.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/plot_manifest.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/plots/relative_vertical_displacement_overlay.png"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/plots/relative_slip_velocity_overlay.png"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/plots/actuator_force_overlay.png"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/plots/contact_state_force_overlay.png"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/plots/condition_total_relative_slip.png"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/plots/condition_time_to_drop.png"
fi
if [[ "${SLIP_SUITE_NAME}" == realism ]]; then
  test -s "${SLIP_OUTPUT_ROOT}/analysis/validation.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/trial_metrics.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/trial_metrics.csv"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/condition_protocol_summary.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/paired_deltas.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/metric_definitions.json"
fi
if [[ "${SLIP_SUITE_NAME}" == geometry_realism ]]; then
  test -s "${SLIP_OUTPUT_ROOT}/models/diagnostic_split_pad.xml"
  test -s "${SLIP_OUTPUT_ROOT}/geometry_model_validation.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/validation.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/trial_metrics.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/trial_metrics.csv"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/condition_protocol_summary.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/paired_deltas.json"
  test -s "${SLIP_OUTPUT_ROOT}/analysis/metric_definitions.json"
fi
echo "Validated scripted diagnostic output: ${SLIP_OUTPUT_ROOT}"
date
