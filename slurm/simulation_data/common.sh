#!/bin/bash

set -euo pipefail

REPOSITORY=/u/mw89/repos/embodied-ai-xarm
OPENPI_ROOT=/u/mw89/repos/openpi
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
CONFIG=/u/mw89/repos/embodied-ai-xarm/configs/data/sim/generation/clean_multitask_stable_v3.yaml
RAW_OUTPUT=/work/nvme/bfmk/mw89/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v3
CONVERTED_OUTPUT=/work/nvme/bfmk/mw89/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v3
SMOKE_OUTPUT=/work/nvme/bfmk/mw89/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_stable_v3
LOG_ROOT=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3

export REPOSITORY OPENPI_ROOT PYTHON CONFIG
export RAW_OUTPUT CONVERTED_OUTPUT SMOKE_OUTPUT LOG_ROOT
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_LEROBOT_HOME=/work/nvme/bfmk/mw89/mujoco_datasets
export HF_HOME=/work/nvme/bfmk/mw89/caches/huggingface
export HF_HUB_CACHE=/work/nvme/bfmk/mw89/caches/huggingface/hub
export HF_DATASETS_CACHE=/work/nvme/bfmk/mw89/caches/huggingface/datasets
export UV_CACHE_DIR=/work/nvme/bfmk/mw89/caches/uv

PHASE=unknown
NEXT_ACTION="Inspect the phase logs and status JSON."
RESUME_COMMAND=""
COMPLETED_WORK=""

print_context() {
  date --iso-8601=seconds
  hostname
  pwd
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
  printf 'SLURM_NODELIST=%s\n' "${SLURM_NODELIST:-}"
  test -n "${SLURM_JOB_ID:-}"
  scontrol show job "$SLURM_JOB_ID" 2>/dev/null | head -n 40
  nvidia-smi
  printf 'PYTHON=%s\n' "$PYTHON"
  "$PYTHON" --version
  printf 'CONFIG=%s\n' "$CONFIG"
  printf 'RAW_OUTPUT=%s\n' "$RAW_OUTPUT"
  printf 'CONVERTED_OUTPUT=%s\n' "$CONVERTED_OUTPUT"
  printf 'SMOKE_OUTPUT=%s\n' "$SMOKE_OUTPUT"
  printf 'LOG_ROOT=%s\n' "$LOG_ROOT"
  git -C "$REPOSITORY" rev-parse HEAD
  git -C "$REPOSITORY" status --short
  git -C "$OPENPI_ROOT" rev-parse HEAD
  module list 2>&1 || true
}

record_phase_status() {
  local state=$1
  local failure=$2
  local -a arguments
  arguments=(
    -m data.sim.generation.cli status
    --config "$CONFIG"
    --phase "$PHASE"
    --job-id "$SLURM_JOB_ID"
    --job-state "$state"
    --next-action "$NEXT_ACTION"
  )
  if test -n "$COMPLETED_WORK"; then
    arguments+=(--completed-work "$COMPLETED_WORK")
  fi
  if test -n "$failure"; then
    arguments+=(--known-failure "$failure")
  fi
  if test -n "$RESUME_COMMAND"; then
    arguments+=(--resume-command "$RESUME_COMMAND")
  fi
  cd "$REPOSITORY"
  "$PYTHON" "${arguments[@]}"
}

finish_phase() {
  local exit_code=$?
  trap - EXIT
  if test "$exit_code" -eq 0; then
    record_phase_status COMPLETED ""
  else
    record_phase_status FAILED "${PHASE} exited with code ${exit_code}"
  fi
  exit "$exit_code"
}

trap finish_phase EXIT
