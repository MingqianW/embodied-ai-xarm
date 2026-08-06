#!/bin/bash
#SBATCH --job-name=xarm-v4-10x-smoke
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real/slurm/%x-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real/slurm/%x-%j.err

source /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/common_v4_10x.sh
PHASE=smoke
NEXT_ACTION="Inspect SMOKE_AUDIT.md and every contact sheet; submit full_generation.sbatch only after visual review passes."
RESUME_COMMAND="sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/smoke_v4_10x.sbatch"

cd "$REPOSITORY"
print_context

"$PYTHON" scripts/check_deltaai_mujoco_environment.py \
  --require-egl \
  --json-output "$LOG_ROOT/DELTAI_MUJOCO_ENVIRONMENT.json"

"$PYTHON" -m sim_mujoco.data_generation.cli generate \
  --config "$CONFIG" \
  --output "$SMOKE_OUTPUT" \
  --smoke \
  --overwrite

"$PYTHON" -m sim_mujoco.data_generation.cli audit \
  --config "$CONFIG" \
  --raw "$SMOKE_OUTPUT" \
  --report-dir "$LOG_ROOT" \
  --decode-all-images \
  --smoke

"$PYTHON" -m sim_mujoco.data_generation.cli permissions \
  --config "$CONFIG" \
  "$SMOKE_OUTPUT" "$LOG_ROOT"

COMPLETED_WORK="Six-task clean smoke generation and automated smoke audit passed; contact sheets require explicit visual review."
