#!/bin/bash
#SBATCH --job-name=xarm-v4-10x-generate
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real/slurm/%x-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real/slurm/%x-%j.err

source /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/common_v4_10x.sh
PHASE=full_generation
NEXT_ACTION="Confirm collection_summary.json complete=true, then submit conversion.sbatch."
RESUME_COMMAND="sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/full_generation_v4_10x.sbatch"

cd "$REPOSITORY"
print_context

"$PYTHON" -m sim_mujoco.data_generation.cli generate \
  --config "$CONFIG" \
  --output "$RAW_OUTPUT" \
  --overwrite

"$PYTHON" -m sim_mujoco.data_generation.cli audit \
  --config "$CONFIG" \
  --raw "$RAW_OUTPUT" \
  --report-dir "$LOG_ROOT" \
  --decode-all-images

"$PYTHON" -m sim_mujoco.data_generation.cli permissions \
  --config "$CONFIG" \
  "$RAW_OUTPUT" "$LOG_ROOT"

COMPLETED_WORK="Exactly 1980 accepted clean episodes generated and the mandatory raw audit passed."
