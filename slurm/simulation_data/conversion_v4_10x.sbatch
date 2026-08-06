#!/bin/bash
#SBATCH --job-name=xarm-v4-10x-convert
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real/slurm/%x-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real/slurm/%x-%j.err

source /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/common_v4_10x.sh
PHASE=conversion
NEXT_ACTION="Submit final_audit.sbatch after conversion metadata reports 1980 episodes."
RESUME_COMMAND="sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/conversion_v4_10x.sbatch"

cd "$REPOSITORY"
print_context

"$PYTHON" -m sim_mujoco.data_generation.cli convert \
  --config "$CONFIG" \
  --raw "$RAW_OUTPUT" \
  --output "$CONVERTED_OUTPUT" \
  --overwrite

"$PYTHON" -m sim_mujoco.data_generation.cli permissions \
  --config "$CONFIG" \
  "$CONVERTED_OUTPUT" "$LOG_ROOT"

COMPLETED_WORK="Canonical LeRobot/OpenPI conversion completed with 1980 accepted episodes and six canonical tasks."
