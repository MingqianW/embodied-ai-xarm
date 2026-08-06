#!/bin/bash
#SBATCH --job-name=xarm-v3-audit
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3/slurm/%x-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3/slurm/%x-%j.err

source /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/common.sh
PHASE=final_audit
NEXT_ACTION="Review DATASET_AUDIT.md and FINAL_HANDOFF.md before starting any separate training task."
RESUME_COMMAND="sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/final_audit.sbatch"

cd "$REPOSITORY"
print_context

"$PYTHON" -m sim_mujoco.data_generation.cli audit \
  --config "$CONFIG" \
  --raw "$RAW_OUTPUT" \
  --converted "$CONVERTED_OUTPUT" \
  --report-dir "$LOG_ROOT" \
  --decode-all-images

"$PYTHON" -m sim_mujoco.data_generation.cli permissions \
  --config "$CONFIG" \
  "$RAW_OUTPUT" "$CONVERTED_OUTPUT" "$SMOKE_OUTPUT" "$LOG_ROOT"

{
  namei -l "$RAW_OUTPUT"
  namei -l "$CONVERTED_OUTPUT"
  getfacl "$RAW_OUTPUT"
  getfacl "$CONVERTED_OUTPUT"
} 2>&1 | tee "$LOG_ROOT/PERMISSIONS_REPORT.txt"

"$PYTHON" -m sim_mujoco.data_generation.cli handoff \
  --config "$CONFIG"

COMPLETED_WORK="Full raw and converted decode/integrity audits passed and group permissions were applied."
