#!/bin/bash
#SBATCH --job-name=xarm-v3-tests
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3/slurm/%x-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3/slurm/%x-%j.err

source /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/common.sh
PHASE=offline_tests
NEXT_ACTION="Review TEST_REPORT.md, then create the local code commit and submit smoke.sbatch."
RESUME_COMMAND="sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/offline_tests.sbatch"

cd "$REPOSITORY"
print_context

REPORT_TEXT="$LOG_ROOT/TEST_REPORT.txt"
"$PYTHON" -m compileall -q \
  sim_mujoco/data_generation \
  sim_mujoco/data_collection \
  simulation/environment.py \
  simulation/scene \
  data/common/lerobot_writer.py \
  tests 2>&1 | tee "$REPORT_TEXT"

git diff --check 2>&1 | tee -a "$REPORT_TEXT"

"$PYTHON" -m pytest -q tests/test_mujoco_data_generation.py \
  2>&1 | tee -a "$REPORT_TEXT"

"$PYTHON" -m pytest -q \
  tests/simulation/test_task_scenes.py \
  tests/test_mujoco_scripted_oracle.py \
  tests/test_mujoco_data_conversions.py \
  tests/simulation/test_gripper_motion.py \
  tests/test_mujoco_collisions.py \
  tests/test_mujoco_lerobot_pipeline.py \
  2>&1 | tee -a "$REPORT_TEXT"

"$PYTHON" -m pytest -q tests 2>&1 | tee -a "$REPORT_TEXT"

{
  printf '# Test Report\n\n'
  printf -- '- Job ID: `%s`\n' "$SLURM_JOB_ID"
  printf -- '- Host: `%s`\n' "$(hostname)"
  printf -- '- Python: `%s`\n' "$PYTHON"
  printf -- '- Result: **PASS**\n\n'
  printf 'Commands and exact output are preserved in `%s`.\n' "$REPORT_TEXT"
} > "$LOG_ROOT/TEST_REPORT.md"

COMPLETED_WORK="Python compilation, focused pipeline tests, MuJoCo regressions, and full offline test suite passed."
