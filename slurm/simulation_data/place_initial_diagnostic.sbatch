#!/bin/bash
#SBATCH --job-name=xarm-v3-place-reset
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3/slurm/%x-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3/slurm/%x-%j.err

source /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/common.sh
PHASE=place_initial_diagnostic
NEXT_ACTION="Inspect settling sensitivity for the canonical Place grasp; do not change the 5 mm validation threshold to improve yield."
RESUME_COMMAND="sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/place_initial_diagnostic.sbatch"

cd "$REPOSITORY"
print_context

diagnostic_root="$LOG_ROOT/diagnostics/place_initial_${SLURM_JOB_ID}"
mkdir -p "$diagnostic_root"
for settle_steps in 500 750 1000 1250 1500 2000; do
  "$PYTHON" sim_mujoco/scripts/diagnose_place_initial_grasp.py \
    --config "$CONFIG" \
    --seed 600000 \
    --settle-steps "$settle_steps" \
    --gripper-half-width-m 0.012 \
    --tcp-to-pepper-z-m -0.030 \
    --output "$diagnostic_root/settle_${settle_steps}.json"
done

"$PYTHON" - "$diagnostic_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("settle_*.json")):
    value = json.loads(path.read_text(encoding="utf-8"))
    result = value["result"]
    rows.append({
        "settle_steps": value["settle_steps"],
        "settle_duration_s": value["settle_duration_s"],
        "success": result["initial_grasp_success"],
        "failure_reason": result["initial_grasp_failure_reason"],
        "maximum_relative_drift_m": result["initial_grasp_max_relative_drift_m"],
        "minimum_gripper_contacts": result[
            "minimum_initial_grasp_gripper_contact_count"
        ],
    })
(root / "summary.json").write_text(
    json.dumps({"results": rows}, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(rows, indent=2))
if not any(row["success"] for row in rows):
    raise SystemExit("No settling duration passed the canonical Place validator")
PY

COMPLETED_WORK="Canonical Place initial-grasp settling diagnostic completed; results are in $diagnostic_root."
