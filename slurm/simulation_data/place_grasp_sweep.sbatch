#!/bin/bash
#SBATCH --job-name=xarm-v3-place-grasp
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3/slurm/%x-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3/slurm/%x-%j.err

source /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/common.sh
PHASE=place_grasp_diagnostic
NEXT_ACTION="Inspect the canonical 20-seed Place initial-grasp summary; rerun smoke if the physical initialization changes."
RESUME_COMMAND="sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/place_grasp_sweep.sbatch"

cd "$REPOSITORY"
print_context

diagnostic_root="$LOG_ROOT/diagnostics/place_grasp_sweep_${SLURM_JOB_ID}"
mkdir -p "$diagnostic_root"
for retry_index in $(seq 0 19); do
  seed=$((600000 + retry_index * 1000))
  "$PYTHON" sim_mujoco/scripts/diagnose_place_initial_grasp.py \
    --config "$CONFIG" \
    --seed "$seed" \
    --settle-steps 500 \
    --gripper-half-width-m 0.012 \
    --tcp-to-pepper-z-m -0.030 \
    --output "$diagnostic_root/seed_${seed}.json"
done

"$PYTHON" - "$diagnostic_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("seed_*.json")):
    value = json.loads(path.read_text(encoding="utf-8"))
    result = value["result"]
    rows.append({
        "seed": value["seed"],
        "tcp_to_pepper_z_m": value["tcp_to_pepper_z_m"],
        "gripper_half_width_m": value["gripper_half_width_m"],
        "success": result["initial_grasp_success"],
        "failure_reason": result["initial_grasp_failure_reason"],
        "maximum_relative_drift_m": result["initial_grasp_max_relative_drift_m"],
        "minimum_gripper_contacts": result[
            "minimum_initial_grasp_gripper_contact_count"
        ],
        "final_offset_m": result[
            "final_initial_grasp_pepper_to_tcp_translation_m"
        ],
    })
rows.sort(key=lambda row: row["seed"])
(root / "summary.json").write_text(
    json.dumps({"results": rows}, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(rows, indent=2))
successes = sum(row["success"] for row in rows)
if successes < 18:
    raise SystemExit(
        f"Place grasp passed only {successes}/{len(rows)} retry seeds; required 18"
    )
PY

COMPLETED_WORK="Canonical 20-seed Place physical-grasp validation completed; results are in $diagnostic_root."
