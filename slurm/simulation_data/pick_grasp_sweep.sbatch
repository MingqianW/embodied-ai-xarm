#!/bin/bash
#SBATCH --job-name=xarm-v3-grasp-sweep
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
PHASE=pick_grasp_diagnostic
NEXT_ACTION="Inspect the canonical 20-seed red-pepper validation summary; rerun smoke if the physical-grasp configuration changes."
RESUME_COMMAND="sbatch /u/mw89/repos/embodied-ai-xarm/slurm/simulation_data/pick_grasp_sweep.sbatch"

cd "$REPOSITORY"
print_context

diagnostic_root="$LOG_ROOT/diagnostics/pick_grasp_sweep_${SLURM_JOB_ID}"
mkdir -p "$diagnostic_root"

# Revalidate the canonical physical grasp across the deterministic retry-seed
# sequence. Failed episodes remain diagnostic evidence and never enter data.
candidates=(
  "red_pepper 100000 250 -0.022"
)
for candidate in "${candidates[@]}"; do
  read -r task seed_start closed_raw tcp_offset <<<"$candidate"
  label="${task}_raw_${closed_raw}_offset_${tcp_offset}"
  candidate_dir="$diagnostic_root/$label"
  set +e
  "$PYTHON" sim_mujoco/scripts/test_scripted_oracle.py \
    --task "$task" \
    --episodes 20 \
    --seed-start "$seed_start" \
    --seed-stride 1000 \
    --object-xy-range 0.02 \
    --object-yaw-range-deg 10 \
    --joint-noise 0.005 \
    --closed-gripper-raw "$closed_raw" \
    --grasp-tcp-offset-from-object-m "$tcp_offset" \
    --output-dir "$candidate_dir"
  candidate_exit=$?
  set -e
  printf '%s\n' "$candidate_exit" >"$candidate_dir/exit_code.txt"
done

"$PYTHON" - "$diagnostic_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("*/summary.json")):
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append({
        "candidate": summary_path.parent.name,
        "task": value["task"],
        "successes": value["successes"],
        "episodes": value["episodes"],
        "success_rate": value["success_rate"],
        "closed_gripper_raw": value["closed_gripper_raw"],
        "grasp_tcp_offset_from_object_m": value[
            "grasp_tcp_offset_from_object_m"
        ],
    })
(root / "sweep_summary.json").write_text(
    json.dumps({"candidates": rows}, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(rows, indent=2))
tasks = {row["task"] for row in rows}
for task in tasks:
    best = max(row["success_rate"] for row in rows if row["task"] == task)
    if best < 0.5:
        raise SystemExit(f"No {task} candidate passed half the diagnostic seeds")
PY

COMPLETED_WORK="Canonical 20-seed Pick physical-grasp validation completed; results are in $diagnostic_root."
