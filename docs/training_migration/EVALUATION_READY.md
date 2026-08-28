# π0.5 xArm Final Checkpoint: Evaluation Ready

The verified path is the durable one-GH200 Slurm evaluation stage. This command allocates
one GH200, 16 CPUs, 220G RAM, and one node for at most 12 hours:

```bash
cd /u/mw89/repos/openpi
EVAL_JOB=$(sbatch --parsable --export=ALL,PIPELINE_STAGE=evaluate /u/mw89/repos/openpi/slurm/train_pi05_xarm_real_sim_50_50_continue.sbatch)
echo "$EVAL_JOB"
```

The job selects the existing environment with the absolute interpreter
`/u/mw89/repos/openpi/.venv/bin/python`, exports all caches and outputs under
`/work/nvme/bfmk/mw89`, and executes the following same-node sequence:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/work/nvme/bfmk/mw89/hf_cache
export WANDB_MODE=online
PYTHON=/u/mw89/repos/openpi/.venv/bin/python
cd /u/mw89/repos/openpi
/u/mw89/repos/openpi/.venv/bin/python scripts/serve_policy.py --host=127.0.0.1 --port=8000 policy:checkpoint --policy.config=pi05_xarm_real_sim_50_50_continue --policy.dir=/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/xarm_pi05_real50_sim50_ep198_continue_30000_to_50000_v1/50000 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
READY=false
for _ in $(seq 1 120); do
  if "$PYTHON" -c 'import socket; s=socket.create_connection(("127.0.0.1",8000),2); s.close()' 2>/dev/null; then
    READY=true
    break
  fi
  sleep 5
done
test "$READY" = true
cd /u/mw89/repos/embodied-ai-xarm
/u/mw89/repos/openpi/.venv/bin/python sim_mujoco/scripts/evaluate_remote_policy_automatic.py --policy-label final_50000 --episodes 20 --seed-start 50000 --host 127.0.0.1 --port 8000 --max-policy-steps 80 --output-dir /work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/final_50000 --video-every 5 --resume
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT
```

The policy server binds only to `127.0.0.1:8000`; no SSH tunnel is used.

Final checkpoint: `/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/xarm_pi05_real50_sim50_ep198_continue_30000_to_50000_v1/50000`
Metrics/videos: `/work/nvme/bfmk/mw89/mujoco_outputs/policy_evaluation/{baseline_30000,final_50000}`
