# Real-robot evaluation

These commands move a physical xArm. Run them only at the supervised robot
station with a clear workspace and an accessible emergency stop. 

The current improved runner is the improved version: it requires explicit
hardware authorization, validates actions, handles cleanup, and saves rollout
data. The original runner preserves the initial lab behavior for comparison;
it connects with hard-coded settings and has fewer safety and logging checks. 

Note that the original runner is may not be the latest version of the lab's original runner. Please check the `/home/xingyu/pipeline/run_pi_xarm.py` script on the lab computer for the most up-to-date version of the original runner.

## Current improved runner

Set the repository and external runtime
paths on the lab computer:

```bash
export XARM_REPOSITORY=/path/to/embodied-ai-xarm
export OPENPI_ROOT=/home/xingyu/pi_0.5/openpi
export OPENPI_SOURCE_ROOT="$OPENPI_ROOT/src"
export XARM_REAL_WORLD_ROOT=/home/xingyu/robot/xarm-calibrate-hanyang
export XARM_POLICY_CHECKPOINT="$OPENPI_ROOT/checkpoint/25000"
export XARM_ROBOT_IP=192.168.1.209
export XARM_RAW_ROOT=/home/xingyu/xarm_pi05_data/policy_evaluation_raw
export PYTHONPATH="$XARM_REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"

cd "$OPENPI_ROOT"
uv run python -m evaluation.real.run_policy --allow-hardware
```

Replace `/path/to/embodied-ai-xarm` with the repository location on the lab
computer. Keep `XARM_RAW_ROOT` separate from human-demonstration training data.
The runner asks for a prompt, shows the inferred actions, and requires explicit
operator/workspace/emergency-stop confirmation before motion.

## Original runner

The original script is preserved verbatim at
`evaluation/real/legacy/run_pi_original.py`:

```bash
export XARM_REPOSITORY=/path/to/embodied-ai-xarm
cd /home/xingyu/pi_0.5/openpi

uv run python -u "$XARM_REPOSITORY/evaluation/real/legacy/run_pi_original.py"
```

The legacy script is retained only for historical comparison and reproduction.
It immediately connects to the hard-coded robot IP, uses a hard-coded config
and checkpoint.
