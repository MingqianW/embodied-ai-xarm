#!/bin/bash
# One isolated xArm slip-diagnostic episode. Submission requires explicit approval.
# Required: OUTPUT_ROOT. Optional: MODEL_SPEC, TASK, SEED, EXECUTE_CHUNK_STEPS.
#SBATCH --job-name=xarm-slip-diag
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/openpi_real_sim/xarm-slip-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/openpi_real_sim/xarm-slip-%j.err

set -euo pipefail

: "${OUTPUT_ROOT:?Set a new isolated OUTPUT_ROOT under /work/nvme/bfmk/mw89.}"

OPENPI_ROOT=/u/mw89/repos/openpi
EMBODIED_AI_ROOT=/u/mw89/repos/embodied-ai-xarm
PYTHON="$OPENPI_ROOT/.venv/bin/python"
MODEL_SPEC="${MODEL_SPEC:-$EMBODIED_AI_ROOT/sim_mujoco/config/formal_models/B.json}"
PROTOCOL="${PROTOCOL:-$EMBODIED_AI_ROOT/sim_mujoco/config/formal_xarm_pi05_eval_v1.json}"
TASK="${TASK:-red_block}"
SEED="${SEED:-50000}"
EXECUTE_CHUNK_STEPS="${EXECUTE_CHUNK_STEPS:-5}"
POST_SUCCESS_SECONDS="${POST_SUCCESS_SECONDS:-2.0}"
DIAGNOSTIC_LATCH_RAW="${DIAGNOSTIC_LATCH_RAW:-}"
FRICTION_VARIANT_MANIFEST="${FRICTION_VARIANT_MANIFEST:-}"
LOG_ROOT=/work/nvme/bfmk/mw89/logs/xarm_slip_diagnostics

export OPENPI_ROOT EMBODIED_AI_ROOT
export XARM_SLIP_TRACE=1
export XARM_SLIP_TRACE_POST_SUCCESS_SECONDS="$POST_SUCCESS_SECONDS"
export HF_LEROBOT_HOME=/work/nvme/bfmk/mw89/datasets/lerobot
export HF_HOME=/work/nvme/bfmk/mw89/caches/huggingface
export HF_DATASETS_CACHE=/work/nvme/bfmk/mw89/caches/huggingface/datasets
export HF_HUB_CACHE=/work/nvme/bfmk/mw89/caches/huggingface/hub
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export UV_CACHE_DIR=/work/nvme/bfmk/mw89/caches/uv
export XDG_CACHE_HOME=/work/nvme/bfmk/mw89/caches/xdg
export JAX_COMPILATION_CACHE_DIR=/work/nvme/bfmk/mw89/caches/jax
export WANDB_DIR=/work/nvme/bfmk/mw89/wandb
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

test -x "$PYTHON"
test -f "$MODEL_SPEC"
test -f "$PROTOCOL"
case "$OUTPUT_ROOT" in
  /work/nvme/bfmk/mw89/*) ;;
  *) echo "OUTPUT_ROOT must be under /work/nvme/bfmk/mw89: $OUTPUT_ROOT" >&2; exit 2 ;;
esac
if test -e "$OUTPUT_ROOT" && test -n "$(ls -A "$OUTPUT_ROOT" 2>/dev/null)"; then
  echo "Refusing non-empty diagnostic output root: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" \
  "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$WANDB_DIR"

JOB_LOG="$LOG_ROOT/${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG"
SERVER_PROVENANCE="$JOB_LOG/server_provenance.json"

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
test -n "${SLURM_JOB_ID:-}"
scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | head -n 40
echo "MODEL_SPEC=$MODEL_SPEC"
echo "PROTOCOL=$PROTOCOL"
echo "TASK=$TASK"
echo "SEED=$SEED"
echo "EXECUTE_CHUNK_STEPS=$EXECUTE_CHUNK_STEPS"
echo "POST_SUCCESS_SECONDS=$POST_SUCCESS_SECONDS"
echo "DIAGNOSTIC_LATCH_RAW=$DIAGNOSTIC_LATCH_RAW"
echo "FRICTION_VARIANT_MANIFEST=$FRICTION_VARIANT_MANIFEST"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "OPENPI_COMMIT=$(git -C "$OPENPI_ROOT" rev-parse HEAD)"
echo "EMBODIED_AI_COMMIT=$(git -C "$EMBODIED_AI_ROOT" rev-parse HEAD)"
nvidia-smi

DIAGNOSTIC_ARGS=(
  --model-spec "$MODEL_SPEC"
  --protocol "$PROTOCOL"
  --output-root "$OUTPUT_ROOT"
  --task "$TASK"
  --seed "$SEED"
  --execute-chunk-steps "$EXECUTE_CHUNK_STEPS"
  --post-success-seconds "$POST_SUCCESS_SECONDS"
  --openpi-root "$OPENPI_ROOT"
  --embodied-ai-root "$EMBODIED_AI_ROOT"
)
if test -n "$DIAGNOSTIC_LATCH_RAW"; then
  DIAGNOSTIC_ARGS+=(--gripper-latch-raw "$DIAGNOSTIC_LATCH_RAW")
fi
if test -n "$FRICTION_VARIANT_MANIFEST"; then
  test -f "$FRICTION_VARIANT_MANIFEST"
  DIAGNOSTIC_ARGS+=(--friction-variant-manifest "$FRICTION_VARIANT_MANIFEST")
fi

cd "$EMBODIED_AI_ROOT"
"$PYTHON" sim_mujoco/scripts/diagnose_xarm_slip.py \
  "${DIAGNOSTIC_ARGS[@]}" \
  --prepare-server-provenance "$SERVER_PROVENANCE"

PORT=$(
  "$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
)
echo "POLICY_PORT=$PORT"

SERVER_PID=""
cleanup() {
  if test -n "$SERVER_PID"; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "$OPENPI_ROOT"
"$PYTHON" scripts/serve_policy.py \
  --host=127.0.0.1 \
  --port="$PORT" \
  --require-request-rng \
  --provenance-json "$SERVER_PROVENANCE" \
  policy:checkpoint \
  --policy.config="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_config"])' "$MODEL_SPEC")" \
  --policy.dir="$("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(str(p["checkpoint_root"]) + "/" + str(p["manager_step"]))' "$MODEL_SPEC")" \
  >"$JOB_LOG/policy_server.out" 2>"$JOB_LOG/policy_server.err" &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server exited before readiness; see $JOB_LOG/policy_server.err" >&2
    exit 1
  fi
  if "$PYTHON" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
    "http://127.0.0.1:$PORT/healthz" 2>/dev/null; then
    break
  fi
  sleep 2
done
"$PYTHON" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
  "http://127.0.0.1:$PORT/healthz"

cd "$EMBODIED_AI_ROOT"
"$PYTHON" sim_mujoco/scripts/diagnose_xarm_slip.py \
  "${DIAGNOSTIC_ARGS[@]}" \
  --host 127.0.0.1 \
  --port "$PORT"

echo "Slip diagnostic complete: $OUTPUT_ROOT"
