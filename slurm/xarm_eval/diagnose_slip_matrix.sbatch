#!/usr/bin/env bash
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=xarm-slip-matrix
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --gpus=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/xarm_slip_matrix_%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/xarm_slip_matrix_%j.err

set -euo pipefail

: "${OUTPUT_BASE:?Set a new OUTPUT_BASE under /work/nvme/bfmk/mw89.}"

OPENPI_ROOT=/u/mw89/repos/openpi
EMBODIED_AI_ROOT=/u/mw89/repos/embodied-ai-xarm
PYTHON=${OPENPI_ROOT}/.venv/bin/python
MODEL_SPEC=${MODEL_SPEC:-${EMBODIED_AI_ROOT}/configs/evaluation/sim/models/B.json}
PROTOCOL=${PROTOCOL:-${EMBODIED_AI_ROOT}/configs/evaluation/sim/protocols/formal_xarm_pi05_eval_v1.json}
TASK=${TASK:-red_block}
XARM_SLIP_SEEDS_TEXT=${XARM_SLIP_SEEDS:-50000 50001 50002}
read -r -a POLICY_SEEDS <<< "${XARM_SLIP_SEEDS_TEXT}"
POST_SUCCESS_SECONDS=${POST_SUCCESS_SECONDS:-2.0}
LATCH_RAW=${LATCH_RAW:-50}
PERSISTENT_ROOT=/work/nvme/bfmk/mw89
JOB_LOG=${PERSISTENT_ROOT}/logs/xarm_slip_diagnostics/${SLURM_JOB_ID}

case "${OUTPUT_BASE}" in
  /work/nvme/bfmk/mw89/*) ;;
  *) echo "OUTPUT_BASE must be under /work/nvme/bfmk/mw89: ${OUTPUT_BASE}" >&2; exit 2 ;;
esac
if [[ -e "${OUTPUT_BASE}" ]]; then
  echo "Refusing existing output base: ${OUTPUT_BASE}" >&2
  exit 3
fi
mkdir -p "${OUTPUT_BASE}" "${JOB_LOG}"

export OPENPI_ROOT EMBODIED_AI_ROOT
export XARM_SLIP_TRACE=1
export XARM_SLIP_TRACE_POST_SUCCESS_SECONDS=${POST_SUCCESS_SECONDS}
export HF_LEROBOT_HOME=${PERSISTENT_ROOT}/datasets/lerobot
export HF_HOME=${PERSISTENT_ROOT}/caches/huggingface
export HF_DATASETS_CACHE=${PERSISTENT_ROOT}/caches/huggingface/datasets
export HF_HUB_CACHE=${PERSISTENT_ROOT}/caches/huggingface/hub
export HUGGINGFACE_HUB_CACHE=${HF_HUB_CACHE}
export UV_CACHE_DIR=${PERSISTENT_ROOT}/caches/uv
export XDG_CACHE_HOME=${PERSISTENT_ROOT}/caches/xdg
export JAX_COMPILATION_CACHE_DIR=${PERSISTENT_ROOT}/caches/jax
export WANDB_DIR=${PERSISTENT_ROOT}/wandb
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" \
  "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${JAX_COMPILATION_CACHE_DIR}" "${WANDB_DIR}"
test -x "${PYTHON}"
test -f "${MODEL_SPEC}"
test -f "${PROTOCOL}"
if [[ ${#POLICY_SEEDS[@]} -lt 1 ]]; then
  echo "XARM_SLIP_SEEDS must contain at least one seed" >&2
  exit 2
fi
for POLICY_SEED in "${POLICY_SEEDS[@]}"; do
  if [[ ! "${POLICY_SEED}" =~ ^[0-9]+$ ]]; then
    echo "Invalid evaluation seed: ${POLICY_SEED}" >&2
    exit 2
  fi
done

date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
test -n "${SLURM_JOB_ID:-}"
scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | head -n 40
echo "OUTPUT_BASE=${OUTPUT_BASE}"
echo "MODEL_SPEC=${MODEL_SPEC}"
echo "PROTOCOL=${PROTOCOL}"
echo "TASK=${TASK}"
echo "POLICY_SEEDS=${POLICY_SEEDS[*]}"
echo "POST_SUCCESS_SECONDS=${POST_SUCCESS_SECONDS}"
echo "LATCH_RAW=${LATCH_RAW}"
cd "${EMBODIED_AI_ROOT}"
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module list 2>&1 || true
nvidia-smi

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup EXIT

for CHUNK_STEPS in 1 2 5; do
  for POLICY_SEED in "${POLICY_SEEDS[@]}"; do
    CHUNK_LOG=${JOB_LOG}/c${CHUNK_STEPS}_seed${POLICY_SEED}
    mkdir -p "${CHUNK_LOG}"
    SERVER_PROVENANCE=${CHUNK_LOG}/server_provenance.json
    NORMAL_OUTPUT=${OUTPUT_BASE}/c${CHUNK_STEPS}_seed${POLICY_SEED}_normal
    LATCH_OUTPUT=${OUTPUT_BASE}/c${CHUNK_STEPS}_seed${POLICY_SEED}_latch_raw${LATCH_RAW}
    COMMON_ARGS=(
      --model-spec "${MODEL_SPEC}"
      --protocol "${PROTOCOL}"
      --task "${TASK}"
      --seed "${POLICY_SEED}"
      --execute-chunk-steps "${CHUNK_STEPS}"
      --post-success-seconds "${POST_SUCCESS_SECONDS}"
      --openpi-root "${OPENPI_ROOT}"
      --embodied-ai-root "${EMBODIED_AI_ROOT}"
    )
    if [[ "${POLICY_SEED}" != "${POLICY_SEEDS[0]}" ]]; then
      COMMON_ARGS+=(--no-video)
    fi

    "${PYTHON}" sim_mujoco/scripts/diagnose_xarm_slip.py \
      "${COMMON_ARGS[@]}" \
      --output-root "${NORMAL_OUTPUT}" \
      --prepare-server-provenance "${SERVER_PROVENANCE}"

    PORT=$("${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
    echo "c${CHUNK_STEPS} seed${POLICY_SEED} POLICY_PORT=${PORT}"
    cd "${OPENPI_ROOT}"
    "${PYTHON}" scripts/serve_policy.py \
      --host=127.0.0.1 \
      --port="${PORT}" \
      --require-request-rng \
      --provenance-json "${SERVER_PROVENANCE}" \
      policy:checkpoint \
      --policy.config="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_config"])' "${MODEL_SPEC}")" \
      --policy.dir="$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(str(p["checkpoint_root"]) + "/" + str(p["manager_step"]))' "${MODEL_SPEC}")" \
      >"${CHUNK_LOG}/policy_server.out" 2>"${CHUNK_LOG}/policy_server.err" &
    SERVER_PID=$!

    for _ in $(seq 1 120); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Policy server exited before readiness; see ${CHUNK_LOG}/policy_server.err" >&2
        exit 1
      fi
      if "${PYTHON}" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
        "http://127.0.0.1:${PORT}/healthz" 2>/dev/null; then
        break
      fi
      sleep 2
    done
    "${PYTHON}" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
      "http://127.0.0.1:${PORT}/healthz"

    cd "${EMBODIED_AI_ROOT}"
    "${PYTHON}" sim_mujoco/scripts/diagnose_xarm_slip.py \
      "${COMMON_ARGS[@]}" \
      --output-root "${NORMAL_OUTPUT}" \
      --host 127.0.0.1 \
      --port "${PORT}"
    "${PYTHON}" sim_mujoco/scripts/diagnose_xarm_slip.py \
      "${COMMON_ARGS[@]}" \
      --output-root "${LATCH_OUTPUT}" \
      --gripper-latch-raw "${LATCH_RAW}" \
      --host 127.0.0.1 \
      --port "${PORT}"
    test -s "${NORMAL_OUTPUT}/diagnostic_manifest.json"
    test -s "${LATCH_OUTPUT}/diagnostic_manifest.json"
    cleanup
  done
done

TRACE_ARGS=()
for CHUNK_STEPS in 1 2 5; do
  for POLICY_SEED in "${POLICY_SEEDS[@]}"; do
    NORMAL_TRACE=${OUTPUT_BASE}/c${CHUNK_STEPS}_seed${POLICY_SEED}_normal/models/B/tasks/${TASK}/seed_${POLICY_SEED}/slip_trace.csv
    LATCH_TRACE=${OUTPUT_BASE}/c${CHUNK_STEPS}_seed${POLICY_SEED}_latch_raw${LATCH_RAW}/models/B/tasks/${TASK}/seed_${POLICY_SEED}/slip_trace.csv
    test -s "${NORMAL_TRACE}"
    test -s "${LATCH_TRACE}"
    TRACE_ARGS+=(--trace "c${CHUNK_STEPS}_seed${POLICY_SEED}_normal=${NORMAL_TRACE}")
    TRACE_ARGS+=(--trace "c${CHUNK_STEPS}_seed${POLICY_SEED}_latch=${LATCH_TRACE}")
  done
done
"${PYTHON}" sim_mujoco/scripts/analyze_xarm_slip_trace.py \
  "${TRACE_ARGS[@]}" \
  --output-dir "${OUTPUT_BASE}/analysis"
test -s "${OUTPUT_BASE}/analysis/slip_analysis.json"

DETAILED_ARGS=(
  --output-base "${OUTPUT_BASE}"
  --output-dir "${OUTPUT_BASE}/detailed_analysis"
  --task "${TASK}"
  --model-id B
  --latch-raw "${LATCH_RAW}"
)
for POLICY_SEED in "${POLICY_SEEDS[@]}"; do
  DETAILED_ARGS+=(--seed "${POLICY_SEED}")
done
"${PYTHON}" sim_mujoco/scripts/analyze_policy_gripper_slip_matrix.py \
  "${DETAILED_ARGS[@]}"
test -s "${OUTPUT_BASE}/detailed_analysis/results.json"
test -s "${OUTPUT_BASE}/detailed_analysis/trace_summary.csv"

echo "Completed c1/c2/c5 normal/latch matrix: ${OUTPUT_BASE}"
date
