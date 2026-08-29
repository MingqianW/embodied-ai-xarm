#!/bin/bash
set -euo pipefail

test -n "${SLURM_JOB_ID:-}"
REPOSITORY="${XARM_REPOSITORY:-$(cd "$(dirname "$0")/../.." && pwd)}"
: "${XARM_PYTHON:?submit through cluster.cli or explicitly export XARM_PYTHON}"
cd "$REPOSITORY"
exec "$XARM_PYTHON" -m cluster.cli run "$@"
