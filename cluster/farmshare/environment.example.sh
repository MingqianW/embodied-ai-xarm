# Source this file after setting these four absolute deployment paths.
# Keep machine-specific values outside the source checkout.
: "${XARM_REPOSITORY:?Set the source checkout path}"
: "${XARM_WORK_ROOT:?Set an external runtime root}"
: "${OPENPI_ROOT:?Set the external pinned OpenPI checkout path}"
: "${XARM_PYTHON:?Set the external environment interpreter path}"
: "${XARM_SLURM_ACCOUNT:?Set the account verified by sacctmgr}"

export XARM_REPOSITORY XARM_WORK_ROOT OPENPI_ROOT XARM_PYTHON XARM_SLURM_ACCOUNT
export XARM_SLURM_PARTITION=gpu
export XARM_SLURM_QOS=gpu
export XARM_SLURM_RESOURCE_CONFIG="$XARM_REPOSITORY/cluster/farmshare/resources.json"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XARM_OUTPUT_PERMISSION_POLICY=owner_only
umask 077
