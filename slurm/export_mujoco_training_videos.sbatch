#!/bin/bash
#SBATCH --job-name=xarm-train-videos
#SBATCH --account=bfmk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/nvme/bfmk/mw89/logs/mujoco_training_video_export/%x-%j.out
#SBATCH --error=/work/nvme/bfmk/mw89/logs/mujoco_training_video_export/%x-%j.err

set -euo pipefail

REPO=/u/mw89/repos/embodied-ai-xarm
OUTPUT=/work/nvme/bfmk/mw89/exports/mujoco_training_videos/per_task_2episodes_v1
LOG_ROOT=/work/nvme/bfmk/mw89/logs/mujoco_training_video_export

test ! -e "$OUTPUT"
mkdir -p "$LOG_ROOT"
date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
git -C "$REPO" rev-parse HEAD
git -C "$REPO" status --short
nvidia-smi 2>/dev/null || true
cd "$REPO"
/u/mw89/repos/openpi/.venv/bin/python sim_mujoco/scripts/export_lerobot_training_videos.py --output "$OUTPUT" --per-task 2 --fps 10
