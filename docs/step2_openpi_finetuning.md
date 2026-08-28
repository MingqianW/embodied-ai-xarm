# Step 2 OpenPI Data Preparation and Fine-Tuning

This project follows the current OpenPI workflow: collect robot demonstrations in a simple raw format, convert them to a LeRobot dataset, compute OpenPI normalization statistics, then fine-tune `pi05_base`.

OpenPI's README describes the same three stages for custom data: convert to LeRobot, define training configs, and run training after `scripts/compute_norm_stats.py`. It also lists `gs://openpi-assets/checkpoints/pi05_base` as the pi0.5 base checkpoint and notes that LoRA fine-tuning needs about 22.5 GB of GPU memory.

## Raw xArm Data Format

Collect episodes into one HDF5 file. The current converter expects:

```text
data/
  demo_000001/
    attrs:
      task_name: string
      language_instruction: string
      success: bool
      failure_reason: string, optional
      operator_id: string, optional
      calibration_id: string, optional
    timestamps: float64[T]
    joint_positions: float32[T, 6]
    end_effector_pose: float32[T, 6]
      # [x, y, z, roll, pitch, yaw], meters and radians
    gripper: float32[T]
      # normalized gripper state, 0=open and 1=closed
    actions: float32[T, 7]
      # [dx, dy, dz, droll, dpitch, dyaw, gripper]
      # deltas are meters/radians; gripper is 0=open and 1=closed
    observations/
      fixed_camera_rgb: uint8[T, H, W, 3]
      wrist_camera_rgb: uint8[T, H, W, 3]
```

Keep every time-indexed array at the same length `T`. Use a fixed control rate, preferably 10 Hz for the first dataset. The images may be collected as BGR if using OpenCV/RealSense; the converter defaults to BGR and writes RGB LeRobot images.

## Data Collection Checklist

1. Collect 5-10 pilot episodes for one task.
2. Run `python scripts/validate_dataset.py --input data/raw/pick_place.hdf5`.
3. Manually replay or visualize a few episodes and confirm image/action sync.
4. Label each episode with `success=true/false`; remove or separate severe failures before the first fine-tune.
5. Scale to balanced batches across objects, positions, lighting, and task language.

Do not mix incompatible action conventions. For this repo, actions are delta end-effector commands with an absolute normalized gripper command.

## Convert HDF5 to LeRobot

Clone and set up OpenPI on a Linux machine with an NVIDIA GPU:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

From the OpenPI environment, run this project's converter:

```bash
uv run python /path/to/embodied-ai-xarm/scripts/prepare_openpi_lerobot.py \
  --input /path/to/embodied-ai-xarm/data/raw/pick_place.hdf5 \
  --repo-id your_hf_username/xarm_pick_place \
  --robot-type xarm6 \
  --fps 10 \
  --overwrite
```

The LeRobot dataset is saved under `$HF_LEROBOT_HOME/your_hf_username/xarm_pick_place`.

## Convert and Upload the Current xArm Raw Folder from WSL

For the raw folder layout under `fine_tune/data/xarm_pi05_data/raw`, keep all paths POSIX-style when running inside WSL. Do not use a Windows path such as `D:\...`; if a quoted Windows path ends in `\`, Bash can swallow the following flags into the same argument.

```bash
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/lerobot_cache}"

python -m data.real.conversion.convert_xarm_raw_to_lerobot \
  --raw-root "fine_tune/data/xarm_pi05_data/raw" \
  --output-dir "converted/xarm_pi05_light" \
  --repo-id local/xarm_pi05_data \
  --robot-type xarm6 \
  --fps 10 \
  --overwrite \
  --skip-light-image-copy \
  --skip-hf-dataset
```

The real LeRobot dataset is written to:

```text
$HF_LEROBOT_HOME/local/xarm_pi05_data
```

Upload that directory, not a Colab-only path like `/content/drive/...`:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
import os

local_repo_id = "local/xarm_pi05_data"
hub_repo_id = "MingqianW/xarm_pi05_data"
folder = Path(os.environ.get("HF_LEROBOT_HOME", "~/lerobot_cache")).expanduser() / local_repo_id

if not folder.exists():
    raise SystemExit(f"Missing converted LeRobot dataset: {folder}")

api = HfApi()
api.create_repo(repo_id=hub_repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(
    folder_path=str(folder),
    repo_id=hub_repo_id,
    repo_type="dataset",
    commit_message="Upload converted xArm LeRobot dataset",
)
PY
```

## Add the xArm OpenPI Config

Print the config snippet:

```bash
python /path/to/embodied-ai-xarm/fine_tune/openpi_xarm_config.py
```

Copy the printed `LeRobotXArmDataConfig` class and `TrainConfig(name="pi05_xarm", ...)` entry into OpenPI's `src/openpi/training/config.py`. Set the `repo_id` to the same id used during conversion.

The snippet maps LeRobot keys as follows:

| LeRobot key | OpenPI training key |
|---|---|
| `image` | `observation/image` |
| `wrist_image` | `observation/wrist_image` |
| `state` | `observation/state` |
| `actions` | `actions` |
| task text | `prompt` |

## Fine-Tune pi0.5

Compute normalization statistics first:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_xarm
```

Then start fine-tuning:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_xarm --exp-name=xarm_pi05_lora --overwrite
```

On Windows PowerShell, use:

```powershell
$env:XLA_PYTHON_CLIENT_MEM_FRACTION = "0.9"
uv run scripts/train.py pi05_xarm --exp-name=xarm_pi05_lora --overwrite
```

You can also use the helper script:

```powershell
.\fine_tune\train_openpi_pi05_xarm.ps1 `
  -OpenPiDir C:\path\to\openpi `
  -RawHdf5 C:\path\to\embodied-ai-xarm\data\raw\pick_place.hdf5 `
  -RepoId your_hf_username/xarm_pick_place
```

After training, serve a checkpoint for robot-side inference:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xarm \
  --policy.dir=checkpoints/pi05_xarm/xarm_pi05_lora/30000
```

Your xArm runtime should send observations with the same keys and units used in training: `observation/image`, `observation/wrist_image`, `observation/state`, and `prompt`.
