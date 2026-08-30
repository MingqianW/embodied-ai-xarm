# Real-world data collection

This workflow uses the xArm/RealSense collector at
`/home/xingyu/robot/xarm-calibrate-hanyang` on the lab computer. The collector
is not currently tracked in this repository. Verify it on the lab computer;
it should be integrated into this repository in the future.

## 1. Start the teach pendant

Run the uFactory Studio client:

```bash
cd /home/xingyu/Downloads
chmod +x ufactory-studio-client-linux-1.0.2.AppImage
./ufactory-studio-client-linux-1.0.2.AppImage
```

Use the teach pendant to control the robot and enable manual mode. Reopen
manual mode from the teach pendant before every trial.

## 2. Collect a demonstration

```bash
cd /home/xingyu/pi_0.5/openpi

uv run python -u /home/xingyu/robot/xarm-calibrate-hanyang/real_world/collect_async_gripper_optimized.py \
  --xarm_ip 192.168.1.209 \
  --base_dir /home/xingyu/xarm_pi05_data/raw \
  --task pick_up_light_blue_block \
  --max_realsense 3 \
  --rs_width 640 \
  --rs_height 480 \
  --rs_fps 30 \
  --poll_hz 30 \
  --save_hz 10 \
  --async_writer \
  --gripper_auto_fine_mm 0
```

Change `--task` to select the task. Change `--base_dir` only when the raw-data
output root should be different. The robot IP, camera settings, and sample
rates can also be changed when the lab setup requires it.

Example task names:

```text
pick_up_dark_blue_block
pick_up_light_blue_block
pick_up_red_block
```

## 3. Keyboard controls

These controls come from the historical lab command and must be verified
against the collector currently installed on the lab computer.

| Key | Function |
|---|---|
| `Space` | Start or stop the current recording |
| `n` | Save/proceed to the next demonstration and restore the initial position |
| `a` | Slowly close the gripper |
| `d` | Slowly open the gripper |

The historical workflow also used `x` to drop a bad trial; verify that binding
before relying on it.

## 4. Basic trial workflow

```text
1. Open the teach pendant.
2. Enable or reopen manual mode.
3. Position the robot and scene.
4. Press Space to start recording.
5. Perform the target demonstration.
6. Press Space to stop or continue recording.
7. Press n to save/proceed and restore the initial position.
8. Reopen manual mode before the next trial.
```

## 5. Check, convert, and hand off to training

Set the repository path once, then run the repository commands with the OpenPI
environment:

```bash
export XARM_REPOSITORY=/path/to/embodied-ai-xarm
export OPENPI_ROOT=/home/xingyu/pi_0.5/openpi
export RAW_ROOT=/home/xingyu/xarm_pi05_data/raw
export HF_LEROBOT_HOME=/home/xingyu/xarm_pi05_data/lerobot
export REPO_ID=local/xarm_pi05_20260703
export DATASET_DIR="$HF_LEROBOT_HOME/$REPO_ID"
export LIGHT_OUTPUT=/home/xingyu/xarm_pi05_data/converted/xarm_pi05_20260703
export PYTHONPATH="$XARM_REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"

cd "$OPENPI_ROOT"
```

Replace `/path/to/embodied-ai-xarm` with the repository location on the lab
computer.

### 1. Check the raw episodes

```bash
uv run python -m data.real.validation.check_xarm_raw_quality \
  --raw-root "$RAW_ROOT" \
  --strict
```

Review every reported warning or error. The checker is read-only and does not
exclude episodes automatically. Move rejected episode directories outside
`$RAW_ROOT/<task>/` before conversion; do not delete the only copy.

### 2. Convert the accepted episodes to LeRobot

For the first complete conversion:

```bash
uv run python -m data.real.conversion.convert_xarm_raw_to_lerobot \
  --raw-root "$RAW_ROOT" \
  --output-dir "$LIGHT_OUTPUT" \
  --repo-id "$REPO_ID" \
  --fps 10 \
  --overwrite
```

For later collections, append only episodes not already recorded in the
conversion manifest:

```bash
uv run python -m data.real.conversion.convert_xarm_raw_to_lerobot \
  --raw-root "$RAW_ROOT" \
  --output-dir "$LIGHT_OUTPUT" \
  --repo-id "$REPO_ID" \
  --fps 10 \
  --append-new
```

Confirm that a real LeRobot dataset was written, rather than only the fallback
JSONL output:

```bash
test -f "$DATASET_DIR/meta/xarm_raw_manifest.json"
find "$DATASET_DIR/data" -type f -name 'episode_*.parquet' | wc -l
```

### 3. Validate the training handoff

```bash
uv run python -m data.real.validation.build_dataset_schema \
  --dataset-dir "$DATASET_DIR" \
  --output "$DATASET_DIR/meta/xarm_dataset_schema.json"

uv run python -m training.validation.openpi_smoke \
  --dataset-dir "$DATASET_DIR" \
  --repo-id "$REPO_ID" \
  --output-json /tmp/xarm_openpi_smoke.json

uv run python -m training.cli show pi05_xarm

uv run python -m training.cli preflight pi05_xarm \
  --dataset-path "real_xarm_pi05_20260703=$DATASET_DIR" \
  --openpi-root "$OPENPI_ROOT"
```

Do not launch training unless the preflight report says `launch_ready: true`
and the expected normalization asset is available.

