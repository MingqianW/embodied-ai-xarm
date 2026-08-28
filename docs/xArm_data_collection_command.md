# xArm Data Collection Commands

This file summarizes the basic commands used for xArm pick-up data collection.

> **External collector:** `collect_async_gripper_optimized.py` and its xArm /
> RealSense hardware adapters are not tracked in this repository. The tracked
> offline boundary starts at `data.real.collection`, which parses the
> collector's `meta.json`, `robot_log.csv`, and camera files. Do not interpret
> the command below as a repository-owned or offline-testable collector.

## 1. Start the Teach Pendant

Run the uFactory Studio client:

```bash
cd /home/xingyu/Downloads
chmod +x ufactory-studio-client-linux-1.0.2.AppImage
./ufactory-studio-client-linux-1.0.2.AppImage
```

Use the teach pendant to control the robot and enable manual mode.

**Important:** During data collection, reopen manual mode from the teach pendant for each trial.

## 2. Run Data Collection

Go to the OpenPI folder:

```bash
cd /home/xingyu/pi_0.5/openpi
```

Example command:

```bash
uv run python -u /home/xingyu/robot/xarm-calibrate-hanyang/real_world/collect_async_gripper_optimized.py \
  --xarm_ip 192.168.1.209 \
  --base_dir /home/xingyu/xarm_pi05_data/raw \
  --task pick_up_blue_block \
  --max_realsense 3 \
  --rs_width 640 \
  --rs_height 480 \
  --rs_fps 30 \
  --poll_hz 30 \
  --save_hz 10 \
  --async_writer \
  --gripper_auto_fine_mm 0
```

This command is only an example. You can change arguments such as `--task`, `--base_dir`, camera settings, save rate, or robot IP as needed.

Example task names:

```text
pick_up_dark_blue_block
pick_up_light_blue_block
pick_up_red_block
```

## 3. Keyboard Controls During Collection

| Key | Function |
|---|---|
| `Space` | Start / stop current data collection |
| `n` | Proceed to the next dataset and restore the initial position |
| `x` | Drop the current dataset |
| `a` | Slowly close the gripper |
| `d` | Slowly open the gripper |
| `[` | Open gripper to minimum position |
| `]` | Close gripper to maximum position |

## 4. Basic Trial Workflow

```text
1. Open teach pendant.
2. Enable or reopen manual mode.
3. Position the robot and scene.
4. Press Space to start recording.
5. Perform the pick-up demonstration.
6. Press Space to stop recording.
7. Press n to save/proceed and restore the initial position.
8. Reopen manual mode from the teach pendant before the next trial.
```

Use `x` if the current trial is bad and should be dropped.
