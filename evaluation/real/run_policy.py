import argparse
import os
import sys
import time
import csv
import json
import re
from datetime import datetime
from pathlib import Path
import numpy as np

from data.common.schema import XARM_STATE_COLUMNS
from evaluation.real.safety import RealExecutionAuthorization
from evaluation.real.safety import XARM6_JOINT_LIMITS_RAD
from evaluation.real.safety import validate_real_action_chunk
from policy_runtime.safety import SafetyConfig

OPENPI_SOURCE_ROOT = Path(
    os.environ.get("OPENPI_SOURCE_ROOT", "/home/xingyu/pi_0.5/openpi/src")
)
REAL_WORLD_SOURCE_ROOT = Path(
    os.environ.get("XARM_REAL_WORLD_ROOT", "/home/xingyu/robot/xarm-calibrate-hanyang")
)


RAW_DATA_ROOT = Path(os.environ.get("XARM_RAW_ROOT", "/home/xingyu/xarm_pi05_data/raw"))
STATE_COLUMNS = XARM_STATE_COLUMNS
DEFAULT_REAL_SAFETY_CONFIG = SafetyConfig(max_joint_delta_rad=0.1)


def _load_external_runtime():
    """Load site-owned policy/hardware packages only after an explicit CLI gate."""

    for source in (OPENPI_SOURCE_ROOT, REAL_WORLD_SOURCE_ROOT):
        if str(source) not in sys.path:
            sys.path.append(str(source))
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        from real_world.camera.multi_realsense import MultiRealsense
        from real_world.xarm6 import XARM6
    except ImportError as exc:
        raise RuntimeError(
            "Real runtime dependencies are external. Set OPENPI_SOURCE_ROOT and "
            "XARM_REAL_WORLD_ROOT to the validated site installations."
        ) from exc
    return training_config, policy_config, XARM6, MultiRealsense


def _confirm_motion_authorization() -> RealExecutionAuthorization:
    def confirmed(prompt: str) -> bool:
        return input(f"{prompt} Type YES to confirm: ").strip() == "YES"

    authorization = RealExecutionAuthorization(
        operator_present=confirmed("An attending operator will supervise every command."),
        workspace_clear=confirmed("The robot workspace is clear and has been assessed."),
        emergency_stop_accessible=confirmed("The physical emergency stop is accessible."),
        robot_motion_confirmed=True,
    )
    authorization.require()
    return authorization


def normalize_task_name(task_name):
    """Accept task names with spaces or underscores and store one stable folder name."""
    normalized = task_name.strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("task name cannot be empty")
    if normalized in {".", ".."} or any(char in normalized for char in ("/", "\\", ":")):
        raise ValueError(f"invalid task name: {task_name!r}")
    return normalized


def normalize_tag(tag):
    normalized = tag.strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_.]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("._")
    if not normalized:
        raise ValueError("tag cannot be empty")
    return normalized


def prompt_nonempty(label, normalizer):
    while True:
        value = input(f"{label}> ").strip()
        try:
            return normalizer(value)
        except ValueError as exc:
            print(exc)


def next_episode_dir(raw_root, task_slug, tag_slug):
    task_dir = raw_root / task_slug
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"episode_{timestamp}_{tag_slug}"
    candidate = task_dir / base_name
    suffix = 1
    while candidate.exists():
        candidate = task_dir / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def save_rgb_png(path, image):
    import cv2

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr[:, :, ::-1])
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


def save_raw_trajectory(records, *, prompt, raw_root=RAW_DATA_ROOT):
    if len(records) < 2:
        print("Not saving raw trajectory: need at least 2 observed frames.")
        return None

    print("\nSave this rollout as raw data.")
    print("Task name can be typed with spaces or underscores, e.g. 'pick up red pepper' or 'pick_up_red_pepper'.")
    task_slug = prompt_nonempty("task name", normalize_task_name)
    tag_slug = prompt_nonempty("trajectory tag", normalize_tag)

    episode_dir = next_episode_dir(raw_root, task_slug, tag_slug)
    image_dir = episode_dir / "realsense_0"
    wrist_image_dir = episode_dir / "realsense_1"
    episode_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    action_chunks = []
    for frame_idx, record in enumerate(records):
        image_rel = Path("realsense_0") / f"{frame_idx:06d}.png"
        wrist_rel = Path("realsense_1") / f"{frame_idx:06d}.png"
        save_rgb_png(image_dir / image_rel.name, record["image"])
        save_rgb_png(wrist_image_dir / wrist_rel.name, record["wrist_image"])

        state = np.asarray(record["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] < len(STATE_COLUMNS):
            raise ValueError(f"record {frame_idx} state has shape {state.shape}, expected at least 7 values")
        row = {
            "ts": f"{float(record['timestamp']):.6f}",
            **{name: f"{float(value):.9g}" for name, value in zip(STATE_COLUMNS, state[: len(STATE_COLUMNS)])},
            "realsense_0_file": image_rel.as_posix(),
            "realsense_1_file": wrist_rel.as_posix(),
        }
        rows.append(row)
        action_chunks.append(np.asarray(record["actions"], dtype=np.float32))

    with (episode_dir / "robot_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", *STATE_COLUMNS, "realsense_0_file", "realsense_1_file"])
        writer.writeheader()
        writer.writerows(rows)

    np.save(episode_dir / "policy_action_chunks.npy", np.asarray(action_chunks, dtype=np.float32))
    meta = {
        "task": task_slug,
        "prompt": prompt,
        "tag": tag_slug,
        "format": "xarm_policy_rollout_raw_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_frames": len(records),
        "state_columns": list(STATE_COLUMNS),
        "image_keys": {
            "realsense_0_file": "base scene camera",
            "realsense_1_file": "wrist camera",
        },
        "notes": [
            "robot_log.csv is compatible with data.real.conversion.convert_xarm_raw_to_lerobot",
            "policy_action_chunks.npy stores the generated action chunk for each observed frame",
        ],
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved raw trajectory: {episode_dir}")
    return episode_dir

def get_latest_rgb(camera_dict, cam_idx=0):
    if cam_idx not in camera_dict:
        raise KeyError(f"Camera index {cam_idx} not found. Available: {list(camera_dict.keys())}")


    cam_data = camera_dict[cam_idx]


    candidate_keys = ["rgb", "color", "image", "bgr", "vis"]


    image = None
    key_used = None


    for key in candidate_keys:
        if key in cam_data and isinstance(cam_data[key], np.ndarray):
            arr = cam_data[key]
            if arr.ndim == 4:
                image = arr[-1]
                key_used = key
                break
            elif arr.ndim in (2, 3):
                image = arr
                key_used = key
                break


    if image is None:
        for key, arr in cam_data.items():
            if isinstance(arr, np.ndarray):
                if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
                    image = arr[-1]
                    key_used = key
                    break
                elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                    image = arr
                    key_used = key
                    break
                elif arr.ndim == 2:
                    image = arr
                    key_used = key
                    break


    if image is None:
        raise KeyError(f"No image-like array found for camera {cam_idx}. Keys: {list(cam_data.keys())}")


    print(f"Using camera {cam_idx} image key: {key_used}", flush=True)


    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]


    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
       
    if key_used == "bgr":
        image = image[:, :, ::-1].copy()
    return image


def get_xarm_api(robot):
    """
    Try to find the underlying xArm SDK API object inside your XARM6 wrapper.
    """
    candidate_attrs = ["arm", "_arm", "api", "_api", "xarm", "_xarm"]


    for name in candidate_attrs:
        if hasattr(robot, name):
            obj = getattr(robot, name)
            if hasattr(obj, "set_servo_angle"):
                print(f"Using robot.{name} as xArm API")
                return obj


    if hasattr(robot, "set_servo_angle"):
        print("Using robot directly as xArm API")
        return robot


    raise AttributeError(
        "Could not find xArm API object inside XARM6. "
        "Run: print(dir(robot)) and show me the output."
    )




def safe_execute_actions(
    robot,
    actions,
    max_steps=2,
    max_joint_delta=DEFAULT_REAL_SAFETY_CONFIG.max_joint_delta_rad,
    joint_speed=0.25,
    joint_acc=1.0,
    gripper_min=DEFAULT_REAL_SAFETY_CONFIG.gripper_min,
    gripper_max=DEFAULT_REAL_SAFETY_CONFIG.gripper_max,
    gripper_speed=1500,
    dt=0.01,
    authorization=None,
    joint_limits=None,
):
    """
    Execute first few actions from the model.


    Expected action format from your pi05_xarm config:
        action[:6] = target joint positions, radians
        action[6]  = gripper target, mm


    Your model has action_dim=32, so we only use the first 7 values.
    The rest are padding / unused.
    """
    if not isinstance(authorization, RealExecutionAuthorization):
        raise PermissionError("RealExecutionAuthorization is required before robot motion")
    authorization.require()

    actions = np.asarray(actions, dtype=np.float32)


    if actions.ndim == 1:
        actions = actions[None, :]

    if actions.ndim != 2 or actions.shape[1] < 7 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite actions with shape (H, >=7), got {actions.shape}")
    if max_steps < 1 or max_joint_delta <= 0 or dt <= 0:
        raise ValueError("Execution steps, joint delta, and command interval must be positive")
    if gripper_min >= gripper_max:
        raise ValueError("gripper_min must be below gripper_max")

    n = min(max_steps, actions.shape[0])
    limits = (
        XARM6_JOINT_LIMITS_RAD
        if joint_limits is None
        else np.asarray(joint_limits, dtype=np.float32)
    )
    current_joints_deg = np.asarray(
        robot.get_current_joint(), dtype=np.float32
    ).reshape(-1)[:6]
    current_joints = np.deg2rad(current_joints_deg).astype(np.float32)
    current_gripper = float(np.asarray(robot.get_gripper_state()).reshape(-1)[0])
    current_state = np.concatenate(
        [current_joints, np.asarray([current_gripper], dtype=np.float32)]
    )
    validation = validate_real_action_chunk(
        actions[:n, :7],
        current_state=current_state,
        joint_limits=limits,
        authorization=authorization,
        config=SafetyConfig(
            max_joint_delta_rad=max_joint_delta,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
        ),
    )
    if not validation.accepted:
        raise ValueError(f"Unsafe real action chunk rejected: {validation.reason}")
    validated_actions = validation.actions
    if validation.messages:
        print("Real action safety adjustments:", *validation.messages, sep="\n  ")

    api = get_xarm_api(robot)
    print("Executing validated action chunk shape:", validated_actions.shape)


    for i in range(n):
        a = validated_actions[i]


        if a.shape[0] < 7:
            raise ValueError(f"Expected action dim >= 7, got {a.shape}")


        target_joints = np.asarray(a[:6], dtype=np.float32)
        target_gripper = float(a[6])


        print(f"[EXEC {i}]")
        print("  validated target_joints:", target_joints)
        print("  target_gripper:", target_gripper)


        # Move joints.
        try:
            ret = api.set_servo_angle(
                angle=target_joints.tolist(),
                is_radian=True,
                speed=joint_speed,
                mvacc=joint_acc,
                wait=False,
            )
        except TypeError:
            ret = api.set_servo_angle(
                angle=target_joints.tolist(),
                is_radian=True,
                speed=joint_speed,
                wait=False, 
            )


        print("  set_servo_angle ret:", ret)
        if ret not in (None, 0):
            raise RuntimeError(f"xArm rejected joint command with status {ret}")


        # Move gripper.
        if hasattr(api, "set_gripper_position"):
            try:
                gret = api.set_gripper_position(
                    target_gripper,
                    wait=False,
                    speed=gripper_speed,
                )
                print("  set_gripper_position ret:", gret)
            except Exception as exc:
                raise RuntimeError("xArm gripper command failed") from exc
            if gret not in (None, 0):
                raise RuntimeError(f"xArm rejected gripper command with status {gret}")


        time.sleep(dt)  






def build_policy_example(robot, cameras, prompt, base_cam_idx=0, wrist_cam_idx=1):
    """Observe the current scene and build the exact input format used by pi05_xarm."""
    obs = cameras.get(k=1)
    image = get_latest_rgb(obs, cam_idx=base_cam_idx)
    wrist_image = get_latest_rgb(obs, cam_idx=wrist_cam_idx)


    joint_raw_deg = np.asarray(robot.get_current_joint(), dtype=np.float32).reshape(-1)
    gripper_raw = robot.get_gripper_state()


    print("raw joint deg shape:", joint_raw_deg.shape, "raw joint deg:", joint_raw_deg)
    print("raw gripper:", gripper_raw)


    joint_position = np.deg2rad(joint_raw_deg[:6]).astype(np.float32)
    gripper_scalar = float(np.asarray(gripper_raw).reshape(-1)[0])
    gripper_position = np.asarray([gripper_scalar], dtype=np.float32)
    state = np.concatenate([joint_position, gripper_position], axis=0).astype(np.float32)


    print("final state shape:", state.shape, "state:", state)
    if state.shape != (7,):
        raise ValueError(f"Expected state shape (7,), got {state.shape}: {state}")


    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }, state




def run_receding_horizon(
    robot,
    cameras,
    policy,
    prompt,
    *,
    cycles=25,
    execute_steps=5,
    base_cam_idx=0,
    wrist_cam_idx=1,
    max_joint_delta=0.1,
    joint_speed=0.25,
    joint_acc=1.0,
    gripper_min=50.0,
    gripper_max=845.0,
    gripper_speed=1500,
    dt=0.01,
    authorization=None,
):
    """Closed-loop policy rollout: observe, infer a chunk, execute first steps, repeat."""
    print(
        "Starting receding-horizon rollout: "
        f"cycles={cycles}, execute_steps={execute_steps}, max_joint_delta={max_joint_delta} rad"
    )
    trajectory_records = []
    for cycle in range(cycles):
        print(f"\n[ROLLOUT {cycle + 1}/{cycles}] observe -> infer")
        example, state = build_policy_example(
            robot,
            cameras,
            prompt,
            base_cam_idx=base_cam_idx,
            wrist_cam_idx=wrist_cam_idx,
        )
        observed_at = time.time()
        actions = np.asarray(policy.infer(example)["actions"], dtype=np.float32)
        trajectory_records.append(
            {
                "timestamp": observed_at,
                "image": np.asarray(example["observation/image"]).copy(),
                "wrist_image": np.asarray(example["observation/wrist_image"]).copy(),
                "state": np.asarray(state, dtype=np.float32).copy(),
                "actions": actions.copy(),
            }
        )
        print("Action chunk shape:", actions.shape)
        print("First action:", actions[0])
        print("First joint delta from current state:", actions[0, :6] - state[:6])
        print("First gripper target:", actions[0, 6])


        safe_execute_actions(
            robot,
            actions,
            max_steps=execute_steps,
            max_joint_delta=max_joint_delta,
            joint_speed=joint_speed,
            joint_acc=joint_acc,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            gripper_speed=gripper_speed,
            dt=dt,
            authorization=authorization,
        )

    if trajectory_records:
        print("\n[ROLLOUT final] observe final state")
        example, state = build_policy_example(
            robot,
            cameras,
            prompt,
            base_cam_idx=base_cam_idx,
            wrist_cam_idx=wrist_cam_idx,
        )
        trajectory_records.append(
            {
                "timestamp": time.time(),
                "image": np.asarray(example["observation/image"]).copy(),
                "wrist_image": np.asarray(example["observation/wrist_image"]).copy(),
                "state": np.asarray(state, dtype=np.float32).copy(),
                "actions": np.zeros_like(trajectory_records[-1]["actions"]),
            }
        )
    return trajectory_records
        
def restore_initial_pose(robot, init_pose=None, speed=80, mvacc=1000, wait=True):
    """
    Move xArm to a safe initial Cartesian pose before running policy.

    init_pose format:
        [x, y, z, roll, pitch, yaw]
    Units:
        x/y/z in mm
        roll/pitch/yaw in degrees
    """
    if init_pose is None:
        init_pose = [450, -170, 270, 180, 0, 0]

    api = get_xarm_api(robot)

    print("Restoring robot to initial pose:", init_pose, flush=True)

    # Make sure robot is enabled and ready.
    if hasattr(api, "motion_enable"):
        api.motion_enable(enable=True)
    if hasattr(api, "set_mode"):
        api.set_mode(0)
    if hasattr(api, "set_state"):
        api.set_state(0)
    time.sleep(0.5)

    # Move to Cartesian pose.
    try:
        ret = api.set_position(
            x=init_pose[0],
            y=init_pose[1],
            z=init_pose[2],
            roll=init_pose[3],
            pitch=init_pose[4],
            yaw=init_pose[5],
            speed=speed,
            mvacc=mvacc,
            wait=wait,
        )
    except TypeError:
        # Some SDK versions use positional args.
        ret = api.set_position(
            *init_pose,
            speed=speed,
            mvacc=mvacc,
            wait=wait,
        )

    print("restore_initial_pose set_position ret:", ret, flush=True)

    if ret != 0:
        raise RuntimeError(
            f"Failed to restore initial pose. xArm ret={ret}. "
            "Check workspace limits, robot mode/state, or pose safety."
        )

    time.sleep(1.0)
    print("Initial pose restore done.", flush=True)

def _parser():
    parser = argparse.ArgumentParser(
        description="Operator-supervised existing xArm policy runtime (not a formal real evaluator)."
    )
    parser.add_argument(
        "--allow-hardware",
        action="store_true",
        help="Acknowledge that this process may connect to and move a physical xArm.",
    )
    return parser


def _run_policy_prompt_loop(robot, cameras, policy) -> None:
    while True:
        prompt = input("prompt> ").strip()
        if prompt in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        example, _ = build_policy_example(robot, cameras, prompt)
        actions = np.asarray(policy.infer(example)["actions"], dtype=np.float32)

        print("Type:", type(actions))
        print("Shape:", actions.shape)
        print("Actions:", actions)

        execute = input("Run closed-loop rollout on robot? [y/N] ").strip().lower()
        if execute == "y":
            authorization = _confirm_motion_authorization()
            cycles_text = input("Number of observe/infer/execute cycles [25]: ").strip()
            cycles = int(cycles_text) if cycles_text else 25
            steps_text = input("Actions to execute per inference [2]: ").strip()
            execute_steps = int(steps_text) if steps_text else 2
            trajectory_records = run_receding_horizon(
                robot,
                cameras,
                policy,
                prompt,
                cycles=cycles,
                execute_steps=execute_steps,
                max_joint_delta=DEFAULT_REAL_SAFETY_CONFIG.max_joint_delta_rad,
                joint_speed=0.25,
                joint_acc=1.0,
                gripper_min=DEFAULT_REAL_SAFETY_CONFIG.gripper_min,
                gripper_max=DEFAULT_REAL_SAFETY_CONFIG.gripper_max,
                gripper_speed=1500,
                dt=0.01,
                authorization=authorization,
            )
            save_raw_trajectory(trajectory_records, prompt=prompt)
        else:
            print("Not executing actions.")


def _cleanup_hardware(robot, cameras) -> list[str]:
    """Release acquired resources in reverse order without short-circuiting."""

    errors: list[str] = []
    if cameras is not None:
        try:
            cameras.stop(wait=True)
        except Exception as exc:  # external cleanup must not block robot cleanup
            errors.append(f"camera cleanup failed: {exc}")
    if robot is not None:
        try:
            robot.disconnect()
        except Exception as exc:  # report after all cleanup attempts
            errors.append(f"robot cleanup failed: {exc}")
    return errors


def _run_hardware_session(
    training_config,
    policy_config,
    robot_factory,
    camera_factory,
    *,
    session_runner=_run_policy_prompt_loop,
) -> None:
    robot = None
    cameras = None
    try:
        robot = robot_factory(
            interface=os.environ.get("XARM_ROBOT_IP", "192.168.1.209")
        )
        print("Robot connected")

        cameras = camera_factory()
        cameras.start(wait=True)
        print(f"Cameras ready: {cameras.is_ready}")
        print(f"Number of active cameras: {cameras.n_cameras}")

        # Policy loading is protected because robot and cameras are live here.
        config = training_config.get_config("pi05_xarm_full_finetune")
        checkpoint_dir = os.environ.get(
            "XARM_POLICY_CHECKPOINT", "/home/xingyu/pi_0.5/openpi/checkpoint/25000"
        )
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
        print("Policy loaded")
        session_runner(robot, cameras, policy)
    finally:
        original_exception_active = sys.exc_info()[0] is not None
        cleanup_errors = _cleanup_hardware(robot, cameras)
        if cleanup_errors:
            message = "; ".join(cleanup_errors)
            if original_exception_active:
                print(f"WARNING: {message}", file=sys.stderr)
            else:
                raise RuntimeError(message)


def main():
    args = _parser().parse_args()
    if not args.allow_hardware:
        raise SystemExit(
            "Hardware access is disabled by default. Review evaluation/real/README.md, "
            "then pass --allow-hardware only at the supervised robot station."
        )
    training_config, policy_config, XARM6, MultiRealsense = _load_external_runtime()
    _run_hardware_session(
        training_config,
        policy_config,
        XARM6,
        MultiRealsense,
    )




if __name__ == "__main__":
    main()













