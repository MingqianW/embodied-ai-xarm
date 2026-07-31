from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_runtime.config import load_yaml
from policy_runtime.remote_policy_client import (
    PolicyConnectionError,
    PolicyTimeoutError,
    RemotePolicyClient,
    RemotePolicyConfig,
)
from policy_runtime.runners import ClosedLoopConfig, run_closed_loop
from policy_runtime.safety import SafetyConfig
from sim_isaac.dependencies import IsaacDependencyError
from sim_isaac.environment import DEFAULT_CONFIG_DIR, IsaacEnvironment
from sim_isaac.recording import create_recorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded OpenPI closed loop in Isaac Sim."
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--max-policy-steps", type=int, default=10)
    parser.add_argument("--execute-chunk-steps", type=int, choices=range(1, 6))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--inference-timeout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task", default="pick_up_object")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG_DIR / "robot.yaml")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CONFIG_DIR / "cameras.yaml")
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONFIG_DIR / "control.yaml")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_CONFIG_DIR / "tasks.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_policy_steps <= 10:
        print(
            "ERROR: --max-policy-steps must be between 1 and 10 for the "
            "bounded validation runner",
            file=sys.stderr,
        )
        return 2
    control = load_yaml(args.control_config)
    policy_config = control["policy"]
    safety = control["safety"]
    recording = control["recording"]
    host = args.host or os.environ.get("OPENPI_POLICY_HOST") or str(policy_config["host"])
    port = int(
        args.port
        if args.port is not None
        else os.environ.get("OPENPI_POLICY_PORT", policy_config["port"])
    )
    prompt = args.prompt or str(policy_config["prompt"])
    timeout = float(args.inference_timeout or policy_config["timeout_s"])
    execute_steps = int(
        args.execute_chunk_steps or policy_config["execute_chunk_steps"]
    )
    should_record = bool(
        recording["enabled"] if args.record is None else args.record
    )
    output_dir = args.output_dir or Path(
        os.environ.get("ISAAC_OUTPUT_DIR", "sim_isaac/output")
    ) / "closed_loop"
    recorder = (
        create_recorder(
            output_dir / "recording",
            fps=int(recording["fps"]),
            max_frames=int(recording["max_frames"]),
            fallback_to_frames=bool(recording["fallback_to_frame_directory"]),
        )
        if should_record
        else None
    )
    try:
        with IsaacEnvironment(
            robot_config_path=args.robot_config,
            camera_config_path=args.camera_config,
            control_config_path=args.control_config,
            task_config_path=args.task_config,
            task_name=args.task,
            prompt=prompt,
            headless=args.headless,
            seed=args.seed,
        ) as environment, RemotePolicyClient(
            RemotePolicyConfig(
                host=host,
                port=port,
                connect_timeout_s=args.connect_timeout,
                inference_timeout_s=timeout,
            )
        ) as policy:
            environment.require_safe("bounded policy closed loop")
            try:
                result = run_closed_loop(
                    environment,
                    policy,
                    ClosedLoopConfig(
                        prompt=prompt,
                        max_policy_steps=args.max_policy_steps,
                        execute_chunk_steps=execute_steps,
                        control_period_s=1.0
                        / float(policy_config["control_hz"]),
                        output_dir=output_dir,
                    ),
                    safety_config=SafetyConfig(
                        action_mode=str(policy_config["action_mode"]),
                        max_joint_delta_rad=float(
                            safety["max_joint_delta_rad"]
                        ),
                        reject_if_clip_exceeds_rad=float(
                            safety["reject_if_clip_exceeds_rad"]
                        ),
                    ),
                    recorder=recorder,
                )
            except KeyboardInterrupt:
                environment.hold_position()
                if recorder is not None:
                    recorder.close()
                print("Stopped by operator; the robot was commanded to hold.")
                return 130
    except (
        IsaacDependencyError,
        FileNotFoundError,
        PolicyConnectionError,
        PolicyTimeoutError,
        RuntimeError,
        ValueError,
    ) as exc:
        if recorder is not None:
            recorder.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
