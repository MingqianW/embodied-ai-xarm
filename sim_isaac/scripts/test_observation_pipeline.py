from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_runtime.image_preprocessing import image_diagnostics
from policy_runtime.observation_builder import validate_policy_observation
from sim_isaac.dependencies import IsaacDependencyError
from sim_isaac.environment import DEFAULT_CONFIG_DIR, IsaacEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one Isaac observation without contacting a policy server."
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG_DIR / "robot.yaml")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CONFIG_DIR / "cameras.yaml")
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONFIG_DIR / "control.yaml")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_CONFIG_DIR / "tasks.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path for automated validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve() if args.output is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"status": "starting"}, indent=2) + "\n",
            encoding="utf-8",
        )
    try:
        with IsaacEnvironment(
            robot_config_path=args.robot_config,
            camera_config_path=args.camera_config,
            control_config_path=args.control_config,
            task_config_path=args.task_config,
            prompt=args.prompt,
            headless=args.headless,
        ) as environment:
            environment.require_safe("offline observation")
            observation = environment.observe()
            validate_policy_observation(observation)
            report = {
                "status": "passed",
                "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
                "keys": sorted(observation.as_openpi_dict()),
                "state_shape": list(observation.state.shape),
                "state_dtype": str(observation.state.dtype),
                "state": observation.state.tolist(),
                "base_image": image_diagnostics(observation.base_image),
                "wrist_image": image_diagnostics(observation.wrist_image),
                "prompt": observation.prompt,
                "frame_ids": observation.frame_ids,
                "metadata": observation.metadata,
                "safety": environment.safety_diagnostics(),
            }
            if output_path is not None:
                output_path.write_text(
                    json.dumps(report, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(report, indent=2))
    except (IsaacDependencyError, FileNotFoundError, RuntimeError, ValueError) as exc:
        if output_path is not None:
            output_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
