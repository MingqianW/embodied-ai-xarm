"""Open the configured xArm experiment scene and keep the Isaac Sim GUI open."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_isaac.environment import DEFAULT_CONFIG_DIR, IsaacEnvironment


def main() -> int:
    # The environment builds the same validated scene used by the inspection
    # and policy scripts; headless=False makes the Isaac Sim window visible.
    with IsaacEnvironment(
        robot_config_path=DEFAULT_CONFIG_DIR / "robot.yaml",
        camera_config_path=DEFAULT_CONFIG_DIR / "cameras.yaml",
        control_config_path=DEFAULT_CONFIG_DIR / "control.yaml",
        task_config_path=DEFAULT_CONFIG_DIR / "tasks.yaml",
        headless=False,
    ) as environment:
        environment.require_safe("GUI scene preview")
        environment.observe()
        print("Isaac Sim GUI is open. Close the Isaac Sim window to exit.")
        app = environment._simulation_app
        while app is not None and app.is_running():
            app.update()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
