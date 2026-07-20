from __future__ import annotations

import json

from camera_calibration_lib import DISCOVERY_PATH, dataset_summary, discover_episodes, write_json


def main() -> None:
    summary = dataset_summary(discover_episodes())
    write_json(DISCOVERY_PATH, summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {DISCOVERY_PATH}")


if __name__ == "__main__":
    main()
