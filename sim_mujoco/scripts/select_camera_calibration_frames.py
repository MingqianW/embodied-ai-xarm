from __future__ import annotations

import argparse
import json

from camera_calibration_lib import (
    MANIFEST_PATH,
    discover_episodes,
    sample_candidates,
    select_diverse_frames,
    verify_raw_snapshot,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-count", type=int, default=12)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=36)
    args = parser.parse_args()
    episodes = discover_episodes()
    candidates = sample_candidates(episodes, max_episodes=args.max_episodes)
    samples = select_diverse_frames(candidates, args.calibration_count, args.validation_count)
    manifest = {
        "schema_version": 1,
        "selection_method": "sharpness/content filtering followed by farthest-point joint-pose sampling",
        "camera_mapping": {"base_camera": "realsense_0", "wrist_camera": "realsense_1"},
        "joint_order": [f"j{i}_rad" for i in range(1, 7)],
        "joint_units": "radians",
        "raw_snapshot": verify_raw_snapshot(episodes),
        "samples": samples,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"candidates": len(candidates), "selected": len(samples), "episodes": sorted({s['episode'] for s in samples})}, indent=2))
    print(f"Saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
