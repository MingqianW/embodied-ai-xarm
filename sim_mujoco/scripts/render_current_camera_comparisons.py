
from __future__ import annotations

from camera_calibration_lib import (
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    read_json,
    render_current_comparisons,
)


def main() -> None:
    manifest = read_json(MANIFEST_PATH)
    config = load_config(CONFIG_PATH)
    samples = manifest["samples"]
    render_current_comparisons(samples, config)
    print(f"Rendered current camera comparisons for {len(samples)} selected frames.")


if __name__ == "__main__":
    main()
