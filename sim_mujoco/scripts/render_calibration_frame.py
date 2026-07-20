from __future__ import annotations

import argparse
from pathlib import Path

from camera_calibration_lib import (
    CALIBRATION_ROOT,
    CalibrationRenderer,
    MANIFEST_PATH,
    load_config,
    policy_image,
    read_json,
    save_rgb,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--camera", choices=("base_camera", "wrist_camera", "both"), default="both")
    parser.add_argument("--output-dir", default=str(CALIBRATION_ROOT / "single_frame"))
    args = parser.parse_args()
    manifest = read_json(MANIFEST_PATH)
    sample = next(item for item in manifest["samples"] if item["sample_id"] == args.sample_id)
    config = load_config()
    render_config = config["render"]
    renderer = CalibrationRenderer(int(render_config["native_width"]), int(render_config["native_height"]))
    cameras = ("base_camera", "wrist_camera") if args.camera == "both" else (args.camera,)
    output_dir = Path(args.output_dir)
    try:
        for camera in cameras:
            native = renderer.render(sample, camera, config[camera])
            short = camera.replace("_camera", "")
            save_rgb(output_dir / f"{args.sample_id}_{short}_640x480.png", native)
            save_rgb(output_dir / f"{args.sample_id}_{short}_224x224.png", policy_image(native, config))
            print(camera, native.shape)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
