from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

from camera_calibration_lib import (
    CONFIG_PATH,
    MANIFEST_PATH,
    PROJECT_ROOT,
    CalibrationRenderer,
    blend_overlay,
    load_config,
    load_rgb,
    project_path,
    read_json,
    write_config,
)


WINDOW_NAME = "Base camera: raw | MuJoCo | overlay"
BUILD_SCRIPT = PROJECT_ROOT / "sim_mujoco" / "scripts" / "build_xarm6_pick_scene.py"

SLIDERS = {
    "position x": (-1.0, 2.0, 1000.0),
    "position y": (-1.5, 1.5, 1000.0),
    "position z": (0.0, 2.0, 1000.0),
    "target x": (-0.5, 1.5, 1000.0),
    "target y": (-1.0, 1.0, 1000.0),
    "target z": (0.0, 1.5, 1000.0),
    "roll deg": (-180.0, 180.0, 10.0),
    "fovy deg": (10.0, 120.0, 10.0),
}


def encode_slider(name: str, value: float) -> int:
    minimum, maximum, scale = SLIDERS[name]
    clipped = min(maximum, max(minimum, float(value)))
    return int(round((clipped - minimum) * scale))


def decode_slider(name: str) -> float:
    minimum, _, scale = SLIDERS[name]
    return minimum + cv2.getTrackbarPos(name, WINDOW_NAME) / scale


def create_sliders(parameters: dict) -> None:
    initial_values = {
        "position x": parameters["position"][0],
        "position y": parameters["position"][1],
        "position z": parameters["position"][2],
        "target x": parameters["target"][0],
        "target y": parameters["target"][1],
        "target z": parameters["target"][2],
        "roll deg": parameters["roll_deg"],
        "fovy deg": parameters["fovy_deg"],
    }
    for name, (_, maximum, scale) in SLIDERS.items():
        minimum = SLIDERS[name][0]
        slider_maximum = int(round((maximum - minimum) * scale))
        cv2.createTrackbar(name, WINDOW_NAME, encode_slider(name, initial_values[name]), slider_maximum, lambda _: None)


def current_parameters() -> dict:
    return {
        "frame": "world",
        "position": [
            decode_slider("position x"),
            decode_slider("position y"),
            decode_slider("position z"),
        ],
        "target": [
            decode_slider("target x"),
            decode_slider("target y"),
            decode_slider("target z"),
        ],
        "roll_deg": decode_slider("roll deg"),
        "fovy_deg": decode_slider("fovy deg"),
    }


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (150, 28), (0, 0, 0), thickness=-1)
    cv2.putText(output, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def save(parameters: dict, config: dict) -> None:
    config["base_camera"] = parameters
    metadata = config.setdefault("metadata", {})
    metadata["status"] = "manually_adjusted"
    metadata["manual_adjustment_utc"] = datetime.now(timezone.utc).isoformat()
    write_config(config, CONFIG_PATH)
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=PROJECT_ROOT, check=True)
    print(f"Saved base camera parameters to {CONFIG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="sample_012")
    args = parser.parse_args()

    manifest = read_json(MANIFEST_PATH)
    sample = next(
        (item for item in manifest["samples"] if item["sample_id"] == args.sample_id),
        None,
    )
    if sample is None:
        valid_ids = ", ".join(item["sample_id"] for item in manifest["samples"])
        raise SystemExit(f"Unknown sample ID {args.sample_id!r}. Valid IDs: {valid_ids}")

    config = load_config(CONFIG_PATH)
    render_config = config["render"]
    renderer = CalibrationRenderer(
        int(render_config["native_width"]),
        int(render_config["native_height"]),
    )
    real = load_rgb(project_path(sample["base_image"]))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1440, 720)
    create_sliders(config["base_camera"])
    previous = None
    saved = False

    try:
        while True:
            parameters = current_parameters()
            signature = tuple(parameters["position"] + parameters["target"] + [parameters["roll_deg"], parameters["fovy_deg"]])
            if signature != previous:
                simulated = renderer.render(sample, "base_camera", parameters)
                overlay = blend_overlay(real, simulated)
                display = np.concatenate(
                    [
                        label_panel(real, "RAW DATA"),
                        label_panel(simulated, "MUJOCO"),
                        label_panel(overlay, "OVERLAY"),
                    ],
                    axis=1,
                )
                cv2.imshow(WINDOW_NAME, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                previous = signature

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save(parameters, config)
                saved = True
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        renderer.close()
        cv2.destroyAllWindows()

    if not saved:
        print("Closed without saving camera changes.")


if __name__ == "__main__":
    main()
