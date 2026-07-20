from __future__ import annotations

import json

import mujoco
import numpy as np
from PIL import Image

from camera_calibration_lib import (
    CALIBRATION_ROOT,
    CONFIG_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RAW_ROOT,
    discover_episodes,
    load_config,
    read_json,
    verify_raw_snapshot,
    write_json,
)


def main() -> None:
    manifest = read_json(MANIFEST_PATH)
    metrics = read_json(METRICS_PATH)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    config = load_config()
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)]
    current_raw = verify_raw_snapshot(discover_episodes())
    expected_raw = manifest["raw_snapshot"]
    camera_config_applied = {}
    for camera_name in ("base_camera", "wrist_camera"):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        expected = config[camera_name]
        camera_config_applied[camera_name] = bool(
            camera_id >= 0
            and np.allclose(model.cam_pos[camera_id], expected["position"], atol=1e-8)
            and np.isclose(model.cam_fovy[camera_id], expected["fovy_deg"], atol=1e-8)
        )
    native_files = sorted((CALIBRATION_ROOT / "after").glob("*_sim_*.png"))
    policy_files = sorted((CALIBRATION_ROOT / "after").glob("*_policy_*.png"))
    native_sizes = sorted({Image.open(path).size for path in native_files})
    policy_sizes = sorted({Image.open(path).size for path in policy_files})
    report = {
        "scene_compiles": True,
        "model_path": str(MODEL_PATH),
        "config_path": str(CONFIG_PATH),
        "native_render_files": len(native_files),
        "native_render_sizes": [list(size) for size in native_sizes],
        "policy_render_files": len(policy_files),
        "policy_render_sizes": [list(size) for size in policy_sizes],
        "camera_config_applied": camera_config_applied,
        "actuator_count": model.nu,
        "actuators": actuator_names,
        "six_arm_plus_one_gripper": model.nu == 7 and actuator_names[-1] == "gripper_actuator",
        "raw_snapshot_unchanged": current_raw == expected_raw,
        "raw_root": str(RAW_ROOT),
        "metrics": metrics["aggregate"],
    }
    print(json.dumps(report, indent=2))
    write_json(CALIBRATION_ROOT / "validation_report.json", report)
    required = (
        report["six_arm_plus_one_gripper"],
        report["raw_snapshot_unchanged"],
        all(camera_config_applied.values()),
        native_sizes == [(640, 480)] and len(native_files) == 32,
        policy_sizes == [(224, 224)] and len(policy_files) == 32,
    )
    if not all(required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
