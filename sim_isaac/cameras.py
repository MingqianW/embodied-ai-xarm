from __future__ import annotations

import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from policy_runtime.config import load_yaml
from policy_runtime.image_preprocessing import (
    ImagePreprocessingConfig,
    ensure_rgb_uint8,
)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    prim_path: str
    parent_frame: str
    translation_m: np.ndarray
    orientation_wxyz: np.ndarray
    vertical_fov_deg: float
    resolution: tuple[int, int]
    near_clip_m: float
    far_clip_m: float
    preprocessing: ImagePreprocessingConfig


def _camera_config(name: str, value: dict[str, Any]) -> CameraConfig:
    pre = value.get("preprocessing", {})
    crop = list(pre.get("crop", [0, 0, 0, 0]))
    if len(crop) != 4:
        raise ValueError(f"{name} preprocessing crop must be [top, bottom, left, right]")
    config = CameraConfig(
        name=name,
        prim_path=str(value["prim_path"]),
        parent_frame=str(value["parent_frame"]),
        translation_m=np.asarray(value["translation_m"], dtype=np.float32),
        orientation_wxyz=np.asarray(
            value["orientation_quaternion_wxyz"], dtype=np.float32
        ),
        vertical_fov_deg=float(value["vertical_fov_deg"]),
        resolution=tuple(int(v) for v in value["resolution"]),
        near_clip_m=float(value["near_clip_m"]),
        far_clip_m=float(value["far_clip_m"]),
        preprocessing=ImagePreprocessingConfig(
            width=int(pre.get("resize_with_pad", [224, 224])[0]),
            height=int(pre.get("resize_with_pad", [224, 224])[1]),
            input_color_order=str(pre.get("input_color_order", "RGB")).upper(),
            flip_horizontal=bool(pre.get("flip_horizontal", False)),
            flip_vertical=bool(pre.get("flip_vertical", False)),
            crop_top=int(crop[0]),
            crop_bottom=int(crop[1]),
            crop_left=int(crop[2]),
            crop_right=int(crop[3]),
        ),
    )
    validate_camera_config(config)
    return config


def load_camera_configs(path: Path) -> dict[str, CameraConfig]:
    root = load_yaml(path)
    return {
        "base": _camera_config("base", root["base_camera"]),
        "wrist": _camera_config("wrist", root["wrist_camera"]),
    }


def validate_camera_config(config: CameraConfig) -> None:
    if config.translation_m.shape != (3,) or not np.isfinite(config.translation_m).all():
        raise ValueError(f"{config.name} camera translation must be finite shape (3,)")
    if config.orientation_wxyz.shape != (4,):
        raise ValueError(f"{config.name} camera orientation must have shape (4,)")
    if not np.isclose(np.linalg.norm(config.orientation_wxyz), 1.0, atol=1e-4):
        raise ValueError(f"{config.name} camera quaternion must be normalized")
    if len(config.resolution) != 2 or min(config.resolution) <= 0:
        raise ValueError(f"{config.name} camera resolution must contain positive width/height")
    if not 1.0 <= config.vertical_fov_deg < 179.0:
        raise ValueError(f"{config.name} camera vertical FOV must be in [1, 179)")
    if not 0.0 < config.near_clip_m < config.far_clip_m:
        raise ValueError(f"{config.name} camera clipping range is invalid")
    if config.preprocessing.width != 224 or config.preprocessing.height != 224:
        raise ValueError(
            f"{config.name} policy preprocessing must produce exactly 224x224 images"
        )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    result = np.asarray(value)
    while result.ndim > 3 and result.shape[0] == 1:
        result = result[0]
    return result


def _experimental_resolution(config: CameraConfig) -> tuple[int, int]:
    """Convert repository (width, height) to CameraSensor (height, width)."""

    width, height = config.resolution
    return height, width


def _usd_camera_apertures(
    config: CameraConfig,
    *,
    focal_length: float = 18.0,
) -> tuple[float, float]:
    """Return horizontal/vertical apertures matching FOV and pixel aspect."""

    width, height = config.resolution
    vertical = 2.0 * focal_length * np.tan(
        np.deg2rad(config.vertical_fov_deg) / 2.0
    )
    horizontal = vertical * float(width) / float(height)
    return float(horizontal), float(vertical)


def _configure_usd_optics(config: CameraConfig) -> None:
    from pxr import Gf, UsdGeom
    import omni.usd

    camera = UsdGeom.Camera.Get(omni.usd.get_context().get_stage(), config.prim_path)
    if not camera:
        raise RuntimeError(f"Camera prim was not created at {config.prim_path}")
    focal_length = 18.0
    horizontal_aperture, vertical_aperture = _usd_camera_apertures(
        config,
        focal_length=focal_length,
    )
    camera.GetFocalLengthAttr().Set(float(focal_length))
    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(float(vertical_aperture))
    camera.GetClippingRangeAttr().Set(
        Gf.Vec2f(float(config.near_clip_m), float(config.far_clip_m))
    )


class _ExperimentalCamera:
    backend = "isaacsim.sensors.experimental.rtx"

    def __init__(self, config: CameraConfig, rendering_hz: float) -> None:
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        authoring = RtxCamera(
            config.prim_path,
            tick_rate=float(rendering_hz),
            translations=np.asarray([config.translation_m], dtype=np.float32),
            orientations=np.asarray([config.orientation_wxyz], dtype=np.float32),
        )
        _configure_usd_optics(config)
        self.sensor = CameraSensor(
            authoring,
            resolution=_experimental_resolution(config),
            annotators=["rgb"],
        )

    def read_rgb(self) -> np.ndarray:
        data, _ = self.sensor.get_data("rgb")
        frame = _as_numpy(data)
        if frame.ndim < 3:
            raise RuntimeError("RTX camera frame is not ready")
        return ensure_rgb_uint8(frame, input_color_order="RGB")

    def close(self) -> None:
        detach = getattr(self.sensor, "detach_annotators", None)
        if detach is not None:
            detach(["rgb"])


class _LegacyCamera:
    backend = "isaacsim.sensors.camera"

    def __init__(self, config: CameraConfig, rendering_hz: float) -> None:
        from isaacsim.sensors.camera import Camera

        self.sensor = Camera(
            prim_path=config.prim_path,
            name=f"{config.name}_camera",
            frequency=float(rendering_hz),
            resolution=config.resolution,
            translation=config.translation_m,
            orientation=config.orientation_wxyz,
        )
        self.sensor.initialize()
        if hasattr(self.sensor, "set_clipping_range"):
            self.sensor.set_clipping_range(config.near_clip_m, config.far_clip_m)
        if hasattr(self.sensor, "set_vertical_aperture"):
            focal = float(self.sensor.get_focal_length())
            aperture = 2.0 * focal * np.tan(np.deg2rad(config.vertical_fov_deg) / 2.0)
            self.sensor.set_vertical_aperture(float(aperture))

    def read_rgb(self) -> np.ndarray:
        return ensure_rgb_uint8(
            np.asarray(self.sensor.get_rgba()),
            input_color_order="RGB",
        )

    def close(self) -> None:
        pass


class IsaacCameraRig:
    """Two-camera rig with Isaac 6 RTX authoring and a 4.5-5.x fallback."""

    def __init__(
        self,
        configs: dict[str, CameraConfig],
        *,
        rendering_hz: float,
    ) -> None:
        if set(configs) != {"base", "wrist"}:
            raise ValueError("Isaac camera rig requires exactly base and wrist configurations")
        self.configs = configs
        self._sensors: dict[str, Any] = {}
        self._frame_ids = {"base": 0, "wrist": 0}
        self._last_capture_wall_s: float | None = None
        self.backend = ""
        force_legacy = os.environ.get("ISAAC_CAMERA_BACKEND", "").strip().lower() == "legacy"
        try:
            if force_legacy:
                raise ImportError("ISAAC_CAMERA_BACKEND=legacy")
            for name, config in configs.items():
                self._sensors[name] = _ExperimentalCamera(config, rendering_hz)
            self.backend = _ExperimentalCamera.backend
        except ImportError:
            for sensor in self._sensors.values():
                sensor.close()
            self._sensors = {
                name: _LegacyCamera(config, rendering_hz)
                for name, config in configs.items()
            }
            self.backend = _LegacyCamera.backend

    def read(self) -> dict[str, np.ndarray]:
        frames = {name: sensor.read_rgb() for name, sensor in self._sensors.items()}
        for name, frame in frames.items():
            expected_w, expected_h = self.configs[name].resolution
            if frame.shape != (expected_h, expected_w, 3):
                raise RuntimeError(
                    f"{name} camera returned {frame.shape}; expected "
                    f"{(expected_h, expected_w, 3)}. Check Isaac resolution ordering."
                )
            self._frame_ids[name] += 1
        self._last_capture_wall_s = time.perf_counter()
        return frames

    @property
    def frame_ids(self) -> dict[str, int]:
        return dict(self._frame_ids)

    def frame_age_s(self) -> float:
        if self._last_capture_wall_s is None:
            return float("inf")
        return time.perf_counter() - self._last_capture_wall_s

    def close(self) -> None:
        for sensor in self._sensors.values():
            sensor.close()
