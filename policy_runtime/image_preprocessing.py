from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ColorOrder = Literal["RGB", "BGR"]


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    width: int = 224
    height: int = 224
    input_color_order: ColorOrder = "RGB"
    flip_horizontal: bool = False
    flip_vertical: bool = False
    crop_top: int = 0
    crop_bottom: int = 0
    crop_left: int = 0
    crop_right: int = 0


def ensure_rgb_uint8(image: np.ndarray, *, input_color_order: ColorOrder = "RGB") -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 image, got shape {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        if value.size and float(np.nanmax(value)) <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0.0, 255.0).astype(np.uint8)
    elif value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    if value.shape[2] == 4:
        value = value[:, :, :3]
    if input_color_order == "BGR":
        value = value[:, :, ::-1]
    elif input_color_order != "RGB":
        raise ValueError(f"Unsupported color order: {input_color_order}")
    return np.ascontiguousarray(value)


def preprocess_policy_image(
    image: np.ndarray,
    config: ImagePreprocessingConfig = ImagePreprocessingConfig(),
) -> np.ndarray:
    """Apply canonical orientation/color handling and OpenPI resize-with-pad."""

    if config.width <= 0 or config.height <= 0:
        raise ValueError("Policy image width and height must be positive")
    value = ensure_rgb_uint8(image, input_color_order=config.input_color_order)

    height, width = value.shape[:2]
    y1 = int(config.crop_top)
    y2 = height - int(config.crop_bottom)
    x1 = int(config.crop_left)
    x2 = width - int(config.crop_right)
    if min(y1, x1, config.crop_bottom, config.crop_right) < 0:
        raise ValueError("Crop values cannot be negative")
    if y1 >= y2 or x1 >= x2:
        raise ValueError(
            f"Crop removes the full image: source={value.shape}, "
            f"crop={(y1, config.crop_bottom, x1, config.crop_right)}"
        )
    value = value[y1:y2, x1:x2]
    if config.flip_vertical:
        value = value[::-1, :, :]
    if config.flip_horizontal:
        value = value[:, ::-1, :]

    try:
        from openpi_client import image_tools
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenPI image preprocessing is unavailable. Install the repository's "
            "third_party/openpi/packages/openpi-client package."
        ) from exc

    resized = image_tools.resize_with_pad(value, config.height, config.width)
    result = np.asarray(resized, dtype=np.uint8)
    expected = (config.height, config.width, 3)
    if result.shape != expected:
        raise ValueError(f"OpenPI preprocessing returned {result.shape}, expected {expected}")
    return np.ascontiguousarray(result)


def image_diagnostics(image: np.ndarray) -> dict[str, object]:
    value = np.asarray(image)
    if value.size == 0:
        raise ValueError("Cannot compute diagnostics for an empty image")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "color_order": "RGB",
        "min": int(value.min()),
        "max": int(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
    }
