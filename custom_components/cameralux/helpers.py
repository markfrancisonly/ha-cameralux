"""Helper utilities for the CameraLux component.

These functions are kept free of Home Assistant dependencies so they can be
unit-tested in isolation.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .const import CONF_HEIGHT, CONF_WIDTH, CONF_X, CONF_Y

_LOGGER = logging.getLogger(__name__)

# Resampling fallback (older Pillow)
try:
    RESAMPLE = Image.Resampling.LANCZOS  # Pillow >= 9.1
except Exception:  # noqa: BLE001
    RESAMPLE = Image.LANCZOS


def build_unique_id(
    name: str, camera_entity_id: str | None, image_url: str | None
) -> str:
    """Unique ID base (camera/url) + short hash of name to allow multiple sensors per camera."""
    suffix = hashlib.md5((name or "unnamed").encode()).hexdigest()[:8]
    if camera_entity_id:
        base = f"cameralux_{camera_entity_id.replace('.', '_')}"
    elif image_url:
        base = f"cameralux_url_{hashlib.md5(image_url.encode()).hexdigest()}"
    else:
        base = "cameralux_manual"
    return f"{base}_{suffix}"


def int_if_value(value) -> int | None:
    """Return int(value) if it is a real value (not None/empty string); otherwise None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_bounded_int(
    val,
    default: int = 0,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Coerce a value to int, clamping to [min_value, max_value] when provided."""
    try:
        iv = int(val)
    except (TypeError, ValueError):
        iv = default
    if min_value is not None and iv < min_value:
        iv = min_value
    if max_value is not None and iv > max_value:
        iv = max_value
    return iv


def coerce_float(val, default: float) -> float:
    """Coerce a value to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def precompute_inverse_gamma_8bit_lut() -> np.ndarray:
    """Generate a 256-entry inverse gamma LUT for sRGB 8-bit values."""
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized = i / 255.0
        if normalized <= 0.04045:
            lut[i] = normalized / 12.92
        else:
            lut[i] = ((normalized + 0.055) / 1.055) ** 2.4
    return lut


INVERSE_GAMMA_LUT = precompute_inverse_gamma_8bit_lut()


def crop_image(image: Image.Image, roi_config: Optional[Dict[str, int]]) -> Image.Image:
    """Crop the image to the provided ROI; treat 0 width/height as full dimension."""
    if not roi_config:
        return image

    img_width, img_height = image.size
    x = roi_config.get(CONF_X, 0) or 0
    y = roi_config.get(CONF_Y, 0) or 0
    w = roi_config.get(CONF_WIDTH, 0) or img_width
    h = roi_config.get(CONF_HEIGHT, 0) or img_height

    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = min(w, img_width - x)
    h = min(h, img_height - y)

    if w <= 0 or h <= 0:
        _LOGGER.warning("Invalid ROI after clamping: %s", roi_config)
        return image

    _LOGGER.debug("ROI: x=%d, y=%d, width=%d, height=%d", x, y, w, h)
    return image.crop((x, y, x + w, y + h))


def resize_image(image: Image.Image, max_pixels: int) -> Image.Image:
    """Resize the image if it exceeds max_pixels, preserving aspect ratio."""
    width, height = image.size
    total_pixels = width * height
    if total_pixels <= max_pixels:
        return image

    scale = math.sqrt(max_pixels / total_pixels)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    _LOGGER.debug(
        "Scaled from (%d x %d) to (%d x %d) for <= %d px",
        width,
        height,
        new_w,
        new_h,
        max_pixels,
    )
    return image.resize((new_w, new_h), RESAMPLE)


def calculate_image_luminance(image: Image.Image) -> Tuple[float, float]:
    """Compute average and perceived luminance from an image in linear space."""
    MODE_GRAYSCALE = "L"
    MODE_PALETTE = "P"
    MODE_CMYK = "CMYK"
    MODE_RGB = "RGB"
    MODE_RGBA = "RGBA"
    MODE_16_BIT_GRAYSCALE = "I;16"
    MODE_FLOAT = "F"
    MODE_LINEAR = "I"

    if image.mode in [MODE_PALETTE, MODE_CMYK]:
        image = image.convert(MODE_RGB)

    img_pixels = np.array(image)

    if img_pixels.ndim == 3 and img_pixels.shape[2] > 3:
        img_pixels = img_pixels[:, :, :3]

    if image.mode == MODE_16_BIT_GRAYSCALE:
        linearized = img_pixels.astype(np.float32) / 65535.0
    elif image.mode in [MODE_LINEAR, MODE_FLOAT]:
        linearized = img_pixels.astype(np.float32)
    elif image.mode in [MODE_RGB, MODE_RGBA, MODE_GRAYSCALE]:
        linearized = INVERSE_GAMMA_LUT[img_pixels.astype(np.uint8)]
    else:
        try:
            image = image.convert(MODE_RGB)
            img_pixels = np.array(image)
            linearized = INVERSE_GAMMA_LUT[img_pixels.astype(np.uint8)]
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Unsupported image mode: {image.mode}") from e

    if linearized.ndim == 2:
        y_plane = linearized
    else:
        y_plane = (
            0.2126 * linearized[:, :, 0]
            + 0.7152 * linearized[:, :, 1]
            + 0.0722 * linearized[:, :, 2]
        )

    avg_lum = float(np.mean(y_plane))
    perceived = float(np.log1p(avg_lum))
    _LOGGER.debug(
        "luminance avg=%f perceived=%f from %s", avg_lum, perceived, image.mode
    )
    return avg_lum, perceived
