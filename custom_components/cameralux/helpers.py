"""Helper utilities for the CameraLux component.

These functions are kept free of Home Assistant dependencies so they can be
unit-tested in isolation. process_image() is the single CPU-bound entry point
and is designed to run in an executor thread.

Imaging formulae:
  chroma    = (mean|Cb-128| + mean|Cr-128|) / 2 on the full frame
              (below threshold => monochrome / IR mode)
  perceived = mean(inverse_srgb_gamma(Y)) ** exponent   (Stevens'-law
              brightness; exponent 1.0 = physically linear)
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .const import (
    CONF_HEIGHT,
    CONF_WIDTH,
    CONF_X,
    CONF_Y,
    IR_GRAY_FRACTION_MIN,
    IR_TRIM_TOP_FRACTION,
)

_LOGGER = logging.getLogger(__name__)

# BILINEAR is statistically equivalent to LANCZOS for mean-based metrics and
# ~20 ms cheaper per update (manifest requires Pillow>=10, so Resampling exists).
RESAMPLE = Image.Resampling.BILINEAR

# Image modes that carry no usable color information: chroma is 0 by definition.
_COLORLESS_MODES = ("L", "LA", "I", "I;16", "F", "1")


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


def float_if_value(value) -> float | None:
    """Return float(value) if it is a real value (not None/empty string); otherwise None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
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

    if x >= img_width or y >= img_height:
        _LOGGER.warning(
            "ROI origin outside the frame, using full image: %s", roi_config
        )
        return image

    x = max(0, x)
    y = max(0, y)
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


def compute_chroma(image: Image.Image) -> float:
    """Shared IR detector: mean chroma deviation from neutral.

    Returns (mean|Cb-128| + mean|Cr-128|) / 2 on the plain 8-bit sRGB image
    (NOT the linearized data): IR/monochrome frames read ~0-4, color scenes
    ~7+. Colorless image modes are 0.0 by definition.
    """
    if image.mode == "P":  # palette images can carry color; expand first
        image = image.convert("RGB")
    if image.mode in _COLORLESS_MODES or len(image.getbands()) < 3:
        return 0.0
    if image.mode != "RGB":
        image = image.convert("RGB")
    ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.int16)
    cb = np.mean(np.abs(ycbcr[:, :, 1] - 128))
    cr = np.mean(np.abs(ycbcr[:, :, 2] - 128))
    return float((cb + cr) / 2.0)


def _linearized_y_plane(image: Image.Image) -> Tuple[np.ndarray, str, str]:
    """Decode any supported PIL mode into a linear-light Y plane in [0, 1].

    Shared by the color and IR luminance paths: the decode/linearize work is
    identical for both; only the statistic taken over the plane differs.
    Returns (y_plane, source_mode, branch) — the mode/branch identify which
    decode path produced the plane, for the callers' debug logs.
    """
    MODE_GRAYSCALE = "L"
    MODE_PALETTE = "P"
    MODE_CMYK = "CMYK"
    MODE_RGB = "RGB"
    MODE_RGBA = "RGBA"
    # Every 16-bit integer grayscale spelling PIL emits (endianness / native).
    MODES_16_BIT = ("I;16", "I;16B", "I;16L", "I;16N")
    MODE_FLOAT = "F"
    MODE_LINEAR = "I"

    # Conversions below rebind `image`, so keep the incoming mode for the log:
    # it identifies which decode branch produced a given reading.
    source_mode = image.mode

    if image.mode in [MODE_PALETTE, MODE_CMYK]:
        image = image.convert(MODE_RGB)

    img_pixels = np.array(image)

    if img_pixels.ndim == 3 and img_pixels.shape[2] > 3:
        img_pixels = img_pixels[:, :, :3]

    if image.mode in MODES_16_BIT or image.mode == MODE_LINEAR:
        # 16/32-bit grayscale sources; already linear-ish, normalize full scale
        linearized = img_pixels.astype(np.float32) / 65535.0
        branch = "16bit"
    elif image.mode == MODE_FLOAT:
        linearized = img_pixels.astype(np.float32)
        branch = "float"
    elif image.mode in [MODE_RGB, MODE_RGBA, MODE_GRAYSCALE]:
        linearized = INVERSE_GAMMA_LUT[img_pixels.astype(np.uint8)]
        branch = "srgb"
    else:
        try:
            image = image.convert(MODE_RGB)
            img_pixels = np.array(image)
            linearized = INVERSE_GAMMA_LUT[img_pixels.astype(np.uint8)]
            branch = "converted"
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

    # Clamp to the documented [0, 1] range: 'I' (up to 2**31-1) and 'F' (any
    # float) can exceed full scale, which would break the max_lux ceiling.
    return np.clip(y_plane, 0.0, 1.0), source_mode, branch


def _trimmed_mean(y_plane: np.ndarray, trim_top_fraction: float) -> float:
    """Mean of the Y plane with the brightest trim_top_fraction dropped."""
    if trim_top_fraction <= 0.0:
        return float(np.mean(y_plane))
    flat = np.sort(y_plane.ravel())
    keep = flat[: max(1, int(flat.size * (1.0 - trim_top_fraction)))]
    return float(np.mean(keep))


def calculate_image_luminance(
    image: Image.Image, trim_top_fraction: float = 0.0
) -> float:
    """Color-frame luminance: spatial mean of the linear-light Y plane.

    A positive trim_top_fraction drops the brightest pixels first (the
    highlight-resistant algorithms: sun patches through windows skew a plain
    mean). The perception transform (avg ** exponent) is applied by the caller
    so the effective exponent can depend on IR detection.
    """
    y_plane, source_mode, branch = _linearized_y_plane(image)
    avg_lum = _trimmed_mean(y_plane, trim_top_fraction)
    _LOGGER.debug(
        "luminance avg=%f from mode %s (%s branch)", avg_lum, source_mode, branch
    )
    return avg_lum


def compute_gray_fraction(image: Image.Image) -> float:
    """Fraction of pixels that are individually gray (max-min channel <= 2).

    Distinguishes a true IR/monochrome frame (every pixel gray, ~1.0) from a
    dark or muted COLOR frame whose mean chroma is also low: sensor noise and
    scattered colored pixels break per-pixel grayness even when the average
    chroma is small. The <=2 tolerance absorbs JPEG chroma-subsampling error.
    """
    if image.mode == "P":
        image = image.convert("RGB")
    if image.mode in _COLORLESS_MODES or len(image.getbands()) < 3:
        return 1.0
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    spread = arr.max(axis=2) - arr.min(axis=2)
    return float(np.mean(spread <= 2))


def calculate_ir_image_luminance(image: Image.Image, statistic: str = "trim") -> float:
    """IR/monochrome-frame luminance under the chosen statistic.

    The camera's illuminator paints hotspots (near surfaces, reflective
    objects, subjects in the beam) onto an otherwise dark scene; on sampled
    night footage those drag a plain mean 1.3-2.6x above the background
    level, and a passing subject moves the mean with zero ambient change.
    The statistic decides how those hotspots weigh in:
      "trim"   -- mean with the brightest IR_TRIM_TOP_FRACTION dropped:
                  point hotspots ignored, broad illumination kept
      "median" -- the scene's background floor: subjects invisible
      "mean"   -- plain mean: responds to everything the illuminator lights
    """
    y_plane, source_mode, branch = _linearized_y_plane(image)
    if statistic == "median":
        avg_lum = float(np.median(y_plane))
    elif statistic == "mean":
        avg_lum = float(np.mean(y_plane))
    else:
        avg_lum = _trimmed_mean(y_plane, IR_TRIM_TOP_FRACTION)
    _LOGGER.debug(
        "IR luminance avg=%f (%s) from mode %s (%s branch)",
        avg_lum,
        statistic,
        source_mode,
        branch,
    )
    return avg_lum


def process_image(
    image_bytes: bytes,
    roi: Optional[Dict[str, int]],
    max_pixels: int,
    ir_chroma_threshold: Optional[float] = None,
    color_trim_fraction: float = 0.0,
    ir_statistic: str = "trim",
) -> Tuple[float, float, bool]:
    """Decode and measure a snapshot in one call (run me in an executor).

    Returns (avg_luminance, chroma, ir_frame). Chroma is measured on the FULL
    frame (an ROI can be legitimately colorless without the camera being in IR
    mode); luminance honors the ROI. When ir_chroma_threshold is given, a frame
    counts as IR only if its chroma falls below the threshold AND nearly all
    pixels are individually gray (IR_GRAY_FRACTION_MIN) — low mean chroma alone
    also matches dark/muted COLOR frames. IR frames use the IR luminance
    method; pass None to always use the color method. color_trim_fraction is
    forwarded to the color method (highlight-resistant algorithms). The
    perception transform is applied by the caller once the effective exponent
    is known.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image.load()

    full_small = resize_image(image, max_pixels)
    chroma = compute_chroma(full_small)
    ir_frame = (
        ir_chroma_threshold is not None
        and chroma < ir_chroma_threshold
        and compute_gray_fraction(full_small) >= IR_GRAY_FRACTION_MIN
    )

    if roi:
        lum_source = resize_image(crop_image(image, roi), max_pixels)
    else:
        lum_source = full_small
    if ir_frame:
        return calculate_ir_image_luminance(lum_source, ir_statistic), chroma, True
    return calculate_image_luminance(lum_source, color_trim_fraction), chroma, False
