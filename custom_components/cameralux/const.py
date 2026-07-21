# custom_components/cameralux/const.py
from __future__ import annotations

# Domain / platforms
DOMAIN = "cameralux"
PLATFORMS = ["sensor"]

# Schema version of the stored config entry.
CONFIG_ENTRY_VERSION = 2

# Config keys (shared)
CONF_ENTITY_ID = "entity_id"
CONF_IMAGE_URL = "image_url"

# ROI (nested) keys
CONF_BRIGHTNESS_ROI = "brightness_roi"
CONF_ROI_ENABLED = "enabled"
CONF_X = "x"
CONF_Y = "y"
CONF_WIDTH = "width"
CONF_HEIGHT = "height"

# Other config.
# full_scale_lux is the lux reported for a pure-white frame. Because
# avg_luminance is normalized to [0, 1] and 1.0 ** exponent == 1.0, the exponent
# drops out at full scale: it is both the calibration knob and the sensor's
# maximum reading. Exposed as the `max_lux` state attribute.
CONF_FULL_SCALE_LUX = "full_scale_lux"
# Named brightness algorithm: picks the (statistic, exponent) pair for color
# frames. The named algorithms are the whole public vocabulary.
CONF_ALGORITHM = "algorithm"
CONF_UPDATE_INTERVAL = "update_interval"
CONF_UNAVAILABLE_BELOW = "unavailable_below"
CONF_UNAVAILABLE_ABOVE = "unavailable_above"

# IR handling.
# chroma = (mean|Cb-128| + mean|Cr-128|) / 2; below threshold (verified by
# per-pixel grayness) = IR/monochrome frame. Detection runs only when
# CONF_IR_ENABLED is on — cameras with no IR mode (tablets, wall panels,
# fixed-exposure) skip the per-frame work and can never false-positive into
# an IR policy. Cameras flip to IR at different chroma levels, so the
# threshold is per-camera. When enabled, the IR algorithm is the policy
# applied after an IR frame is detected:
#   a named algorithm  -- measure the IR frame with that curve (shares the
#                         one full_scale_lux; the normal unavailable bounds
#                         apply)
#   "fixed"            -- report CONF_IR_FIXED_LUX, a declared policy value
#                         exempt from the unavailable bounds (default 0)
#   "unavailable"      -- take the sensor offline rather than report a number
#                         nobody measured (the honesty valve)
CONF_IR_ENABLED = "ir_enabled"
DEFAULT_IR_ENABLED = False
CONF_IR_ALGORITHM = "ir_algorithm"
IR_ALGORITHM_FIXED = "fixed"
IR_ALGORITHM_UNAVAILABLE = "unavailable"
DEFAULT_IR_ALGORITHM = IR_ALGORITHM_FIXED
CONF_IR_FIXED_LUX = "ir_fixed_lux"
DEFAULT_IR_FIXED_LUX = 0.0
CONF_IR_CHROMA_THRESHOLD = "ir_chroma_threshold"

# UI-only helper keys (persisted in entry data/options as needed)
CONF_SOURCE = "source"  # "camera" or "url"
SOURCE_CAMERA = "camera"
SOURCE_URL = "url"

# Defaults / behavior
MAX_PIXELS = 250_000
SUGGESTED_DISPLAY_PRECISION = 3
MIN_FULL_SCALE_LUX = 0.001
MIN_CHROMA_THRESHOLD = 0.0
MAX_CHROMA_THRESHOLD = 64.0
MIN_LUX_BOUND = 0.0
# Lux reported for a pure-white frame; anchors a comfortably-lit indoor frame
# (avg_lin ~0.04-0.09 -> perceived ~0.2-0.3 at exponent 0.5) to ~200-300 real lux.
DEFAULT_FULL_SCALE_LUX = 1000
DEFAULT_UPDATE_INTERVAL = 30  # seconds
# Stevens'-law brightness exponent: perceived = statistic(luminance) ** exponent.
# 0.5 approximates human brightness perception for extended scenes;
# 1.0 is physically-linear lux (best for fixed-exposure / no-IR cameras, whose
# night frames stay on the color path — a compressive exponent would read a
# dark room at 40-140 lx).
# Named algorithms: value stored in the entry -> (color trim fraction, exponent).
ALGORITHM_HUMAN = "human_brightness"
ALGORITHM_HUMAN_HR = "human_brightness_hr"  # highlight-resistant (sun patches)
ALGORITHM_LINEAR = "linear"
DEFAULT_ALGORITHM = ALGORITHM_HUMAN
ALGORITHM_PROFILES = {
    ALGORITHM_HUMAN: (0.0, 0.5),
    ALGORITHM_HUMAN_HR: (0.05, 0.5),
    ALGORITHM_LINEAR: (0.0, 1.0),
}
# Calibrated live against a mixed camera fleet: IR/monochrome frames read
# chroma 0.75-3.7, color scenes 7.3-16.6.
DEFAULT_IR_CHROMA_THRESHOLD = 6.0
# IR-frame handling, validated against sampled recordings from that fleet:
# a frame only counts as IR when nearly every pixel is individually gray —
# true IR frames measure ~1.0, while dark color frames (sensor noise) and
# muted color scenes stay <= 0.67. This guard lets one default chroma
# threshold work across cameras that would otherwise need per-camera values.
IR_GRAY_FRACTION_MIN = 0.85
# IR illuminators paint hotspots (near surfaces, subjects in the beam) that
# drag a plain mean 1.3-2.6x above the scene's background level; the IR
# luminance statistic therefore drops the brightest pixels.
IR_TRIM_TOP_FRACTION = 0.05
