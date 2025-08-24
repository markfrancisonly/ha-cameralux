# custom_components/cameralux/const.py
from __future__ import annotations

# Domain / platforms
DOMAIN = "cameralux"
PLATFORMS = ["sensor"]

# Config keys (shared)
CONF_SENSORS = "sensors"
CONF_ENTITY_ID = "entity_id"
CONF_IMAGE_URL = "image_url"

# ROI (nested) keys
CONF_BRIGHTNESS_ROI = "brightness_roi"
CONF_ROI_ENABLED = "enabled"
CONF_X = "x"
CONF_Y = "y"
CONF_WIDTH = "width"
CONF_HEIGHT = "height"

# Other config
CONF_CALIBRATION_FACTOR = "calibration_factor"
CONF_UPDATE_INTERVAL = "update_interval"
CONF_UNAVAILABLE_BELOW = "unavailable_below"
CONF_UNAVAILABLE_ABOVE = "unavailable_above"

# UI-only helper keys (persisted in entry data/options as needed)
CONF_SOURCE = "source"  # "camera" or "url"
SOURCE_CAMERA = "camera"
SOURCE_URL = "url"

# Defaults / behavior
MAX_PIXELS = 250_000
SUGGESTED_DISPLAY_PRECISION = 3
DEFAULT_CALIBRATION_FACTOR = 2000
DEFAULT_UPDATE_INTERVAL = 30  # seconds
