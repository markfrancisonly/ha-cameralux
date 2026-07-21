"""Config-entry migration for the severable Camera Lux 1.x bridge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

from ..const import (
    CONF_ALGORITHM,
    CONF_IR_ALGORITHM,
    CONF_IR_ENABLED,
    CONF_IR_FIXED_LUX,
    CONFIG_ENTRY_VERSION,
    DEFAULT_IR_FIXED_LUX,
)
from . import LEGACY_ALGORITHM

_LOGGER = logging.getLogger(__name__)

# Frozen 1.x and abandoned pre-release spellings. They intentionally do not
# leak into the current constants or config flow.
_V1_FULL_SCALE = "calibration_factor"
_FULL_SCALE = "full_scale_lux"
_RETIRED_KEYS = (
    "ir_calibration_factor",
    "ir_full_scale_lux",
    "ir_exponent",
    "ir_unavailable_below",
    "ir_unavailable_above",
)
_RETIRED_IR_ALGORITHMS = (
    "ir_ambient",
    "ir_background",
    "ir_scene",
    "custom",
)


def _migrate_1_to_2(data: dict, options: dict, title: str) -> None:
    """Convert a 1.x entry without changing its response curve."""
    merged = {**data, **options}
    if merged.get(CONF_ALGORITHM) is None:
        data[CONF_ALGORITHM] = LEGACY_ALGORITHM
        data[CONF_IR_ALGORITHM] = LEGACY_ALGORITHM
        _LOGGER.info(
            "CameraLux '%s': retaining its 1.x response curve; choose a current "
            "algorithm in Options when ready to recalibrate.",
            title,
        )

    data[CONF_IR_FIXED_LUX] = DEFAULT_IR_FIXED_LUX
    if merged.get(CONF_IR_ENABLED) is None:
        data[CONF_IR_ENABLED] = False

    for mapping in (data, options):
        if _V1_FULL_SCALE in mapping:
            mapping[_FULL_SCALE] = mapping.pop(_V1_FULL_SCALE)
        if mapping.get(CONF_IR_ALGORITHM) in _RETIRED_IR_ALGORITHMS:
            mapping.pop(CONF_IR_ALGORITHM)
        for key in _RETIRED_KEYS:
            mapping.pop(key, None)


async def async_migrate_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> bool:
    """Bring a Camera Lux config entry up to the current schema version."""
    if entry.version > CONFIG_ENTRY_VERSION:
        _LOGGER.error(
            "CameraLux entry '%s' is schema v%s but this build only understands "
            "v%s; downgrade is not supported.",
            entry.title,
            entry.version,
            CONFIG_ENTRY_VERSION,
        )
        return False
    if entry.version == CONFIG_ENTRY_VERSION:
        return True

    from_version = entry.version
    data = dict(entry.data)
    options = dict(entry.options)

    if from_version < 2:
        _migrate_1_to_2(data, options, entry.title)

    hass.config_entries.async_update_entry(
        entry,
        data=data,
        options=options,
        version=CONFIG_ENTRY_VERSION,
    )
    _LOGGER.info(
        "CameraLux '%s' migrated from config entry schema v%s to v%s.",
        entry.title,
        from_version,
        CONFIG_ENTRY_VERSION,
    )
    return True
