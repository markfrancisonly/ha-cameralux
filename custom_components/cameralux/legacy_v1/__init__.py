"""Severable compatibility boundary for Camera Lux 1.x entries.

Delete this package, the migration re-export in the integration root, and the
factory call in sensor.async_setup_entry when 1.x compatibility is retired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from ..sensor import CameraLuxSensor

LEGACY_ALGORITHM = "legacy_v1"


def create_sensor_if_legacy(
    hass: "HomeAssistant",
    name: str,
    config: dict[str, Any],
    stable_unique_id: str,
) -> "CameraLuxSensor | None":
    """Build the compatibility entity only when a migrated entry requests it."""
    if config.get("algorithm") != LEGACY_ALGORITHM:
        return None

    # Deferred to avoid a circular import while the modern sensor module loads.
    from .sensor import V1CameraLuxSensor

    return V1CameraLuxSensor(hass, name, config, stable_unique_id)
