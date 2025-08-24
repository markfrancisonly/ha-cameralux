# custom_components/cameralux/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - optional Home Assistant import for typing
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

from .const import PLATFORMS


async def async_setup(hass: "HomeAssistant", config: dict) -> bool:
    # Explicitly ignore YAML configuration to prevent import flow
    return True


async def async_setup_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> bool:
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    # Reload entities when options change
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_reload_entry(hass: "HomeAssistant", entry: "ConfigEntry"):
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
