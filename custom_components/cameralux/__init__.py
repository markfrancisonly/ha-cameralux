"""Camera Lux integration setup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

from .const import PLATFORMS


async def async_setup_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> bool:
    """Set up Camera Lux from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_reload_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> None:
    """Reload Camera Lux when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> bool:
    """Unload a Camera Lux config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_migrate_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> bool:
    """Delegate old-entry migration to the isolated compatibility package."""
    # Compatibility seam: Home Assistant requires this hook at the integration
    # root. The import remains deferred for every current entry setup.
    from .legacy_v1.migration import async_migrate_entry as migrate_legacy_entry

    return await migrate_legacy_entry(hass, entry)
