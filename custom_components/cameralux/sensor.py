# custom_components/cameralux/sensor.py
"""CameraLux sensor entity for Home Assistant."""

from __future__ import annotations

import asyncio
import io
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import aiohttp
import homeassistant.helpers.config_validation as cv
import voluptuous as vol
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CALLBACK_TYPE, CoreState, HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from PIL import Image, UnidentifiedImageError

# Unit compatibility shim (older HA uses LIGHT_LUX constant)
try:
    from homeassistant.const import UnitOfIlluminance  # HA 2023.9+

    UNIT_LUX = UnitOfIlluminance.LUX
except Exception:  # noqa: BLE001
    from homeassistant.const import LIGHT_LUX as UNIT_LUX

from .const import (
    CONF_BRIGHTNESS_ROI,
    CONF_CALIBRATION_FACTOR,
    CONF_ENTITY_ID,
    CONF_HEIGHT,
    CONF_IMAGE_URL,
    CONF_ROI_ENABLED,
    CONF_SENSORS,
    CONF_SOURCE,
    CONF_UNAVAILABLE_ABOVE,
    CONF_UNAVAILABLE_BELOW,
    CONF_UPDATE_INTERVAL,
    CONF_WIDTH,
    CONF_X,
    CONF_Y,
    DEFAULT_CALIBRATION_FACTOR,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
    MAX_PIXELS,
    SOURCE_CAMERA,
    SOURCE_URL,
    SUGGESTED_DISPLAY_PRECISION,
)
from .helpers import (
    build_unique_id,
    calculate_image_luminance,
    coerce_float,
    crop_image,
    get_bounded_int,
    resize_image,
)

_LOGGER = logging.getLogger(__name__)


# YAML schemas (kept for import support)
BRIGHTNESS_ROI_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_ROI_ENABLED, default=False): cv.boolean,
        vol.Optional(CONF_X): vol.Coerce(int),
        vol.Optional(CONF_Y): vol.Coerce(int),
        vol.Optional(CONF_WIDTH): vol.Coerce(int),
        vol.Optional(CONF_HEIGHT): vol.Coerce(int),
    }
)

SENSOR_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_ENTITY_ID): cv.entity_id,
        vol.Optional(CONF_IMAGE_URL): cv.url,
        vol.Optional(CONF_BRIGHTNESS_ROI): BRIGHTNESS_ROI_SCHEMA,
        vol.Optional(
            CONF_CALIBRATION_FACTOR, default=DEFAULT_CALIBRATION_FACTOR
        ): cv.positive_float,
        vol.Optional(CONF_UPDATE_INTERVAL, default=DEFAULT_UPDATE_INTERVAL): vol.All(
            vol.Coerce(int), vol.Range(min=5, max=3600)
        ),
        vol.Optional(CONF_UNAVAILABLE_BELOW): vol.All(
            vol.Coerce(float), vol.Range(min=0)
        ),
        vol.Optional(CONF_UNAVAILABLE_ABOVE): vol.All(
            vol.Coerce(float), vol.Range(min=0)
        ),
    },
    extra=vol.ALLOW_EXTRA,
)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {vol.Required(CONF_SENSORS): vol.Schema({vol.Required(cv.string): SENSOR_SCHEMA})}
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
):
    """Import YAML-defined sensors into config entries (one-time conversion)."""
    _LOGGER.debug("CameraLux YAML import starting")
    for name, sensor_config in config[CONF_SENSORS].items():
        data = {
            "name": name,
            CONF_ENTITY_ID: sensor_config.get(CONF_ENTITY_ID),
            CONF_IMAGE_URL: sensor_config.get(CONF_IMAGE_URL),
            CONF_BRIGHTNESS_ROI: sensor_config.get(CONF_BRIGHTNESS_ROI, {}),
            CONF_CALIBRATION_FACTOR: sensor_config.get(
                CONF_CALIBRATION_FACTOR, DEFAULT_CALIBRATION_FACTOR
            ),
            CONF_UPDATE_INTERVAL: sensor_config.get(
                CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL
            ),
            CONF_UNAVAILABLE_BELOW: sensor_config.get(CONF_UNAVAILABLE_BELOW),
            CONF_UNAVAILABLE_ABOVE: sensor_config.get(CONF_UNAVAILABLE_ABOVE),
        }
        hass.async_create_task(
            hass.config_entries.flow.async_init(
                DOMAIN,
                context={"source": "import"},
                data=data,
            )
        )
    return


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
):
    """Create entities from a UI config entry and its options."""
    data = {**entry.data, **entry.options}

    source = data.get(CONF_SOURCE)
    if source not in (SOURCE_CAMERA, SOURCE_URL):
        source = SOURCE_CAMERA if data.get(CONF_ENTITY_ID) else SOURCE_URL

    camera_entity = data.get(CONF_ENTITY_ID) if source == SOURCE_CAMERA else None
    image_url = data.get(CONF_IMAGE_URL) if source == SOURCE_URL else None

    if camera_entity and image_url:
        _LOGGER.warning(
            "Both camera entity and image URL provided; preferring camera entity '%s'.",
            camera_entity,
        )
        image_url = None

    cfg = {
        CONF_ENTITY_ID: camera_entity,
        CONF_IMAGE_URL: image_url,
        CONF_BRIGHTNESS_ROI: data.get(CONF_BRIGHTNESS_ROI, {}),
        CONF_CALIBRATION_FACTOR: data.get(
            CONF_CALIBRATION_FACTOR, DEFAULT_CALIBRATION_FACTOR
        ),
        CONF_UPDATE_INTERVAL: data.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
        CONF_UNAVAILABLE_BELOW: data.get(CONF_UNAVAILABLE_BELOW),
        CONF_UNAVAILABLE_ABOVE: data.get(CONF_UNAVAILABLE_ABOVE),
    }

    # Use entry.entry_id as the stable unique_id so Options edits never change the entity id.
    async_add_entities(
        [
            CameraLuxSensor(
                hass,
                entry.data.get("name") or "Camera Lux",
                cfg,
                stable_unique_id=entry.entry_id,
            )
        ]
    )


class CameraLuxSensor(SensorEntity):
    """Camera-based illuminance sensor that samples an image from a camera entity or an image URL."""

    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.ILLUMINANCE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UNIT_LUX
    _attr_suggested_display_precision = SUGGESTED_DISPLAY_PRECISION

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        config: dict,
        stable_unique_id: str | None = None,
    ):
        """Initialize the sensor from a validated config dict."""
        super().__init__()
        self.hass = hass
        self._attr_name = name

        self.lux: float | None = None
        self.avg_luminance: float | None = None
        self.perceived_luminance: float | None = None

        self.remove_interval_update: CALLBACK_TYPE | None = None
        self.is_updating = False

        self.camera_entity_id = config.get(CONF_ENTITY_ID) or None
        self.image_url = (
            (config.get(CONF_IMAGE_URL) or None) if not self.camera_entity_id else None
        )

        roi_cfg = config.get(CONF_BRIGHTNESS_ROI) or {}
        self.roi_enabled: bool = bool(roi_cfg.get(CONF_ROI_ENABLED, False))
        self.roi = {
            CONF_X: get_bounded_int(roi_cfg.get(CONF_X), 0, 0, None),
            CONF_Y: get_bounded_int(roi_cfg.get(CONF_Y), 0, 0, None),
            CONF_WIDTH: get_bounded_int(roi_cfg.get(CONF_WIDTH), 0, 0, None),
            CONF_HEIGHT: get_bounded_int(roi_cfg.get(CONF_HEIGHT), 0, 0, None),
        }

        self.luminance_to_lux_calibration_factor = max(
            0.0,
            coerce_float(
                config.get(CONF_CALIBRATION_FACTOR, DEFAULT_CALIBRATION_FACTOR),
                DEFAULT_CALIBRATION_FACTOR,
            ),
        )
        self.update_interval = get_bounded_int(
            config.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
            DEFAULT_UPDATE_INTERVAL,
            5,
            3600,
        )
        unavail_below_cfg = config.get(CONF_UNAVAILABLE_BELOW)
        unavail_above_cfg = config.get(CONF_UNAVAILABLE_ABOVE)
        try:
            self.unavailable_below: float | None = (
                float(unavail_below_cfg) if unavail_below_cfg is not None else None
            )
        except (TypeError, ValueError):
            self.unavailable_below = None
        try:
            self.unavailable_above: float | None = (
                float(unavail_above_cfg) if unavail_above_cfg is not None else None
            )
        except (TypeError, ValueError):
            self.unavailable_above = None

        if not self.camera_entity_id and not self.image_url:
            _LOGGER.error(
                "CameraLux '%s' has neither camera entity nor image URL configured; sensor will stay unavailable.",
                name,
            )

        self._attr_unique_id = stable_unique_id or build_unique_id(
            name=name,
            camera_entity_id=self.camera_entity_id,
            image_url=self.image_url,
        )

        if self.camera_entity_id:
            _LOGGER.info(
                "CameraLux '%s': calibration %.3f (camera %s, ROI %s)",
                name,
                self.luminance_to_lux_calibration_factor,
                self.camera_entity_id,
                "enabled" if self.roi_enabled else "disabled",
            )
        elif self.image_url:
            _LOGGER.info(
                "CameraLux '%s': calibration %.3f (image URL, ROI %s)",
                name,
                self.luminance_to_lux_calibration_factor,
                "enabled" if self.roi_enabled else "disabled",
            )

    @property
    def available(self) -> bool:
        """Return True when a lux value has been computed."""
        return self.lux is not None

    @property
    def native_value(self) -> float | None:
        """Return the current illuminance in lux."""
        return self.lux

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra attributes for debugging and calibration."""
        if self.available:
            return {
                "calibration_factor": self.luminance_to_lux_calibration_factor,
                "avg_luminance": self.avg_luminance,
                "perceived_luminance": self.perceived_luminance,
                "roi_enabled": self.roi_enabled,
                "roi": self.roi,
                "unavailable_below": self.unavailable_below,
                "unavailable_above": self.unavailable_above,
            }
        return {}

    async def async_added_to_hass(self):
        """Start periodic updates when the entity is added to Home Assistant."""

        async def start_polling(_=None):
            self.hass.async_create_task(self.async_update())
            self.remove_interval_update = async_track_time_interval(
                self.hass, self.async_update, timedelta(seconds=self.update_interval)
            )

        if self.hass.state is not CoreState.running:
            self.async_on_remove(
                self.hass.bus.async_listen_once(
                    EVENT_HOMEASSISTANT_STARTED, start_polling
                )
            )
        else:
            await start_polling()

    async def async_will_remove_from_hass(self):
        """Stop periodic updates when the entity is removed."""
        if self.remove_interval_update:
            self.remove_interval_update()
            self.remove_interval_update = None

    async def async_update(self, _: datetime | None = None) -> None:
        """Fetch an image, compute luminance, and update the lux value."""
        if self.hass.state is not CoreState.running:
            _LOGGER.debug(
                "Skipping '%s' update - Home Assistant not running", self._attr_name
            )
            return
        if self.is_updating:
            _LOGGER.debug("'%s' update already in progress; skipping.", self._attr_name)
            return

        try:
            self.is_updating = True
            image: Image.Image | None = None

            if self.camera_entity_id:
                image = await self.async_get_image_from_camera_entity_id(
                    self.camera_entity_id
                )
            elif self.image_url:
                image = await self.async_fetch_image_from_url(self.image_url)
            else:
                self.avg_luminance = None
                self.perceived_luminance = None
                self.lux = None
                return

            if image:
                roi_for_calc = self.roi if self.roi_enabled else None
                cropped_image = crop_image(image, roi_for_calc)
                resized_image = resize_image(cropped_image, MAX_PIXELS)

                self.avg_luminance, self.perceived_luminance = (
                    calculate_image_luminance(resized_image)
                )
                self.lux = (
                    self.perceived_luminance * self.luminance_to_lux_calibration_factor
                )
                if (
                    self.unavailable_below is not None
                    and self.lux < self.unavailable_below
                ) or (
                    self.unavailable_above is not None
                    and self.lux > self.unavailable_above
                ):
                    _LOGGER.debug(
                        "%s lux %s hit unavailable threshold; setting unavailable",
                        self._attr_name,
                        self.lux,
                    )
                    self.avg_luminance = None
                    self.perceived_luminance = None
                    self.lux = None
                else:
                    _LOGGER.debug(
                        "%s updated to %s %s", self._attr_name, self.lux, UNIT_LUX
                    )
            else:
                self.avg_luminance = None
                self.perceived_luminance = None
                self.lux = None

        except asyncio.CancelledError:
            # Propagate task cancellation to avoid blocking shutdown
            raise
        except Exception as e:  # noqa: BLE001
            self.avg_luminance = None
            self.perceived_luminance = None
            self.lux = None
            _LOGGER.error("Error updating CameraLux '%s': %s", self._attr_name, e)
        finally:
            self.is_updating = False
            self.async_schedule_update_ha_state(force_refresh=True)

    async def async_get_image_from_camera_entity_id(
        self, entity_id: str
    ) -> Image.Image | None:
        """Fetch a snapshot from a Home Assistant camera entity."""
        component = self.hass.data.get("camera")
        if component is None:
            _LOGGER.critical("Camera integration not set up")
            return None

        camera = component.get_entity(entity_id)
        if camera is None:
            _LOGGER.error("Camera entity %s not found", entity_id)
            return None

        try:
            image_bytes = await camera.async_camera_image()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # camera may raise HomeAssistantError
            _LOGGER.error("Error retrieving image from camera %s: %s", entity_id, e)
            return None

        if not image_bytes:
            _LOGGER.warning("Camera %s image not ready", entity_id)
            return None

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            _LOGGER.debug(
                "%s retrieved image from camera %s", self._attr_name, entity_id
            )
            return image
        except (UnidentifiedImageError, OSError, ValueError) as e:
            _LOGGER.error("Invalid image bytes from camera %s: %s", entity_id, e)
            return None

    async def async_fetch_image_from_url(self, url: str) -> Image.Image | None:
        """Fetch an image from a URL using HA's shared aiohttp session."""
        session = async_get_clientsession(self.hass)
        try:
            async with session.get(url, timeout=15) as response:
                if response.status != 200:
                    _LOGGER.error(
                        "Failed to fetch %s image from URL '%s': HTTP %d",
                        self._attr_name,
                        url,
                        response.status,
                    )
                    return None
                content = await response.read()
        except aiohttp.ClientError as e:
            _LOGGER.error(
                "HTTP error while fetching %s image from URL '%s': %s",
                self._attr_name,
                url,
                e,
            )
            return None
        except asyncio.TimeoutError:
            _LOGGER.error(
                "Timeout while fetching %s image from URL '%s'", self._attr_name, url
            )
            return None

        try:
            image = Image.open(io.BytesIO(content))
            image.load()
            _LOGGER.debug("%s fetched image from URL '%s'", self._attr_name, url)
            return image
        except (UnidentifiedImageError, OSError, ValueError) as e:
            _LOGGER.error("Invalid image content from URL '%s': %s", url, e)
            return None
