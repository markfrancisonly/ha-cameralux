"""Platform for sensor integration."""

from __future__ import annotations

import io
import math
import logging
import asyncio
from datetime import timedelta, datetime
from typing import Optional, Dict

import aiohttp
import numpy as np
from PIL import Image
import hashlib

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
    PLATFORM_SCHEMA,
)
from homeassistant.const import LIGHT_LUX, EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CoreState, CALLBACK_TYPE, HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_interval
import homeassistant.helpers.config_validation as cv
import voluptuous as vol

_LOGGER = logging.getLogger(__name__)

# Configuration Constants
CONF_SENSORS = "sensors"
CONF_ENTITY_ID = "entity_id"
CONF_IMAGE_URL = "image_url"
CONF_BRIGHTNESS_ROI = "brightness_roi"
CONF_X = "x"
CONF_Y = "y"
CONF_WIDTH = "width"
CONF_HEIGHT = "height"
CONF_CALIBRATION_FACTOR = "calibration_factor"
CONF_UPDATE_INTERVAL = "update_interval"

# Default Values
MAX_PIXELS = 250000
SUGGESTED_DISPLAY_PRECISION = 3
DEFAULT_CALIBRATION_FACTOR = 2000
DEFAULT_UPDATE_INTERVAL = 30  # seconds


# Configuration Schemas
BRIGHTNESS_ROI_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_X): cv.positive_int,
        vol.Required(CONF_Y): cv.positive_int,
        vol.Required(CONF_WIDTH): cv.positive_int,
        vol.Required(CONF_HEIGHT): cv.positive_int,
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
    },
    extra=vol.ALLOW_EXTRA,
)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {vol.Required(CONF_SENSORS): vol.Schema({vol.Required(cv.string): SENSOR_SCHEMA})}
)


def precompute_inverse_gamma_8bit_lut() -> np.ndarray:
    """Generate an inverse gamma LUT to linearize sRGB values."""
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized = i / 255.0
        if normalized <= 0.04045:
            lut[i] = normalized / 12.92
        else:
            lut[i] = ((normalized + 0.055) / 1.055) ** 2.4
    return lut


INVERSE_GAMMA_LUT = precompute_inverse_gamma_8bit_lut()


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
):
    """Set up the Camera Lux sensor platform."""
    _LOGGER.debug("Camera Lux platform setup started")

    sensors = []
    for name, sensor_config in config[CONF_SENSORS].items():

        try:
            sensor = CameraLuxSensor(hass, name, sensor_config)
            sensors.append(sensor)

        except Exception as e:
            _LOGGER.error(
                "Failed to initialize CameraLux sensor '%s': %s",
                name,
                e,
            )
            continue

    async_add_entities(sensors)


class CameraLuxSensor(SensorEntity):
    """Representation of a Camera-based Lux Sensor."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        config: dict,
    ):
        super().__init__()

        """Initialize the sensor."""
        self.hass = hass
        self._name = name

        self._lux: float | None = None
        self._avg_luminance: float | None = None
        self._perceived_luminance: float | None = None

        self._remove_interval_update: CALLBACK_TYPE | None = None
        self._is_updating = False

        # Extract configuration parameters with defaults
        self._camera_entity_id = config.get(CONF_ENTITY_ID)
        self._image_url = config.get(CONF_IMAGE_URL)
        self._roi = config.get(CONF_BRIGHTNESS_ROI)
        self._luminance_to_lux_calibration_factor: float = config.get(
            CONF_CALIBRATION_FACTOR, DEFAULT_CALIBRATION_FACTOR
        )
        self._update_interval = config.get(
            CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL
        )

        if not self._camera_entity_id and not self._image_url:
            raise ValueError(
                f"CameraLux sensor '{self.name}' must have either 'entity_id' or 'image_url' configured."
            )

        self._unique_id = self.create_unique_id()

        if self._camera_entity_id:
            _LOGGER.info(
                "CameraLux sensor %s initialized with calibration factor %.3f using camera entity %s",
                name,
                self._luminance_to_lux_calibration_factor,
                self._camera_entity_id,
            )
        if self._image_url:
            _LOGGER.info(
                "CameraLux sensor %s initialized with calibration factor %.3f using image URL %s",
                name,
                self._luminance_to_lux_calibration_factor,
                self._image_url,
            )

    def create_unique_id(self) -> str | None:
        """Return a unique ID for the sensor."""
        if self._camera_entity_id:
            return f"cameralux_{self._camera_entity_id.replace('.', '_')}"

        if self._image_url:
            url_hash = hashlib.md5(self._image_url.encode()).hexdigest()
            return f"cameralux_url_{url_hash}"
        return None

    @property
    def should_poll(self) -> bool:
        """No polling needed. Updates handled by interval."""
        return False

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def unique_id(self) -> str:
        """Return the unique ID of the sensor."""
        return self._unique_id

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return LIGHT_LUX

    @property
    def state_class(self) -> str:
        """Return the state class."""
        return SensorStateClass.MEASUREMENT

    @property
    def device_class(self) -> str:
        """Return the device class."""
        return SensorDeviceClass.ILLUMINANCE

    @property
    def native_value(self) -> float | None:
        """Return the native value."""
        return self._lux

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes of the sensor."""
        if self.available:
            return {
                "calibration_factor": self._luminance_to_lux_calibration_factor,
                "avg_luminance": f"{self._avg_luminance:.6f}",
                "perceived_luminance": f"{self._perceived_luminance:.6f}",
            }
        else:
            return {}

    @property
    def suggested_display_precision(self) -> int:
        """Return the suggested display precision."""
        return SUGGESTED_DISPLAY_PRECISION

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._lux is not None

    async def async_added_to_hass(self):
        """Register callbacks when entity is added to hass."""

        async def start_polling(_=None):
            self.hass.async_create_task(self.async_update())

            # Schedule periodic updates based on the update_interval
            self._remove_interval_update = async_track_time_interval(
                self.hass, self.async_update, timedelta(seconds=self._update_interval)
            )

        if self.hass.state is not CoreState.running:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, start_polling)
        else:
            await start_polling()

    async def async_will_remove_from_hass(self):
        """Cleanup when entity is removed from hass."""
        if self._remove_interval_update:
            self._remove_interval_update()

    async def async_update(self, _: datetime | None = None) -> None:
        """Fetch new state data for the sensor."""

        if self.hass.state is not CoreState.running:
            _LOGGER.debug(
                "Skipping CameraLux sensor '%s' update - Home Assistant not fully started",
                self.name,
            )
            return

        if self._is_updating:
            _LOGGER.debug(
                "CameraLux sensor '%s' update already in progress. Skipping this update.",
                self.name,
            )
            return

        try:
            self._is_updating = True

            if self._camera_entity_id:
                # Retrieve the camera entity using the helper method
                image = await self.async_get_image_from_camera_entity_id(
                    self._camera_entity_id
                )

            elif self._image_url:
                # Fetch the image from the provided HTTP URL
                image = await self.async_fetch_image_from_url(self._image_url)

            if image:
                # Calculate lux using average pixel luminance
                cropped_image = self.crop_image(image, self._roi)
                resized_image = self.resize_image(cropped_image, MAX_PIXELS)

                self._avg_luminance, self._perceived_luminance = (
                    self.calculate_image_luminance(resized_image)
                )
                self._lux = (
                    self._perceived_luminance
                    * self._luminance_to_lux_calibration_factor
                )

                _LOGGER.debug("%s updated to %s %s", self.name, self._lux, LIGHT_LUX)

            else:
                self._avg_luminance = None
                self._perceived_luminance = None
                self._lux = None

        except Exception as e:
            self._avg_luminance = None
            self._perceived_luminance = None
            self._lux = None

            _LOGGER.error(
                "Error updating CameraLux sensor '%s': %s",
                self.name,
                e,
            )

        finally:
            self._is_updating = False
            self.async_schedule_update_ha_state(force_refresh=True)

    async def async_get_image_from_camera_entity_id(
        self, entity_id: str
    ) -> Image.Image | None:
        """Retrieve the Camera entity from the entity ID."""

        if (component := self.hass.data.get("camera")) is None:
            _LOGGER.critical("Camera integration not set up")
            return None

        if (camera := component.get_entity(entity_id)) is None:
            _LOGGER.error("Camera entity %s not found", entity_id)
            return None

        image_bytes = await camera.async_camera_image()
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            _LOGGER.debug("%s retrieved image from camera %s", self.name, entity_id)
            return image
        else:
            _LOGGER.warning("Camera %s image not ready", entity_id)
            return None

    async def async_fetch_image_from_url(self, url: str) -> Image.Image | None:
        """
        Asynchronously fetch an image from the provided HTTP URL.
        """
        session = async_get_clientsession(self.hass)
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.read()
                    _LOGGER.debug("%s fetched image from URL '%s'", self.name, url)

                    image = Image.open(io.BytesIO(content))
                    image.load()
                    return image
                else:
                    _LOGGER.error(
                        "Failed to fetch %s image from URL '%s': HTTP %d",
                        self.name,
                        url,
                        response.status,
                    )
                    return None

        except aiohttp.ClientError as e:
            _LOGGER.error(
                "HTTP error while fetching %s image from URL '%s': %s",
                self.name,
                url,
                e,
            )
            return None

        except asyncio.TimeoutError:
            _LOGGER.error(
                "Timeout while fetching %s image from URL '%s'", self.name, url
            )
            return None

    def crop_image(
        self, image: Image.Image, roi_config: Optional[Dict[str, int]]
    ) -> Image.Image:
        """
        Crop the image based on the region of interest (ROI). Returns the entire image if ROI is not specified
        """
        if not roi_config:
            return image

        img_width, img_height = image.size
        x, y = roi_config.get("x", 0), roi_config.get("y", 0)
        w, h = roi_config.get("width", img_width), roi_config.get("height", img_height)

        # Adjust ROI coordinates and dimensions to fit within image dimensions
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = min(w, img_width - x)
        h = min(h, img_height - y)

        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid ROI dimensions after adjustment: {roi_config}")

        _LOGGER.debug(
            "%s image cropped to ROI: x=%d, y=%d, width=%d, height=%d",
            self.name,
            x,
            y,
            w,
            h,
        )
        return image.crop((x, y, x + w, y + h))

    def resize_image(self, image: Image.Image, max_pixels: int) -> Image.Image:
        """
        Resize an image to have at most max_pixels using high quality Lanczos filter
        """
        # Calculate the total number of pixels in the image
        width, height = image.size
        total_pixels = width * height

        if total_pixels <= max_pixels:
            return image

        # Calculate the scaling factor to reduce the image to max_pixels
        scale_factor = math.sqrt(max_pixels / total_pixels)
        new_width = max(1, int(width * scale_factor))
        new_height = max(1, int(height * scale_factor))

        _LOGGER.debug(
            "%s image scaled from (%d x %d) to (%d x %d) to reduce total pixels from %d to <= %d",
            self.name,
            width,
            height,
            new_width,
            new_height,
            total_pixels,
            max_pixels,
        )

        # Resize the image using resampling filter
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def calculate_image_luminance(self, image: Image.Image) -> tuple[float, float]:
        """
        Calculate image average pixel luminance and perceived luminance.
        Supports:
            - 8-bit sRGB images (RGB or RGBA)
            - 16-bit linear grayscale images (I;16)
            - Linear images (F, I)
            - Converts common image modes (P, CMYK) to RGB before processing
            - If there's an alpha channel, it is stripped off before luminance calculation.
        Returns:
            A tuple containing (avg_luminance, perceived_luminance).
        """

        MODE_GRAYSCALE = "L"
        MODE_PALETTE = "P"
        MODE_CMYK = "CMYK"
        MODE_RGB = "RGB"
        MODE_RGBA = "RGBA"
        MODE_16_BIT_GRAYSCALE = "I;16"
        MODE_FLOAT = "F"
        MODE_LINEAR = "I"

        # Convert palette and CMYK images to RGB
        if image.mode in [MODE_PALETTE, MODE_CMYK]:
            image = image.convert(MODE_RGB)

        img_pixels = np.array(image)

        # Remove alpha channel if present
        if img_pixels.ndim == 3 and img_pixels.shape[2] > 3:
            img_pixels = img_pixels[:, :, :3]

        # Normalize pixel values depending on mode/bit-depth
        if image.mode == MODE_16_BIT_GRAYSCALE:
            # 16-bit linear grayscale
            linearized_pixels = img_pixels.astype(np.float32) / 65535.0

        elif image.mode in [MODE_LINEAR, MODE_FLOAT]:
            # 32-bit images
            linearized_pixels = img_pixels.astype(np.float32)

        elif image.mode in [MODE_RGB, MODE_RGBA, MODE_GRAYSCALE]:
            # 8-bit gamma encoded (sRGB/Grayscale)
            linearized_pixels = np.take(INVERSE_GAMMA_LUT, img_pixels.astype(np.uint8))

        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")

        if linearized_pixels.ndim == 2:
            # Grayscale
            y = linearized_pixels

        else:
            # For RGB, compute luminance (Y) using Rec. 709 coefficients
            y = (
                0.2126 * linearized_pixels[:, :, 0]
                + 0.7152 * linearized_pixels[:, :, 1]
                + 0.0722 * linearized_pixels[:, :, 2]
            )

        avg_luminance = np.mean(y)

        # Apply Weber–Fechner Law to approximate human perception
        perceived_luminance = np.log1p(avg_luminance)

        _LOGGER.debug(
            "%s avg pixel luminance %f perceived luminance %f from %s image",
            self.name,
            avg_luminance,
            perceived_luminance,
            image.mode,
        )
        return float(avg_luminance), float(perceived_luminance)
