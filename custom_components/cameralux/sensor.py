"""Platform for sensor integration."""
from __future__ import annotations

import glob
 
from datetime import timedelta
import io
import math

from PIL import Image
from PIL import ImageStat

from homeassistant.components.sensor import SensorEntity, SensorDeviceClass, PLATFORM_SCHEMA
from homeassistant.const import (
    CONF_SENSORS, 
    LIGHT_LUX,
)
from homeassistant.components.camera import Camera, DOMAIN as CAMERA_DOMAIN 
from homeassistant.exceptions import HomeAssistantError

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from typing import cast

import homeassistant.helpers.config_validation as cv
import voluptuous as vol
import logging

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=30)


PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_SENSORS): {cv.string: cv.entity_domain(CAMERA_DOMAIN)},
    }
)


"""
Create emulated-lux sensor from from camera entity

sensor:
  - platform: jpeglux
    entities:
      - camera_one
      - camera_two

"""
async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):

    _LOGGER.debug(
        "Camera Lux platform setup started"
    )

    sensors = [CameraLux(hass, name, entity_id) for name, entity_id in config[CONF_SENSORS].items()]
    async_add_entities(sensors, True)


class CameraLux(SensorEntity):
    """Representation of a Sensor."""

    def __init__(self, hass, name, entity_id):
        """Initialize the sensor."""
        self._hass = hass
        self._camera_entity_id = entity_id
        self._name = name
        self._state = None
        self._available = True

        _LOGGER.debug(
            "Initialized camera lux sensor %s for %s", name, entity_id
        )

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def state(self):
        """Return the state of the sensor."""
        return self._state

    @property
    def unique_id(self) -> str:
        """Return the unique ID of the sensor."""
        return "camera_lux_" + self._name

    @property
    def unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return LIGHT_LUX

    @property
    def device_class(self) -> str:
        """Return the illuminance."""
        return SensorDeviceClass.ILLUMINANCE

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    async def async_update(self) -> None:

        _LOGGER.debug(
            "Updating cameralux sensor %s for %s", self.name, self._camera_entity_id 
        )

        try:
            camera = CameraLux.get_camera_from_entity_id(self._hass, self._camera_entity_id)
            if camera:
                image_bytes = await camera.async_camera_image()    
                if image_bytes:
                    image = Image.open(io.BytesIO(image_bytes))

                    lux = CameraLux.get_image_lux(image)
                    self._state = round(lux, 2)
                    self._available = True
            else:
                self._state = None
                self._available = False      

        except Exception as e:
            _LOGGER.error(
                "Error updating cameralux sensor %s: %s", self.name, e
            )
            self._state = None
            self._available = False
 
    @staticmethod
    def get_image_lux(image):
        
        # get average pixel level for each band in the image
        stat = ImageStat.Stat(image)
        r,g,b = stat.mean

        # calc perceived brightness value based on the degree each color 
        # component (RGB) affect the human perception of brightness as suggested 
        # by the National Television System Committee (NTSC) for converting color 
        # feeds to black and white televisions set

        return math.sqrt(0.299*(r**2) + 0.587*(g**2) + 0.114*(b**2))

    @staticmethod 
    def get_camera_from_entity_id(hass: HomeAssistant, entity_id: str) -> Camera:
        """Get camera component from entity_id."""
        component = hass.data.get(CAMERA_DOMAIN)

        if component is None:
            raise HomeAssistantError("Camera integration not set up")

        camera = component.get_entity(entity_id)

        if camera is None:
            raise HomeAssistantError(entity_id + " not found")

        if not camera.is_on:
            raise HomeAssistantError(entity_id + " is off")

        return cast(Camera, camera)