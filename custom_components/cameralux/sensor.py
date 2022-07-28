"""Platform for sensor integration."""
from __future__ import annotations

import glob
 
import numpy as np
from numpy.linalg import norm

from datetime import timedelta

from homeassistant.components.sensor import SensorEntity, PLATFORM_SCHEMA
from homeassistant.const import (
    CONF_SENSORS,
    DEVICE_CLASS_ILLUMINANCE, 
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

try:
    # Verify that the OpenCV python package is pre-installed
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=30)


PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_SENSORS): {cv.string: cv.entity_domain(CAMERA_DOMAIN)},
    }
)


"""
Emulated-lux sensor based on camera snapshot image 

- platform: cameralux
  sensors:
    Doorbell lux: camera.doorbell
    Family lux: camera.family
    Kitchen lux: camera.kitchen
"""

async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):

    _LOGGER.debug(
        "Camera Lux platform setup started"
    )

    if not CV2_IMPORTED:
        _LOGGER.error(
            "No OpenCV library found! Install or compile for your system "
            "following instructions here: http://opencv.org/releases.html"
        )
        return
        
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
        return DEVICE_CLASS_ILLUMINANCE

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    async def async_update(self) -> None:

        _LOGGER.debug(
            "Updating camera lux sensor %s for %s", self.name, self._camera_entity_id 
        )

        try:
            camera = _get_camera_from_entity_id(self._hass, self._camera_entity_id)
            if camera:
                image_bytes = await camera.async_camera_image()    
                if image_bytes:

                    image = np.asarray(bytearray(image_bytes), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                    lux = _get_image_brightness(image)
                    self._state = round(lux, 2)
                    self._available = True
        except:
            _LOGGER.debug(
                "Error updating camera lux sensor %s", self.name
            )
            self._state = None
            self._available = False
 

def _is_image_brighter_than(image, dim=10, thresh=0.5):

    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))

    # Convert color space to LAB format and extract L channel
    L = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]

    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)

    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh

def _get_image_brightness(image):
    if len(image.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(image, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(image)

 
def _get_camera_from_entity_id(hass: HomeAssistant, entity_id: str) -> Camera:
    """Get camera component from entity_id."""
    component = hass.data.get(CAMERA_DOMAIN)

    if component is None:
        raise HomeAssistantError("Camera integration not set up")

    camera = component.get_entity(entity_id)

    if camera is None:
        raise HomeAssistantError("Camera not found")

    if not camera.is_on:
        raise HomeAssistantError("Camera is off")

    return cast(Camera, camera)


 