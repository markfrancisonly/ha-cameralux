# custom_components/cameralux/sensor.py
"""CameraLux sensor entity for Home Assistant."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from homeassistant.components.camera import async_get_image
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry

# homeassistant.const exports LIGHT_LUX; there is no UnitOfIlluminance enum.
from homeassistant.const import (
    EVENT_HOMEASSISTANT_STARTED,
    LIGHT_LUX,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import CoreState, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    ALGORITHM_PROFILES,
    CONF_ALGORITHM,
    CONF_BRIGHTNESS_ROI,
    CONF_ENTITY_ID,
    CONF_FULL_SCALE_LUX,
    CONF_HEIGHT,
    CONF_IMAGE_URL,
    CONF_IR_ALGORITHM,
    CONF_IR_CHROMA_THRESHOLD,
    CONF_IR_ENABLED,
    CONF_IR_FIXED_LUX,
    CONF_ROI_ENABLED,
    CONF_SOURCE,
    CONF_UNAVAILABLE_ABOVE,
    CONF_UNAVAILABLE_BELOW,
    CONF_UPDATE_INTERVAL,
    CONF_WIDTH,
    CONF_X,
    CONF_Y,
    DEFAULT_ALGORITHM,
    DEFAULT_FULL_SCALE_LUX,
    DEFAULT_IR_ALGORITHM,
    DEFAULT_IR_CHROMA_THRESHOLD,
    DEFAULT_IR_FIXED_LUX,
    DEFAULT_UPDATE_INTERVAL,
    IR_ALGORITHM_FIXED,
    IR_ALGORITHM_UNAVAILABLE,
    MAX_PIXELS,
    SOURCE_CAMERA,
    SOURCE_URL,
    SUGGESTED_DISPLAY_PRECISION,
)
from .helpers import (
    coerce_float,
    float_if_value,
    get_bounded_int,
    process_image,
)

_LOGGER = logging.getLogger(__name__)


def _redact_url(url: str | None) -> str | None:
    """Strip any user:password@ credentials before a URL reaches the log.

    Users paste logs into bug reports; a snapshot URL may embed basic-auth
    userinfo. Only the authority is inspected, so an '@' in the path or query
    string is left untouched; the rest of the URL is kept for diagnosis.
    """
    if not url or "@" not in url:
        return url
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    # userinfo lives in the authority, which ends at the first / ? or #
    authority_len = len(rest)
    for delim in ("/", "?", "#"):
        i = rest.find(delim)
        if i != -1:
            authority_len = min(authority_len, i)
    authority, tail = rest[:authority_len], rest[authority_len:]
    if "@" not in authority:
        return url
    _, _, host = authority.rpartition("@")
    return f"{scheme}://***@{host}{tail}"


def _bounds_text(below: float | None, above: float | None) -> str:
    """Render the active unavailable bounds for a log line."""
    lo = "none" if below is None else f"{below:g}"
    hi = "none" if above is None else f"{above:g}"
    return f"below={lo} above={hi}"


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
):
    """Create entities from a UI config entry and its options."""
    data = {**entry.data, **entry.options}

    source = data.get(CONF_SOURCE)
    if source not in (SOURCE_CAMERA, SOURCE_URL):
        source = SOURCE_CAMERA if data.get(CONF_ENTITY_ID) else SOURCE_URL

    cfg = dict(data)
    cfg[CONF_ENTITY_ID] = data.get(CONF_ENTITY_ID) if source == SOURCE_CAMERA else None
    cfg[CONF_IMAGE_URL] = data.get(CONF_IMAGE_URL) if source == SOURCE_URL else None

    name = entry.data.get("name") or "Camera Lux"

    # Compatibility seam: current entries bypass the compatibility import. To
    # retire 1.x support, delete legacy_v1/ and this conditional factory call.
    entity = None
    if cfg.get(CONF_ALGORITHM) not in ALGORITHM_PROFILES:
        from .legacy_v1 import create_sensor_if_legacy

        entity = create_sensor_if_legacy(hass, name, cfg, entry.entry_id)
    if entity is None:
        entity = CameraLuxSensor(hass, name, cfg, stable_unique_id=entry.entry_id)

    # entry.entry_id remains stable across Options edits.
    async_add_entities([entity])


class CameraLuxSensor(SensorEntity):
    """Camera-based illuminance sensor sampling a camera entity or an image URL."""

    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.ILLUMINANCE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = LIGHT_LUX
    _attr_suggested_display_precision = SUGGESTED_DISPLAY_PRECISION

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        config: dict,
        stable_unique_id: str,
    ):
        """Initialize the sensor from a validated config dict."""
        super().__init__()
        self.hass = hass
        self._attr_name = name
        self._attr_unique_id = stable_unique_id

        self.lux: float | None = None
        self.avg_luminance: float | None = None
        self.perceived_luminance: float | None = None
        self.chroma: float | None = None
        self.ir_active: bool = False  # IR profile enabled AND chroma below threshold
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

        self.full_scale_lux = max(
            0.0,
            coerce_float(
                config.get(CONF_FULL_SCALE_LUX, DEFAULT_FULL_SCALE_LUX),
                DEFAULT_FULL_SCALE_LUX,
            ),
        )
        # Named algorithm -> (color trim fraction, exponent).
        self.algorithm, self.color_trim_fraction, self.exponent = self._resolve_curve(
            config.get(CONF_ALGORITHM)
        )
        self.update_interval = get_bounded_int(
            config.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
            DEFAULT_UPDATE_INTERVAL,
            5,
            3600,
        )
        self.unavailable_below = float_if_value(config.get(CONF_UNAVAILABLE_BELOW))
        self.unavailable_above = float_if_value(config.get(CONF_UNAVAILABLE_ABOVE))

        # IR: detection runs only when enabled (cameras with no IR mode skip
        # the per-frame work and can never false-positive into an IR policy).
        # The IR algorithm is the policy applied after an IR frame is
        # detected — measure with a named curve, report a fixed value, or go
        # unavailable.
        self.ir_enabled = bool(config.get(CONF_IR_ENABLED, False))
        (
            self.ir_algorithm,
            self.ir_trim_fraction,
            self.ir_exponent,
        ) = self._resolve_ir(config.get(CONF_IR_ALGORITHM))
        self.ir_fixed_lux = max(
            0.0,
            coerce_float(
                config.get(CONF_IR_FIXED_LUX, DEFAULT_IR_FIXED_LUX),
                DEFAULT_IR_FIXED_LUX,
            ),
        )
        self.ir_chroma_threshold = coerce_float(
            config.get(CONF_IR_CHROMA_THRESHOLD, DEFAULT_IR_CHROMA_THRESHOLD),
            DEFAULT_IR_CHROMA_THRESHOLD,
        )

        if not self.camera_entity_id and not self.image_url:
            _LOGGER.error(
                "CameraLux '%s' has neither camera entity nor image URL configured; "
                "sensor will stay unavailable.",
                name,
            )

        if not self.ir_enabled:
            ir_desc = "detect off"
        elif self.ir_algorithm == IR_ALGORITHM_FIXED:
            ir_desc = f"fixed {self.ir_fixed_lux:g} lx"
        elif self.ir_algorithm == IR_ALGORITHM_UNAVAILABLE:
            ir_desc = "unavailable"
        else:
            ir_desc = self.ir_algorithm
        _LOGGER.info(
            "CameraLux '%s': %s, full scale %.3f lx %s (source %s, ROI %s, night mode %s)",
            name,
            self.algorithm,
            self.full_scale_lux,
            self._curve_description(),
            self.camera_entity_id or _redact_url(self.image_url),
            "enabled" if self.roi_enabled else "disabled",
            ir_desc,
        )

    def _resolve_curve(self, algorithm: str | None) -> tuple[str, float, float]:
        """Resolve the configured algorithm to (algorithm, trim, exponent).

        Invalid stored values fall back to the default. Compatibility
        subclasses may pin their own curve.
        """
        if algorithm not in ALGORITHM_PROFILES:
            algorithm = DEFAULT_ALGORITHM
        trim, exponent = ALGORITHM_PROFILES[algorithm]
        return algorithm, trim, exponent

    def _perceive(self, avg_luminance: float) -> float:
        """Perception transform: linear-mean luminance -> perceived brightness.

        Everything downstream — lux, max_lux, bounds — derives from this
        single response-curve seam.
        """
        return max(0.0, avg_luminance) ** self.exponent

    def _curve_description(self) -> str:
        """Human-readable summary of the response curve for the startup log."""
        return f"exponent {self.exponent:.2f}"

    def _resolve_ir(self, ir_algorithm: str | None) -> tuple[str, float, float]:
        """Resolve the IR policy to (ir_algorithm, trim, exponent).

        A named algorithm measures IR frames with that curve; "fixed" and
        "unavailable" are policies (trim/exponent inert). Anything
        unrecognized falls back to the default. Compatibility subclasses may
        extend the stored vocabulary.
        """
        if ir_algorithm in ALGORITHM_PROFILES:
            trim, exponent = ALGORITHM_PROFILES[ir_algorithm]
            return ir_algorithm, trim, exponent
        if ir_algorithm in (IR_ALGORITHM_FIXED, IR_ALGORITHM_UNAVAILABLE):
            return ir_algorithm, 0.0, 1.0
        return DEFAULT_IR_ALGORITHM, 0.0, 1.0

    def _ir_measures(self) -> bool:
        """True when the IR policy is a measuring algorithm."""
        return self.ir_algorithm not in (
            IR_ALGORITHM_FIXED,
            IR_ALGORITHM_UNAVAILABLE,
        )

    def _ir_perceive(self, avg_luminance: float) -> float:
        """Perception transform for IR frames under a measuring IR algorithm
        (the IR counterpart of _perceive)."""
        return max(0.0, avg_luminance) ** self.ir_exponent

    @property
    def max_lux(self) -> float:
        """Lux a pure-white frame would report.

        avg_luminance is normalized to [0, 1] (the inverse-gamma LUT maps 0-255
        onto that range and the Rec.709 weights sum to exactly 1.0), so the
        ceiling is by definition the perception transform of full scale. For
        the power-law curves 1.0 ** exponent == 1.0, making it exactly the
        full-scale value.
        """
        return self._perceive(1.0) * self.full_scale_lux

    @property
    def available(self) -> bool:
        """Available when a lux value has been computed."""
        return self.lux is not None

    @property
    def native_value(self) -> float | None:
        """Return the current illuminance in lux."""
        return self.lux

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Calculated values that explain the lux output — configuration
        lives in the options UI, not here. Insertion order is the display order.
        """
        attrs: dict[str, Any] = {}
        attrs["chroma"] = round(self.chroma, 2) if self.chroma is not None else None
        if self.ir_enabled:
            # KEY attribute: True = the frame read as IR/monochrome, so the
            # IR policy is in force. Absent when detection is disabled —
            # claiming IR/not-IR without looking would be a lie.
            attrs["ir_detected"] = self.ir_active
        attrs["avg_luminance"] = (
            round(self.avg_luminance, 4) if self.avg_luminance is not None else None
        )
        attrs["perceived_luminance"] = (
            round(self.perceived_luminance, 4)
            if self.perceived_luminance is not None
            else None
        )
        # the policy in force this cycle: the IR one for an IR frame
        attrs["algorithm"] = self.ir_algorithm if self.ir_active else self.algorithm
        attrs["perception_exponent"] = (
            self.ir_exponent if self.ir_active else self.exponent
        )
        # reading for a pure-white frame
        attrs["max_lux"] = self.max_lux
        # shown only when set
        if self.unavailable_above is not None:
            attrs["unavailable_above_lux"] = self.unavailable_above
        if self.unavailable_below is not None:
            attrs["unavailable_below_lux"] = self.unavailable_below
        return attrs

    async def async_added_to_hass(self):
        """Start periodic updates when the entity is added to Home Assistant."""

        async def start_polling(_=None):
            self.hass.async_create_task(self.async_update())
            self.async_on_remove(
                async_track_time_interval(
                    self.hass,
                    self.async_update,
                    timedelta(seconds=self.update_interval),
                )
            )

        if self.hass.state is not CoreState.running:
            unsub = None

            async def _on_started(_event=None):
                nonlocal unsub
                unsub = None  # one-shot listener has consumed itself
                await start_polling()

            unsub = self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STARTED, _on_started
            )
            self.async_on_remove(lambda: unsub() if unsub else None)
        else:
            await start_polling()

    async def async_update(self, _: datetime | None = None) -> None:
        """Fetch an image, measure it (in the executor), and update lux."""
        if self.hass.state is not CoreState.running:
            return
        if self.is_updating:
            _LOGGER.debug("'%s' update already in progress; skipping.", self._attr_name)
            return

        # Read before self.lux is touched, so the branches below can report the
        # availability transition rather than the steady state.
        was_available = self.lux is not None

        try:
            self.is_updating = True

            if self.camera_entity_id:
                image_bytes = await self.async_get_camera_snapshot(
                    self.camera_entity_id
                )
            elif self.image_url:
                image_bytes = await self.async_fetch_image_from_url(self.image_url)
            else:
                image_bytes = None

            if not image_bytes:
                if was_available:
                    _LOGGER.debug(
                        "'%s' became unavailable: no image from %s",
                        self._attr_name,
                        self.camera_entity_id
                        or _redact_url(self.image_url)
                        or "no source",
                    )
                self.lux = None
                return

            roi = self.roi if self.roi_enabled else None
            # Detection runs only when enabled (threshold None disables it);
            # the IR algorithm is the policy applied AFTER an IR frame is
            # detected. A measuring IR algorithm uses its own statistic
            # (trim 0 == plain mean, trim 0.05 == the trimmed mean); for
            # fixed/unavailable the statistic only feeds the avg_luminance
            # diagnostic (hotspot-resistant).
            if self._ir_measures():
                ir_statistic = "trim" if self.ir_trim_fraction > 0 else "mean"
            else:
                ir_statistic = "trim"
            (
                self.avg_luminance,
                self.chroma,
                self.ir_active,
            ) = await self.hass.async_add_executor_job(
                process_image,
                image_bytes,
                roi,
                MAX_PIXELS,
                self.ir_chroma_threshold if self.ir_enabled else None,
                self.color_trim_fraction,
                ir_statistic,
            )

            if self.ir_active and self.ir_algorithm == IR_ALGORITHM_UNAVAILABLE:
                # Honesty valve: in night mode the camera measures its own
                # illuminator; go offline rather than report a number nobody
                # measured.
                if was_available:
                    _LOGGER.debug(
                        "'%s' became unavailable: IR frame, ir_algorithm=unavailable",
                        self._attr_name,
                    )
                self.perceived_luminance = None
                self.lux = None
                return

            if self.ir_active and self.ir_algorithm == IR_ALGORITHM_FIXED:
                # A declared policy value, not a measurement: the unavailable
                # bounds do not judge it (pick "unavailable" for that).
                if not was_available:
                    _LOGGER.debug(
                        "'%s' became available: IR frame, fixed %.3f lx",
                        self._attr_name,
                        self.ir_fixed_lux,
                    )
                self.perceived_luminance = None
                self.lux = self.ir_fixed_lux
                return

            # Measuring IR algorithm or a day frame: the appropriate curve,
            # the shared full scale, the normal bounds.
            if self.ir_active:
                self.perceived_luminance = self._ir_perceive(self.avg_luminance)
            else:
                self.perceived_luminance = self._perceive(self.avg_luminance)
            below = self.unavailable_below
            above = self.unavailable_above
            lux = self.perceived_luminance * self.full_scale_lux

            below_trip = below is not None and lux < below
            above_trip = above is not None and lux > above

            if below_trip or above_trip:
                crossed = (
                    f"unavailable_below {below:g}"
                    if below_trip
                    else f"unavailable_above {above:g}"
                )
                _LOGGER.debug(
                    "'%s' %s: lux %.3f crossed %s (chroma=%.2f)",
                    self._attr_name,
                    "became unavailable" if was_available else "still unavailable",
                    lux,
                    crossed,
                    self.chroma,
                )
                self.lux = None  # diagnostics are kept for the next good cycle
            else:
                if not was_available:
                    _LOGGER.debug(
                        "'%s' became available: lux %.3f within bounds (%s) "
                        "(chroma=%.2f)",
                        self._attr_name,
                        lux,
                        _bounds_text(below, above),
                        self.chroma,
                    )
                self.lux = lux

        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if was_available:
                _LOGGER.debug("'%s' became unavailable: update failed", self._attr_name)
            self.lux = None
            _LOGGER.error("Error updating CameraLux '%s': %s", self._attr_name, e)
        finally:
            self.is_updating = False
            self.async_write_ha_state()

    async def async_get_camera_snapshot(self, entity_id: str) -> bytes | None:
        """Fetch a snapshot from a camera entity via HA core (timeout included).

        The camera's own state gates the fetch: async_get_image can serve a
        CACHED frame for an unavailable camera, which would keep this sensor
        confidently measuring months-old pixels. A stale reading is worse than
        an unavailable one.
        """
        camera_state = self.hass.states.get(entity_id)
        if camera_state is None or camera_state.state in (
            STATE_UNAVAILABLE,
            STATE_UNKNOWN,
        ):
            _LOGGER.debug(
                "'%s': camera %s is %s; not measuring a cached frame",
                self._attr_name,
                entity_id,
                camera_state.state if camera_state else "missing",
            )
            return None
        try:
            snapshot = await async_get_image(self.hass, entity_id)
        except HomeAssistantError as e:
            _LOGGER.error("Error retrieving image from camera %s: %s", entity_id, e)
            return None
        return snapshot.content or None

    async def async_fetch_image_from_url(self, url: str) -> bytes | None:
        """Fetch an image from a URL using HA's shared aiohttp session."""
        session = async_get_clientsession(self.hass)
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status != 200:
                    _LOGGER.error(
                        "Failed to fetch %s image from URL '%s': HTTP %d",
                        self._attr_name,
                        _redact_url(url),
                        response.status,
                    )
                    return None
                return await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            _LOGGER.error(
                "Error fetching %s image from URL '%s': %s",
                self._attr_name,
                _redact_url(url),
                e,
            )
            return None
