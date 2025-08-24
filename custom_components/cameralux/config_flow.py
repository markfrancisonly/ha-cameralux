# custom_components/cameralux/config_flow.py
from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_BRIGHTNESS_ROI,
    CONF_CALIBRATION_FACTOR,
    CONF_ENTITY_ID,
    CONF_HEIGHT,
    CONF_IMAGE_URL,
    CONF_ROI_ENABLED,
    CONF_SOURCE,
    CONF_UNAVAILABLE_ABOVE,
    CONF_UNAVAILABLE_BELOW,
    CONF_UPDATE_INTERVAL,
    CONF_WIDTH,
    CONF_X,
    CONF_Y,
    DOMAIN,
    SOURCE_CAMERA,
    SOURCE_URL,
)

_LOGGER = logging.getLogger(__name__)
UI_ROI_ENABLED = "roi_enabled"  # UI-only toggle name

from .helpers import build_unique_id, int_if_value


def nullable(sel):
    return vol.Any(None, sel)


class CameraLuxConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for CameraLux."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Initial setup: choose Source, set camera/url, optional ROI, calibration, interval."""
        errors: dict[str, str] = {}

        # Defaults only for required/always-present fields; avoid default=None for optional.
        defaults = {
            "name": "",
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_IMAGE_URL: "",
            CONF_CALIBRATION_FACTOR: 2000,
            CONF_UPDATE_INTERVAL: 30,
            UI_ROI_ENABLED: False,
        }

        if user_input is not None:
            _LOGGER.debug("Received user input: %s", user_input)
            try:
                name = (user_input.get("name") or "").strip()
                source = user_input.get(CONF_SOURCE, SOURCE_CAMERA)
                entity_id = user_input.get(CONF_ENTITY_ID)
                image_url = (user_input.get(CONF_IMAGE_URL) or "").strip()
                calibration = int(user_input.get(CONF_CALIBRATION_FACTOR, 2000))
                interval = int(user_input.get(CONF_UPDATE_INTERVAL, 30))
                unavail_below = int_if_value(user_input.get(CONF_UNAVAILABLE_BELOW))
                unavail_above = int_if_value(user_input.get(CONF_UNAVAILABLE_ABOVE))
                roi_enabled = bool(user_input.get(UI_ROI_ENABLED, False))
                roi_x = int_if_value(user_input.get(CONF_X))
                roi_y = int_if_value(user_input.get(CONF_Y))
                roi_width = int_if_value(user_input.get(CONF_WIDTH))
                roi_height = int_if_value(user_input.get(CONF_HEIGHT))

                # Validate requireds
                if not name:
                    errors["name"] = "required"
                if source == SOURCE_CAMERA and not entity_id:
                    errors[CONF_ENTITY_ID] = "required"
                if source == SOURCE_URL and not image_url:
                    errors[CONF_IMAGE_URL] = "required"

                if not errors:
                    roi_dict = {CONF_ROI_ENABLED: roi_enabled}
                    if roi_x is not None:
                        roi_dict[CONF_X] = roi_x
                    if roi_y is not None:
                        roi_dict[CONF_Y] = roi_y
                    if roi_width is not None:
                        roi_dict[CONF_WIDTH] = roi_width
                    if roi_height is not None:
                        roi_dict[CONF_HEIGHT] = roi_height

                    data = {
                        "name": name,
                        CONF_ENTITY_ID: entity_id if source == SOURCE_CAMERA else None,
                        CONF_IMAGE_URL: image_url if source == SOURCE_URL else None,
                        CONF_BRIGHTNESS_ROI: roi_dict,
                        CONF_CALIBRATION_FACTOR: float(calibration),
                        CONF_UPDATE_INTERVAL: int(interval),
                        CONF_UNAVAILABLE_BELOW: unavail_below,
                        CONF_UNAVAILABLE_ABOVE: unavail_above,
                        # saving source is optional for your logic, but harmless and clearer:
                        CONF_SOURCE: source,
                    }

                    unique_id = build_unique_id(name, entity_id, image_url)
                    await self.async_set_unique_id(unique_id)
                    self._abort_if_unique_id_configured()
                    _LOGGER.debug("Creating config entry with data: %s", data)
                    return self.async_create_entry(title=name, data=data)

            except ValueError as e:
                _LOGGER.error("Invalid input: %s", e)
                errors["base"] = "invalid_input"

        # ---- Build form schema (nullable selectors for optionals; no default=None) ----
        fields = {
            vol.Required("name", default=defaults["name"]): str,
            vol.Required(CONF_SOURCE, default=defaults[CONF_SOURCE]): selector.selector(
                {
                    "select": {
                        "options": [
                            {"label": "Camera entity", "value": SOURCE_CAMERA},
                            {"label": "Image URL", "value": SOURCE_URL},
                        ],
                        "mode": "dropdown",
                    }
                }
            ),
            vol.Optional(CONF_ENTITY_ID): nullable(
                selector.selector({"entity": {"domain": "camera"}})
            ),
            vol.Optional(
                CONF_IMAGE_URL, default=defaults[CONF_IMAGE_URL]
            ): selector.selector({"text": {}}),
            vol.Required(
                CONF_CALIBRATION_FACTOR, default=defaults[CONF_CALIBRATION_FACTOR]
            ): selector.selector({"number": {"min": 1, "step": 1, "mode": "box"}}),
            vol.Required(
                CONF_UPDATE_INTERVAL, default=defaults[CONF_UPDATE_INTERVAL]
            ): selector.selector({"number": {"min": 5, "step": 1, "mode": "box"}}),
            vol.Optional(CONF_UNAVAILABLE_BELOW): nullable(
                selector.selector({"number": {"min": 0, "mode": "box"}})
            ),
            vol.Optional(CONF_UNAVAILABLE_ABOVE): nullable(
                selector.selector({"number": {"min": 0, "mode": "box"}})
            ),
            vol.Required(
                UI_ROI_ENABLED, default=defaults[UI_ROI_ENABLED]
            ): selector.selector({"boolean": {}}),
            vol.Optional(CONF_X): nullable(
                selector.selector({"number": {"min": 0, "mode": "box"}})
            ),
            vol.Optional(CONF_Y): nullable(
                selector.selector({"number": {"min": 0, "mode": "box"}})
            ),
            vol.Optional(CONF_WIDTH): nullable(
                selector.selector({"number": {"min": 0, "mode": "box"}})
            ),
            vol.Optional(CONF_HEIGHT): nullable(
                selector.selector({"number": {"min": 0, "mode": "box"}})
            ),
        }

        return self.async_show_form(
            step_id="user", data_schema=vol.Schema(fields), errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(entry):
        return CameraLuxOptionsFlow(entry)


class CameraLuxOptionsFlow(config_entries.OptionsFlow):
    """Options: allow updating Source, Entity/URL, ROI, calibration, interval."""

    def __init__(self, entry):
        self.entry = entry

    async def async_step_init(self, user_input=None):
        return await self.async_step_edit()

    async def async_step_edit(self, user_input=None):
        errors: dict[str, str] = {}

        data = {**self.entry.data, **self.entry.options}

        source_def = data.get(CONF_SOURCE)
        if not source_def:
            source_def = SOURCE_CAMERA if data.get(CONF_ENTITY_ID) else SOURCE_URL

        entity_def = data.get(CONF_ENTITY_ID)
        url_def = data.get(CONF_IMAGE_URL) or ""
        cal_def = int(data.get(CONF_CALIBRATION_FACTOR, 2000))
        interval_def = int(data.get(CONF_UPDATE_INTERVAL, 30))
        unavail_below_def = data.get(CONF_UNAVAILABLE_BELOW)
        unavail_above_def = data.get(CONF_UNAVAILABLE_ABOVE)

        roi = data.get(CONF_BRIGHTNESS_ROI) or {}
        roi_enabled_def = bool(roi.get(CONF_ROI_ENABLED, False))
        x_def = roi.get(CONF_X)
        y_def = roi.get(CONF_Y)
        w_def = roi.get(CONF_WIDTH)
        h_def = roi.get(CONF_HEIGHT)

        if user_input is not None:
            source = user_input.get(CONF_SOURCE, source_def)
            entity_in = user_input.get(CONF_ENTITY_ID)
            url_in = (user_input.get(CONF_IMAGE_URL) or "").strip()
            cal_in = int(user_input.get(CONF_CALIBRATION_FACTOR, cal_def))
            interval_in = int(user_input.get(CONF_UPDATE_INTERVAL, interval_def))
            unavail_below_in = int_if_value(user_input.get(CONF_UNAVAILABLE_BELOW))
            unavail_above_in = int_if_value(user_input.get(CONF_UNAVAILABLE_ABOVE))
            roi_enabled_in = bool(user_input.get(UI_ROI_ENABLED, roi_enabled_def))

            roi_new: dict = {CONF_ROI_ENABLED: roi_enabled_in}
            xv = int_if_value(user_input.get(CONF_X))
            yv = int_if_value(user_input.get(CONF_Y))
            wv = int_if_value(user_input.get(CONF_WIDTH))
            hv = int_if_value(user_input.get(CONF_HEIGHT))
            if xv is not None:
                roi_new[CONF_X] = xv
            if yv is not None:
                roi_new[CONF_Y] = yv
            if wv is not None:
                roi_new[CONF_WIDTH] = wv
            if hv is not None:
                roi_new[CONF_HEIGHT] = hv

            if source == SOURCE_CAMERA and not entity_in:
                errors[CONF_ENTITY_ID] = "required"
            if source == SOURCE_URL and not url_in:
                errors[CONF_IMAGE_URL] = "required"

            if not errors:
                new_opts = {
                    CONF_SOURCE: source,
                    CONF_ENTITY_ID: entity_in if source == SOURCE_CAMERA else None,
                    CONF_IMAGE_URL: url_in if source == SOURCE_URL else None,
                    CONF_CALIBRATION_FACTOR: float(cal_in),
                    CONF_UPDATE_INTERVAL: int(interval_in),
                    CONF_BRIGHTNESS_ROI: roi_new,
                    CONF_UNAVAILABLE_BELOW: unavail_below_in,
                    CONF_UNAVAILABLE_ABOVE: unavail_above_in,
                }
                return self.async_create_entry(title="", data=new_opts)

            # keep user input in form after errors
            source_def = source
            entity_def = entity_in
            url_def = url_in
            cal_def = cal_in
            interval_def = interval_in
            unavail_below_def = unavail_below_in
            unavail_above_def = unavail_above_in
            roi_enabled_def = roi_enabled_in
            x_def = roi_new.get(CONF_X, x_def)
            y_def = roi_new.get(CONF_Y, y_def)
            w_def = roi_new.get(CONF_WIDTH, w_def)
            h_def = roi_new.get(CONF_HEIGHT, h_def)

        # ---- Build options form (nullable selectors for optionals) ----
        fields: dict = {}
        fields[vol.Required(CONF_SOURCE, default=source_def)] = selector.selector(
            {
                "select": {
                    "options": [
                        {"label": "Camera entity", "value": SOURCE_CAMERA},
                        {"label": "Image URL", "value": SOURCE_URL},
                    ],
                    "mode": "dropdown",
                }
            }
        )

        if isinstance(entity_def, str) and entity_def:
            entity_key = vol.Optional(CONF_ENTITY_ID, default=entity_def)
        else:
            entity_key = vol.Optional(CONF_ENTITY_ID)
        fields[entity_key] = nullable(
            selector.selector({"entity": {"domain": "camera"}})
        )

        fields[vol.Optional(CONF_IMAGE_URL, default=url_def)] = selector.selector(
            {"text": {}}
        )

        fields[vol.Required(CONF_CALIBRATION_FACTOR, default=cal_def)] = (
            selector.selector({"number": {"min": 1, "step": 1, "mode": "box"}})
        )
        fields[vol.Required(CONF_UPDATE_INTERVAL, default=interval_def)] = (
            selector.selector({"number": {"min": 5, "step": 1, "mode": "box"}})
        )

        if unavail_below_def is not None:
            fields[vol.Optional(CONF_UNAVAILABLE_BELOW, default=unavail_below_def)] = (
                nullable(selector.selector({"number": {"min": 0, "mode": "box"}}))
            )
        else:
            fields[vol.Optional(CONF_UNAVAILABLE_BELOW)] = nullable(
                selector.selector({"number": {"min": 0, "mode": "box"}})
            )

        if unavail_above_def is not None:
            fields[vol.Optional(CONF_UNAVAILABLE_ABOVE, default=unavail_above_def)] = (
                nullable(selector.selector({"number": {"min": 0, "mode": "box"}}))
            )
        else:
            fields[vol.Optional(CONF_UNAVAILABLE_ABOVE)] = nullable(
                selector.selector({"number": {"min": 0, "mode": "box"}})
            )

        fields[vol.Required(UI_ROI_ENABLED, default=roi_enabled_def)] = (
            selector.selector({"boolean": {}})
        )

        def add_roi(key, val):
            if val is not None:
                fields[vol.Optional(key, default=val)] = nullable(
                    selector.selector({"number": {"min": 0, "mode": "box"}})
                )
            else:
                fields[vol.Optional(key)] = nullable(
                    selector.selector({"number": {"min": 0, "mode": "box"}})
                )

        add_roi(CONF_X, x_def)
        add_roi(CONF_Y, y_def)
        add_roi(CONF_WIDTH, w_def)
        add_roi(CONF_HEIGHT, h_def)

        return self.async_show_form(
            step_id="edit", data_schema=vol.Schema(fields), errors=errors
        )
