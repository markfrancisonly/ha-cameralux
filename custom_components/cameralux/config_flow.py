# custom_components/cameralux/config_flow.py
from __future__ import annotations

import hashlib

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
    CONF_UNAVAILABLE_BELOW,
    CONF_UNAVAILABLE_ABOVE,
    CONF_UPDATE_INTERVAL,
    CONF_WIDTH,
    CONF_X,
    CONF_Y,
    DOMAIN,
    SOURCE_CAMERA,
    SOURCE_URL,
)

UI_ROI_ENABLED = "roi_enabled"  # UI-only toggle name


def build_entry_unique_id(
    name: str, camera_entity_id: str | None, image_url: str | None
) -> str:
    """Unique ID base (camera/url) + short hash of name to allow multiple sensors per camera."""
    suffix = hashlib.md5((name or "unnamed").encode()).hexdigest()[:8]
    if camera_entity_id:
        base = f"cameralux_{camera_entity_id.replace('.', '_')}"
    elif image_url:
        base = f"cameralux_url_{hashlib.md5(image_url.encode()).hexdigest()}"
    else:
        base = "cameralux_manual"
    return f"{base}_{suffix}"


def int_if_value(value) -> int | None:
    """Return int(value) if it is a real value (not None/empty string); otherwise None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_if_value(value) -> float | None:
    """Return float(value) if it is a real value; otherwise None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CameraLuxConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for CameraLux."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Initial setup: choose Source, set camera/url, optional ROI, calibration, interval."""
        errors: dict[str, str] = {}

        name_def = ""
        source_def = SOURCE_CAMERA
        entity_id_def: str | None = None
        image_url_def = ""
        cal_def = 2000
        interval_def = 30
        unavail_below_def = None
        unavail_above_def = None
        roi_enabled_ui = False

        x_def = y_def = w_def = h_def = None

        if user_input is not None:
            name_def = (user_input.get("name") or "").strip()
            source_def = user_input.get(CONF_SOURCE, SOURCE_CAMERA)
            entity_id_def = user_input.get(CONF_ENTITY_ID)
            image_url_def = (user_input.get(CONF_IMAGE_URL) or "").strip()
            cal_def = int(user_input.get(CONF_CALIBRATION_FACTOR, 2000))
            interval_def = int(user_input.get(CONF_UPDATE_INTERVAL, 30))
            unavail_below_def = float_if_value(
                user_input.get(CONF_UNAVAILABLE_BELOW)
            )
            unavail_above_def = float_if_value(
                user_input.get(CONF_UNAVAILABLE_ABOVE)
            )
            roi_enabled_ui = bool(user_input.get(UI_ROI_ENABLED, False))

            if CONF_X in user_input:
                x_def = int_if_value(user_input[CONF_X])
            if CONF_Y in user_input:
                y_def = int_if_value(user_input[CONF_Y])
            if CONF_WIDTH in user_input:
                w_def = int_if_value(user_input[CONF_WIDTH])
            if CONF_HEIGHT in user_input:
                h_def = int_if_value(user_input[CONF_HEIGHT])

            chosen_entity = entity_id_def if source_def == SOURCE_CAMERA else None
            chosen_url = image_url_def if source_def == SOURCE_URL else None
            if source_def == SOURCE_CAMERA and not chosen_entity:
                errors[CONF_ENTITY_ID] = "required"
            if source_def == SOURCE_URL and not chosen_url:
                errors[CONF_IMAGE_URL] = "required"

            if not errors:
                roi_dict: dict = {CONF_ROI_ENABLED: roi_enabled_ui}

                if CONF_X in user_input:
                    roi_dict[CONF_X] = int_if_value(user_input.get(CONF_X))
                if CONF_Y in user_input:
                    roi_dict[CONF_Y] = int_if_value(user_input.get(CONF_Y))
                if CONF_WIDTH in user_input:
                    roi_dict[CONF_WIDTH] = int_if_value(user_input.get(CONF_WIDTH))
                if CONF_HEIGHT in user_input:
                    roi_dict[CONF_HEIGHT] = int_if_value(user_input.get(CONF_HEIGHT))

                data = {
                    "name": name_def,
                    CONF_ENTITY_ID: chosen_entity,
                    CONF_IMAGE_URL: chosen_url or None,
                    CONF_BRIGHTNESS_ROI: roi_dict,
                    CONF_CALIBRATION_FACTOR: float(cal_def),
                    CONF_UPDATE_INTERVAL: int(interval_def),
                    CONF_UNAVAILABLE_BELOW: unavail_below_def,
                    CONF_UNAVAILABLE_ABOVE: unavail_above_def,
                }
                unique_id = build_entry_unique_id(name_def, chosen_entity, chosen_url)
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(title=name_def, data=data)

        if isinstance(entity_id_def, str) and entity_id_def:
            entity_field = vol.Optional(CONF_ENTITY_ID, default=entity_id_def)
        else:
            entity_field = vol.Optional(CONF_ENTITY_ID)

        fields: dict = {}
        fields[vol.Required("name", default=name_def)] = str
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
        fields[entity_field] = selector.selector({"entity": {"domain": "camera"}})
        fields[vol.Optional(CONF_IMAGE_URL, default=image_url_def)] = selector.selector(
            {"text": {}}
        )

        fields[vol.Optional(CONF_CALIBRATION_FACTOR, default=cal_def)] = (
            selector.selector({"number": {"min": 1, "step": 1, "mode": "box"}})
        )
        fields[vol.Optional(CONF_UPDATE_INTERVAL, default=interval_def)] = (
            selector.selector({"number": {"min": 5, "step": 1, "mode": "box"}})
        )
        fields[vol.Optional(CONF_UNAVAILABLE_BELOW)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )
        fields[vol.Optional(CONF_UNAVAILABLE_ABOVE)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )
        fields[vol.Required(UI_ROI_ENABLED, default=roi_enabled_ui)] = (
            selector.selector({"boolean": {}})
        )

        fields[vol.Optional(CONF_X)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        fields[vol.Optional(CONF_Y)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        fields[vol.Optional(CONF_WIDTH)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        fields[vol.Optional(CONF_HEIGHT)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        return self.async_show_form(
            step_id="user", data_schema=vol.Schema(fields), errors=errors
        )

    async def async_step_import(self, import_config: dict):
        """YAML → UI import. Prefer camera if both present; store ROI with 'enabled'. Do not inject zeros."""
        name = (import_config.get("name") or "").strip()
        entity_id = import_config.get(CONF_ENTITY_ID)
        image_url = (import_config.get(CONF_IMAGE_URL) or "").strip() or None

        if entity_id:
            chosen_entity, chosen_url = entity_id, None
        elif image_url:
            chosen_entity, chosen_url = None, image_url
        else:
            return self.async_abort(reason="need_source")

        roi_yaml = import_config.get(CONF_BRIGHTNESS_ROI) or {}
        roi: dict = {CONF_ROI_ENABLED: bool(roi_yaml.get(CONF_ROI_ENABLED, False))}

        if CONF_X in roi_yaml:
            roi[CONF_X] = int_if_value(roi_yaml.get(CONF_X))
        if CONF_Y in roi_yaml:
            roi[CONF_Y] = int_if_value(roi_yaml.get(CONF_Y))
        if CONF_WIDTH in roi_yaml:
            roi[CONF_WIDTH] = int_if_value(roi_yaml.get(CONF_WIDTH))
        if CONF_HEIGHT in roi_yaml:
            roi[CONF_HEIGHT] = int_if_value(roi_yaml.get(CONF_HEIGHT))

        data = {
            "name": name,
            CONF_ENTITY_ID: chosen_entity,
            CONF_IMAGE_URL: chosen_url,
            CONF_BRIGHTNESS_ROI: roi,
            CONF_CALIBRATION_FACTOR: float(
                import_config.get(CONF_CALIBRATION_FACTOR, 2000.0)
            ),
            CONF_UPDATE_INTERVAL: int(import_config.get(CONF_UPDATE_INTERVAL, 30)),
            CONF_UNAVAILABLE_BELOW: float_if_value(
                import_config.get(CONF_UNAVAILABLE_BELOW)
            ),
            CONF_UNAVAILABLE_ABOVE: float_if_value(
                import_config.get(CONF_UNAVAILABLE_ABOVE)
            ),
        }

        unique_id = build_entry_unique_id(name, chosen_entity, chosen_url)
        await self.async_set_unique_id(unique_id)
        self._abort_if_unique_id_configured()
        return self.async_create_entry(title=name, data=data)

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
        unavail_below_def = float_if_value(data.get(CONF_UNAVAILABLE_BELOW))
        unavail_above_def = float_if_value(data.get(CONF_UNAVAILABLE_ABOVE))

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
            unavail_below_in = float_if_value(
                user_input.get(CONF_UNAVAILABLE_BELOW)
            )
            unavail_above_in = float_if_value(
                user_input.get(CONF_UNAVAILABLE_ABOVE)
            )
            roi_enabled_in = bool(user_input.get(UI_ROI_ENABLED, roi_enabled_def))

            roi_new: dict = {CONF_ROI_ENABLED: roi_enabled_in}

            if CONF_X in user_input:
                x_val = int_if_value(user_input.get(CONF_X))
                if x_val is not None:
                    roi_new[CONF_X] = x_val
            if CONF_Y in user_input:
                y_val = int_if_value(user_input.get(CONF_Y))
                if y_val is not None:
                    roi_new[CONF_Y] = y_val
            if CONF_WIDTH in user_input:
                w_val = int_if_value(user_input.get(CONF_WIDTH))
                if w_val is not None:
                    roi_new[CONF_WIDTH] = w_val
            if CONF_HEIGHT in user_input:
                h_val = int_if_value(user_input.get(CONF_HEIGHT))
                if h_val is not None:
                    roi_new[CONF_HEIGHT] = h_val

            if source == SOURCE_CAMERA and not entity_in:
                errors[CONF_ENTITY_ID] = "required"
            if source == SOURCE_URL and not url_in:
                errors[CONF_IMAGE_URL] = "required"

            if not errors:
                new_opts = {
                    CONF_SOURCE: source,
                    CONF_CALIBRATION_FACTOR: float(cal_in),
                    CONF_UPDATE_INTERVAL: int(interval_in),
                    CONF_BRIGHTNESS_ROI: roi_new,
                }
                if source == SOURCE_CAMERA and entity_in is not None:
                    new_opts[CONF_ENTITY_ID] = entity_in
                if source == SOURCE_URL and url_in:
                    new_opts[CONF_IMAGE_URL] = url_in
                if unavail_below_in is not None:
                    new_opts[CONF_UNAVAILABLE_BELOW] = unavail_below_in
                elif data.get(CONF_UNAVAILABLE_BELOW) == 0:
                    new_opts[CONF_UNAVAILABLE_BELOW] = 0
                if unavail_above_in is not None:
                    new_opts[CONF_UNAVAILABLE_ABOVE] = unavail_above_in
                elif data.get(CONF_UNAVAILABLE_ABOVE) == 0:
                    new_opts[CONF_UNAVAILABLE_ABOVE] = 0
                return self.async_create_entry(title="", data=new_opts)

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

        if isinstance(entity_def, str) and entity_def:
            entity_field = vol.Optional(CONF_ENTITY_ID, default=entity_def)
        else:
            entity_field = vol.Optional(CONF_ENTITY_ID)

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
        fields[entity_field] = selector.selector({"entity": {"domain": "camera"}})
        fields[vol.Optional(CONF_IMAGE_URL, default=url_def)] = selector.selector(
            {"text": {}}
        )

        fields[vol.Required(CONF_CALIBRATION_FACTOR, default=cal_def)] = (
            selector.selector({"number": {"min": 1, "step": 1, "mode": "box"}})
        )
        fields[vol.Required(CONF_UPDATE_INTERVAL, default=interval_def)] = (
            selector.selector({"number": {"min": 5, "step": 1, "mode": "box"}})
        )

        fields[vol.Optional(CONF_UNAVAILABLE_BELOW)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )
        fields[vol.Optional(CONF_UNAVAILABLE_ABOVE)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        fields[vol.Required(UI_ROI_ENABLED, default=roi_enabled_def)] = (
            selector.selector({"boolean": {}})
        )

        fields[vol.Optional(CONF_X)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        fields[vol.Optional(CONF_Y)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        fields[vol.Optional(CONF_WIDTH)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        fields[vol.Optional(CONF_HEIGHT)] = selector.selector(
            {"number": {"min": 0, "mode": "box"}}
        )

        return self.async_show_form(
            step_id="edit", data_schema=vol.Schema(fields), errors=errors
        )
