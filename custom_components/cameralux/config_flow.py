# custom_components/cameralux/config_flow.py
from __future__ import annotations

import logging
import math

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    ALGORITHM_HUMAN,
    ALGORITHM_HUMAN_HR,
    ALGORITHM_LINEAR,
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
    CONFIG_ENTRY_VERSION,
    DEFAULT_ALGORITHM,
    DEFAULT_FULL_SCALE_LUX,
    DEFAULT_IR_ALGORITHM,
    DEFAULT_IR_CHROMA_THRESHOLD,
    DEFAULT_IR_ENABLED,
    DEFAULT_IR_FIXED_LUX,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
    IR_ALGORITHM_FIXED,
    IR_ALGORITHM_UNAVAILABLE,
    MAX_CHROMA_THRESHOLD,
    MIN_CHROMA_THRESHOLD,
    MIN_FULL_SCALE_LUX,
    MIN_LUX_BOUND,
    SOURCE_CAMERA,
    SOURCE_URL,
)
from .helpers import build_unique_id, float_if_value, int_if_value

_LOGGER = logging.getLogger(__name__)
UI_ROI_ENABLED = "roi_enabled"  # UI-only toggle name

_NUM = selector.selector(
    {"number": {"min": MIN_LUX_BOUND, "mode": "box", "step": "any"}}
)

# One schema for both the user step and the options step ("name" is prepended
# in the user step only); values are prefilled via add_suggested_values_to_schema.
STEP_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_SOURCE, default=SOURCE_CAMERA): selector.selector(
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
        vol.Optional(CONF_ENTITY_ID): selector.selector(
            {"entity": {"domain": "camera"}}
        ),
        vol.Optional(CONF_IMAGE_URL): selector.selector({"text": {}}),
        # Named algorithm chooses the (statistic, exponent) pair for color
        # frames; the numeric exponent below applies only to "custom".
        vol.Required(CONF_ALGORITHM, default=DEFAULT_ALGORITHM): selector.selector(
            {
                "select": {
                    "options": [
                        {"label": "Human brightness", "value": ALGORITHM_HUMAN},
                        {
                            "label": "Human brightness, highlight-resistant",
                            "value": ALGORITHM_HUMAN_HR,
                        },
                        {
                            "label": "Linear (fixed-exposure / no-IR cameras)",
                            "value": ALGORITHM_LINEAR,
                        },
                    ],
                    "mode": "dropdown",
                }
            }
        ),
        vol.Required(
            CONF_FULL_SCALE_LUX, default=DEFAULT_FULL_SCALE_LUX
        ): selector.selector(
            {"number": {"min": MIN_FULL_SCALE_LUX, "mode": "box", "step": "any"}}
        ),
        vol.Optional(CONF_UNAVAILABLE_BELOW): _NUM,
        vol.Optional(CONF_UNAVAILABLE_ABOVE): _NUM,
        # IR: detection (chroma pre-filter + gray-pixel verification) runs
        # only when enabled; cameras flip to IR at different chroma levels,
        # so the threshold is per-camera. The IR algorithm is the policy
        # applied after an IR frame is detected: measure it with a named
        # curve, report a fixed value, or go offline.
        vol.Required(CONF_IR_ENABLED, default=DEFAULT_IR_ENABLED): selector.selector(
            {"boolean": {}}
        ),
        vol.Required(
            CONF_IR_CHROMA_THRESHOLD, default=DEFAULT_IR_CHROMA_THRESHOLD
        ): selector.selector(
            {
                "number": {
                    "min": MIN_CHROMA_THRESHOLD,
                    "max": MAX_CHROMA_THRESHOLD,
                    "step": 0.5,
                    "mode": "box",
                }
            }
        ),
        vol.Required(
            CONF_IR_ALGORITHM, default=DEFAULT_IR_ALGORITHM
        ): selector.selector(
            {
                "select": {
                    "options": [
                        {"label": "Human brightness", "value": ALGORITHM_HUMAN},
                        {
                            "label": "Human brightness, highlight-resistant",
                            "value": ALGORITHM_HUMAN_HR,
                        },
                        {"label": "Linear", "value": ALGORITHM_LINEAR},
                        {
                            "label": "Fixed brightness — report a set lux while IR detected",
                            "value": IR_ALGORITHM_FIXED,
                        },
                        {
                            "label": "Unavailable — go offline while IR detected",
                            "value": IR_ALGORITHM_UNAVAILABLE,
                        },
                    ],
                    "mode": "dropdown",
                }
            }
        ),
        vol.Required(CONF_IR_FIXED_LUX, default=DEFAULT_IR_FIXED_LUX): _NUM,
        vol.Required(
            CONF_UPDATE_INTERVAL, default=DEFAULT_UPDATE_INTERVAL
        ): selector.selector({"number": {"min": 5, "step": 1, "mode": "box"}}),
        vol.Required(UI_ROI_ENABLED, default=False): selector.selector({"boolean": {}}),
        vol.Optional(CONF_X): _NUM,
        vol.Optional(CONF_Y): _NUM,
        vol.Optional(CONF_WIDTH): _NUM,
        vol.Optional(CONF_HEIGHT): _NUM,
    }
)


def _is_blank(raw) -> bool:
    """True when a field was omitted or submitted empty."""
    return raw is None or (isinstance(raw, str) and not raw.strip())


def _optional_float(
    user_input: dict,
    key: str,
    errors: dict[str, str],
    min_value: float,
    max_value: float | None = None,
) -> float | None:
    """Parse an optional bounded float.

    Returns None when the field is omitted or blank. A field that is present but
    unparseable, non-finite, or out of range records an error and returns None.
    """
    raw = user_input.get(key)
    if _is_blank(raw):
        return None
    parsed = float_if_value(raw)
    if (
        parsed is None  # present but not a number
        or not math.isfinite(parsed)
        or parsed < min_value
        or (max_value is not None and parsed > max_value)
    ):
        errors[key] = "invalid_input"
        return None
    return parsed


def _required_float(
    user_input: dict,
    key: str,
    default: float,
    errors: dict[str, str],
    min_value: float,
    max_value: float | None = None,
) -> float:
    """Parse a required bounded float, defaulting only when omitted or blank.

    A present-but-invalid value records an error. The returned value is only
    meaningful when `errors` is empty.
    """
    if _is_blank(user_input.get(key)):
        return default
    parsed = _optional_float(user_input, key, errors, min_value, max_value)
    return default if parsed is None else parsed


def _validate_bounds(
    below: float | None,
    above: float | None,
    above_key: str,
    errors: dict[str, str],
) -> None:
    """Reject contradictory unavailable bounds."""
    if below is not None and above is not None and below > above:
        errors[above_key] = "invalid_input"


def _parse_step(user_input: dict) -> tuple[dict, dict[str, str]]:
    """Validate/normalize a submitted form into entry data; returns (data, errors)."""
    errors: dict[str, str] = {}
    source = user_input.get(CONF_SOURCE, SOURCE_CAMERA)
    entity_id = user_input.get(CONF_ENTITY_ID)
    image_url = (user_input.get(CONF_IMAGE_URL) or "").strip()

    if source == SOURCE_CAMERA and not entity_id:
        errors[CONF_ENTITY_ID] = "required"
    if source == SOURCE_URL and not image_url:
        errors[CONF_IMAGE_URL] = "required"

    roi = {CONF_ROI_ENABLED: bool(user_input.get(UI_ROI_ENABLED, False))}
    for key in (CONF_X, CONF_Y, CONF_WIDTH, CONF_HEIGHT):
        val = int_if_value(user_input.get(key))
        if val is not None:
            roi[key] = val

    full_scale_lux = _required_float(
        user_input,
        CONF_FULL_SCALE_LUX,
        DEFAULT_FULL_SCALE_LUX,
        errors,
        MIN_FULL_SCALE_LUX,
    )
    algorithm = user_input.get(CONF_ALGORITHM) or DEFAULT_ALGORITHM
    if algorithm not in (
        ALGORITHM_HUMAN,
        ALGORITHM_HUMAN_HR,
        ALGORITHM_LINEAR,
    ):
        errors[CONF_ALGORITHM] = "invalid_input"
        algorithm = DEFAULT_ALGORITHM
    unavailable_below = _optional_float(
        user_input, CONF_UNAVAILABLE_BELOW, errors, MIN_LUX_BOUND
    )
    unavailable_above = _optional_float(
        user_input, CONF_UNAVAILABLE_ABOVE, errors, MIN_LUX_BOUND
    )
    _validate_bounds(
        unavailable_below, unavailable_above, CONF_UNAVAILABLE_ABOVE, errors
    )

    ir_enabled = bool(user_input.get(CONF_IR_ENABLED, DEFAULT_IR_ENABLED))
    # Cameras flip to IR at different chroma levels; per-camera threshold.
    ir_chroma_threshold = _required_float(
        user_input,
        CONF_IR_CHROMA_THRESHOLD,
        DEFAULT_IR_CHROMA_THRESHOLD,
        errors,
        MIN_CHROMA_THRESHOLD,
        MAX_CHROMA_THRESHOLD,
    )
    ir_algorithm = user_input.get(CONF_IR_ALGORITHM) or DEFAULT_IR_ALGORITHM
    if ir_algorithm not in (
        ALGORITHM_HUMAN,
        ALGORITHM_HUMAN_HR,
        ALGORITHM_LINEAR,
        IR_ALGORITHM_FIXED,
        IR_ALGORITHM_UNAVAILABLE,
    ):
        errors[CONF_IR_ALGORITHM] = "invalid_input"
        ir_algorithm = DEFAULT_IR_ALGORITHM
    # Stored regardless of the selection so switching and back keeps values.
    ir_fixed_lux = _required_float(
        user_input,
        CONF_IR_FIXED_LUX,
        DEFAULT_IR_FIXED_LUX,
        errors,
        MIN_LUX_BOUND,
    )
    data = {
        CONF_SOURCE: source,
        CONF_ENTITY_ID: entity_id if source == SOURCE_CAMERA else None,
        CONF_IMAGE_URL: image_url if source == SOURCE_URL else None,
        CONF_BRIGHTNESS_ROI: roi,
        CONF_FULL_SCALE_LUX: full_scale_lux,
        CONF_ALGORITHM: algorithm,
        CONF_UPDATE_INTERVAL: int_if_value(user_input.get(CONF_UPDATE_INTERVAL))
        or DEFAULT_UPDATE_INTERVAL,
        CONF_UNAVAILABLE_BELOW: unavailable_below,
        CONF_UNAVAILABLE_ABOVE: unavailable_above,
        CONF_IR_ENABLED: ir_enabled,
        CONF_IR_CHROMA_THRESHOLD: ir_chroma_threshold,
        CONF_IR_ALGORITHM: ir_algorithm,
        CONF_IR_FIXED_LUX: ir_fixed_lux,
    }
    return data, errors


def _suggested_from(data: dict) -> dict:
    """Flatten stored entry data into form-field suggested values."""
    roi = data.get(CONF_BRIGHTNESS_ROI) or {}
    suggested = {k: v for k, v in data.items() if v is not None}
    suggested.pop(CONF_BRIGHTNESS_ROI, None)
    suggested[UI_ROI_ENABLED] = bool(roi.get(CONF_ROI_ENABLED, False))
    for key in (CONF_X, CONF_Y, CONF_WIDTH, CONF_HEIGHT):
        if roi.get(key) is not None:
            suggested[key] = roi[key]

    # Do not surface stored values outside the current public vocabulary.
    # Saving Options replaces them with an explicit current choice.
    if suggested.get(CONF_ALGORITHM) not in ALGORITHM_PROFILES:
        suggested[CONF_ALGORITHM] = DEFAULT_ALGORITHM
    if suggested.get(CONF_IR_ALGORITHM) not in (
        *ALGORITHM_PROFILES,
        IR_ALGORITHM_FIXED,
        IR_ALGORITHM_UNAVAILABLE,
    ):
        suggested[CONF_IR_ALGORITHM] = DEFAULT_IR_ALGORITHM
    return suggested


class CameraLuxConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for CameraLux."""

    VERSION = CONFIG_ENTRY_VERSION

    async def async_step_user(self, user_input=None):
        """Initial setup: name + shared settings schema."""
        errors: dict[str, str] = {}

        if user_input is not None:
            name = (user_input.get("name") or "").strip()
            if not name:
                errors["name"] = "required"
            data, step_errors = _parse_step(user_input)
            errors.update(step_errors)
            if not errors:
                data["name"] = name
                unique_id = build_unique_id(
                    name, data.get(CONF_ENTITY_ID), data.get(CONF_IMAGE_URL)
                )
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(title=name, data=data)

        schema = vol.Schema({vol.Required("name"): str, **STEP_SCHEMA.schema})
        if user_input is not None:
            schema = self.add_suggested_values_to_schema(schema, user_input)
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(entry):
        return CameraLuxOptionsFlow()


class CameraLuxOptionsFlow(config_entries.OptionsFlow):
    """Options: same shared schema, prefilled from the entry."""

    async def async_step_init(self, user_input=None):
        errors: dict[str, str] = {}

        if user_input is not None:
            data, errors = _parse_step(user_input)
            if not errors:
                return self.async_create_entry(title="", data=data)
            suggested = user_input
        else:
            suggested = _suggested_from(
                {**self.config_entry.data, **self.config_entry.options}
            )

        schema = self.add_suggested_values_to_schema(STEP_SCHEMA, suggested)
        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)
