"""Config flow + sensor. Needs Home Assistant (selectors, voluptuous, entity).

Self-skips when HA is absent so `pytest` still passes on a bare checkout;
requirements-test.txt installs HA so CI actually runs these.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("homeassistant", reason="config_flow/sensor import Home Assistant")

from custom_components.cameralux.config_flow import (  # noqa: E402
    STEP_SCHEMA,
    CameraLuxConfigFlow,
    _parse_step,
    _suggested_from,
)
from custom_components.cameralux.const import (  # noqa: E402
    ALGORITHM_HUMAN,
    ALGORITHM_HUMAN_HR,
    ALGORITHM_LINEAR,
    CONF_ALGORITHM,
    CONF_BRIGHTNESS_ROI,
    CONF_ENTITY_ID,
    CONF_FULL_SCALE_LUX,
    CONF_IR_ALGORITHM,
    CONF_IR_CHROMA_THRESHOLD,
    CONF_IR_ENABLED,
    CONF_IR_FIXED_LUX,
    CONF_ROI_ENABLED,
    CONF_SOURCE,
    CONF_UNAVAILABLE_ABOVE,
    CONF_UNAVAILABLE_BELOW,
    CONFIG_ENTRY_VERSION,
    DEFAULT_ALGORITHM,
    DEFAULT_FULL_SCALE_LUX,
    DEFAULT_IR_ALGORITHM,
    DEFAULT_IR_CHROMA_THRESHOLD,
    DEFAULT_IR_FIXED_LUX,
    IR_ALGORITHM_FIXED,
    IR_ALGORITHM_UNAVAILABLE,
    SOURCE_CAMERA,
)
from custom_components.cameralux.sensor import (  # noqa: E402
    CameraLuxSensor,
    _redact_url,
    async_setup_entry,
)

BASE = {CONF_SOURCE: SOURCE_CAMERA, CONF_ENTITY_ID: "camera.x"}


def parse(**overrides):
    return _parse_step({**BASE, **overrides})


def make_sensor(**cfg):
    base = {
        CONF_ENTITY_ID: "camera.x",
        CONF_FULL_SCALE_LUX: 1000,
        CONF_IR_CHROMA_THRESHOLD: 6.0,
    }
    base.update(cfg)
    return CameraLuxSensor(None, "T", base, "uid")


# --- config flow ----------------------------------------------------------


def test_flow_version_matches_schema_version():
    """A drifted VERSION would stamp new entries with the wrong schema and skip
    migration silently."""
    assert CameraLuxConfigFlow.VERSION == CONFIG_ENTRY_VERSION


def test_schema_fills_required_fields_from_a_bare_submission():
    """Required-with-default: a blank box arrives as a concrete value."""
    out = STEP_SCHEMA(dict(BASE))
    assert out[CONF_FULL_SCALE_LUX] == DEFAULT_FULL_SCALE_LUX
    assert out[CONF_IR_ALGORITHM] == DEFAULT_IR_ALGORITHM
    assert out[CONF_IR_FIXED_LUX] == DEFAULT_IR_FIXED_LUX


def test_omitted_and_blank_take_the_default():
    assert parse()[0][CONF_FULL_SCALE_LUX] == DEFAULT_FULL_SCALE_LUX
    assert parse(**{CONF_FULL_SCALE_LUX: ""})[0][CONF_FULL_SCALE_LUX] == (
        DEFAULT_FULL_SCALE_LUX
    )


@pytest.mark.parametrize("bad", ["abc", "nan", 0, -1])
def test_present_but_invalid_full_scale_errors_instead_of_defaulting(bad):
    assert (
        parse(**{CONF_FULL_SCALE_LUX: bad})[1][CONF_FULL_SCALE_LUX] == "invalid_input"
    )


def test_contradictory_bounds_are_rejected():
    assert (
        parse(**{CONF_UNAVAILABLE_BELOW: 100, CONF_UNAVAILABLE_ABOVE: 50})[1][
            CONF_UNAVAILABLE_ABOVE
        ]
        == "invalid_input"
    )


# --- algorithm selection ---------------------------------------------------


def test_bare_submission_stores_the_default_algorithm():
    data, errors = parse()
    assert not errors
    assert data[CONF_ALGORITHM] == ALGORITHM_HUMAN


def test_unknown_algorithm_is_an_error():
    _, errors = parse(**{CONF_ALGORITHM: "bogus"})
    assert errors[CONF_ALGORITHM] == "invalid_input"


def test_algorithm_resolution_on_the_sensor():
    human = make_sensor(**{CONF_ALGORITHM: ALGORITHM_HUMAN})
    assert (human.exponent, human.color_trim_fraction) == (0.5, 0.0)
    hr = make_sensor(**{CONF_ALGORITHM: ALGORITHM_HUMAN_HR})
    assert (hr.exponent, hr.color_trim_fraction) == (0.5, 0.05)
    lin = make_sensor(**{CONF_ALGORITHM: ALGORITHM_LINEAR})
    assert (lin.exponent, lin.color_trim_fraction) == (1.0, 0.0)


def test_unknown_stored_algorithms_do_not_leak_into_the_options_ui():
    suggested = _suggested_from(
        {CONF_ALGORITHM: "retired", CONF_IR_ALGORITHM: "retired"}
    )
    assert suggested[CONF_ALGORITHM] == DEFAULT_ALGORITHM
    assert suggested[CONF_IR_ALGORITHM] == DEFAULT_IR_ALGORITHM


# --- night (IR) mode ---------------------------------------------------------


def test_ir_algorithm_defaults_and_validation():
    data, errors = parse()
    assert not errors
    assert data[CONF_IR_ALGORITHM] == IR_ALGORITHM_FIXED
    assert data[CONF_IR_FIXED_LUX] == 0.0

    _, errors = parse(**{CONF_IR_ALGORITHM: "bogus"})
    assert errors[CONF_IR_ALGORITHM] == "invalid_input"

    for choice in (
        ALGORITHM_HUMAN,
        ALGORITHM_HUMAN_HR,
        ALGORITHM_LINEAR,
        IR_ALGORITHM_FIXED,
        IR_ALGORITHM_UNAVAILABLE,
    ):
        data, errors = parse(**{CONF_IR_ALGORITHM: choice})
        assert not errors
        assert data[CONF_IR_ALGORITHM] == choice


def test_ir_fixed_lux_is_stored_regardless_of_selection():
    """Switching selections and back must not discard configured values."""
    data, errors = parse(
        **{CONF_IR_ALGORITHM: IR_ALGORITHM_UNAVAILABLE, CONF_IR_FIXED_LUX: 3}
    )
    assert not errors
    assert data[CONF_IR_FIXED_LUX] == 3.0


def test_invalid_ir_fixed_lux_errors():
    assert parse(**{CONF_IR_FIXED_LUX: "xyz"})[1][CONF_IR_FIXED_LUX] == "invalid_input"
    assert parse(**{CONF_IR_FIXED_LUX: -1})[1][CONF_IR_FIXED_LUX] == "invalid_input"


def test_ir_enabled_checkbox_parses_and_defaults_off():
    data, errors = parse()
    assert not errors
    assert data[CONF_IR_ENABLED] is False
    assert parse(**{CONF_IR_ENABLED: True})[0][CONF_IR_ENABLED] is True


def test_chroma_threshold_on_the_form():
    """Cameras flip to IR at different chroma levels; per-camera tuning."""
    data, errors = parse()
    assert not errors
    assert data[CONF_IR_CHROMA_THRESHOLD] == DEFAULT_IR_CHROMA_THRESHOLD
    data, errors = parse(**{CONF_IR_CHROMA_THRESHOLD: 2.5})
    assert not errors
    assert data[CONF_IR_CHROMA_THRESHOLD] == 2.5
    assert (
        parse(**{CONF_IR_CHROMA_THRESHOLD: 999})[1][CONF_IR_CHROMA_THRESHOLD]
        == "invalid_input"
    )


def test_ir_resolution_on_the_sensor():
    s = make_sensor()
    assert s.ir_enabled is False  # default: detection off
    assert s.ir_algorithm == IR_ALGORITHM_FIXED  # default policy
    assert s.ir_fixed_lux == 0.0
    assert not s._ir_measures()

    measuring = make_sensor(**{CONF_IR_ALGORITHM: ALGORITHM_LINEAR})
    assert measuring.ir_algorithm == ALGORITHM_LINEAR
    assert measuring._ir_measures()
    assert (measuring.ir_trim_fraction, measuring.ir_exponent) == (0.0, 1.0)
    assert measuring._ir_perceive(0.25) == 0.25  # linear

    hr = make_sensor(**{CONF_IR_ALGORITHM: ALGORITHM_HUMAN_HR})
    assert (hr.ir_trim_fraction, hr.ir_exponent) == (0.05, 0.5)

    assert (
        make_sensor(**{CONF_IR_ALGORITHM: "bogus"}).ir_algorithm == IR_ALGORITHM_FIXED
    )
    fixed = make_sensor(**{CONF_IR_FIXED_LUX: 2.5})
    assert fixed.ir_fixed_lux == 2.5
    enabled = make_sensor(**{CONF_IR_ENABLED: True})
    assert enabled.ir_enabled is True


# --- max_lux ----------------------------------------------------------------


def test_max_lux_is_the_perceived_white_frame():
    assert make_sensor().max_lux == 1000.0  # 1.0 ** exponent * full scale
    lin = make_sensor(**{CONF_ALGORITHM: ALGORITHM_LINEAR, CONF_FULL_SCALE_LUX: 500})
    assert lin.max_lux == 500.0


# --- state attributes: exact contract (order, names, values) --------------


def test_attributes_order_names_and_ir_detected():
    sensor = make_sensor(
        **{
            CONF_IR_ENABLED: True,
            CONF_UNAVAILABLE_BELOW: 1,
            CONF_UNAVAILABLE_ABOVE: 900,
        }
    )
    sensor.ir_active = True
    attrs = sensor.extra_state_attributes
    assert list(attrs) == [
        "chroma",
        "ir_detected",
        "avg_luminance",
        "perceived_luminance",
        "algorithm",
        "perception_exponent",
        "max_lux",
        "unavailable_above_lux",
        "unavailable_below_lux",
    ]
    assert attrs["ir_detected"] is True
    # the policy in force on an IR frame is the IR selection
    assert attrs["algorithm"] == IR_ALGORITHM_FIXED
    assert attrs["max_lux"] == 1000.0
    sensor.ir_active = False
    assert sensor.extra_state_attributes["algorithm"] == ALGORITHM_HUMAN


def test_effective_roi_attribute_is_runtime_diagnostic_only():
    sensor = make_sensor(**{CONF_BRIGHTNESS_ROI: {CONF_ROI_ENABLED: True, "width": 50}})
    assert "effective_roi" not in sensor.extra_state_attributes
    sensor.effective_roi = {"x": 0, "y": 0, "width": 50, "height": 100}
    assert sensor.extra_state_attributes["effective_roi"] == {
        "x": 0,
        "y": 0,
        "width": 50,
        "height": 100,
    }

    disabled = make_sensor()
    disabled.effective_roi = sensor.effective_roi
    assert "effective_roi" not in disabled.extra_state_attributes


def test_ir_detected_is_a_key_attribute_gated_by_the_checkbox():
    """Present (true/false) whenever detection runs; absent when detection is
    off — claiming IR/not-IR without looking would be a lie."""
    enabled = make_sensor(**{CONF_IR_ENABLED: True}).extra_state_attributes
    assert enabled["ir_detected"] is False  # detection ran, frame is color
    assert "ir_detected" not in make_sensor().extra_state_attributes


def test_unavailable_bounds_appear_only_when_set():
    plain = make_sensor().extra_state_attributes
    assert "unavailable_above_lux" not in plain and "unavailable_below_lux" not in plain
    bounded = make_sensor(**{CONF_UNAVAILABLE_ABOVE: 900}).extra_state_attributes
    assert bounded["unavailable_above_lux"] == 900.0
    assert "unavailable_below_lux" not in bounded


def test_algorithm_attribute_reflects_the_configured_choice():
    lin = make_sensor(**{CONF_ALGORITHM: ALGORITHM_LINEAR}).extra_state_attributes
    assert lin["algorithm"] == ALGORITHM_LINEAR
    assert lin["perception_exponent"] == 1.0
    hr = make_sensor(**{CONF_ALGORITHM: ALGORITHM_HUMAN_HR}).extra_state_attributes
    assert hr["algorithm"] == ALGORITHM_HUMAN_HR


def test_sensor_unavailable_until_a_lux_value_exists():
    sensor = make_sensor()
    assert sensor.available is False and sensor.native_value is None
    sensor.lux = 12.5
    assert sensor.available is True and sensor.native_value == 12.5


# --- credential redaction -------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://user:pass@cam/snap.jpg", "http://***@cam/snap.jpg"),
        ("https://cam.local/snap.jpg", "https://cam.local/snap.jpg"),
        # an '@' in the query must NOT be mistaken for credentials
        ("http://cam/snap?token=a@b", "http://cam/snap?token=a@b"),
        (None, None),
    ],
)
def test_redact_url_strips_credentials(url, expected):
    assert _redact_url(url) == expected


# --- stale-camera gate ------------------------------------------------------


def _hass_with_camera_state(state):
    get = lambda _eid: SimpleNamespace(state=state) if state else None  # noqa: E731
    return SimpleNamespace(states=SimpleNamespace(get=get))


@pytest.mark.parametrize("cam_state", ["unavailable", "unknown", None])
def test_snapshot_is_refused_when_the_camera_is_not_live(cam_state):
    """async_get_image can serve a cached frame for a dead camera; the sensor
    must go unavailable instead of measuring months-old pixels."""
    sensor = make_sensor()
    sensor.hass = _hass_with_camera_state(cam_state)
    assert asyncio.run(sensor.async_get_camera_snapshot("camera.x")) is None


# --- setup ----------------------------------------------------------------


def test_async_setup_entry_adds_one_entity_with_stable_id():
    entry = SimpleNamespace(
        data={CONF_ENTITY_ID: "camera.test", "name": "Test"}, options={}, entry_id="1"
    )
    add_entities = MagicMock()
    asyncio.run(async_setup_entry(None, entry, add_entities))
    add_entities.assert_called_once()
    (entities,) = add_entities.call_args[0]
    assert len(entities) == 1 and entities[0].unique_id == "1"


def test_async_setup_entry_options_override_data():
    entry = SimpleNamespace(
        data={CONF_ENTITY_ID: "camera.test", CONF_FULL_SCALE_LUX: 2000.0, "name": "T"},
        options={CONF_FULL_SCALE_LUX: 500.0},
        entry_id="1",
    )
    add_entities = MagicMock()
    asyncio.run(async_setup_entry(None, entry, add_entities))
    (entities,) = add_entities.call_args[0]
    assert entities[0].max_lux == 500.0


# --- frozen-feed detection (same-cycle pair fetch) ------------------------


def _jpeg(color):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
    return buf.getvalue()


def _rig(monkeypatch, sensor, frames):
    """Wire a scripted snapshot sequence and neutralize the verify delay."""
    from types import SimpleNamespace

    from homeassistant.core import CoreState

    from custom_components.cameralux import sensor as sensor_mod

    async def fake_snapshot(_entity_id):
        return frames.pop(0)

    async def fake_executor(fn, *args):
        return fn(*args)

    sensor.hass = SimpleNamespace(
        state=CoreState.running, async_add_executor_job=fake_executor
    )
    monkeypatch.setattr(sensor, "async_get_camera_snapshot", fake_snapshot)
    monkeypatch.setattr(sensor, "async_write_ha_state", lambda: None)
    monkeypatch.setattr(sensor_mod, "FROZEN_VERIFY_DELAY_SECONDS", 0)


def test_live_pair_publishes_immediately(monkeypatch):
    """Two differing fetches in one cycle = live feed: the fresh frame is
    measured with no cross-cycle delay."""
    sensor = make_sensor(frozen_detection=True)
    frames = [_jpeg((10, 10, 10)), _jpeg((200, 200, 200))]
    _rig(monkeypatch, sensor, frames)

    asyncio.run(sensor.async_update())
    assert sensor.lux is not None  # published in the SAME cycle


def test_frozen_pair_never_measured(monkeypatch):
    """Byte-identical pair = stuck/substituted frame: offline, nothing
    published — including on the very first outage."""
    sensor = make_sensor(frozen_detection=True)
    dummy = _jpeg((50, 50, 50))
    frames = [dummy, dummy]
    _rig(monkeypatch, sensor, frames)

    asyncio.run(sensor.async_update())
    assert sensor.lux is None

    # recovery: a live pair next cycle publishes right away
    frames[:] = [_jpeg((10, 10, 10)), _jpeg((201, 201, 201))]
    asyncio.run(sensor.async_update())
    assert sensor.lux is not None


def test_missing_verify_fetch_stays_offline(monkeypatch):
    """If the second fetch fails the frame is unverified: offline."""
    sensor = make_sensor(frozen_detection=True)
    frames = [_jpeg((10, 10, 10)), None]
    _rig(monkeypatch, sensor, frames)

    asyncio.run(sensor.async_update())
    assert sensor.lux is None


def test_detection_off_measures_single_fetch(monkeypatch):
    """Default (checkbox off): one fetch per cycle, identical bytes across
    cycles are measured normally — infrequently updated image sources must
    not flap unavailable."""
    sensor = make_sensor()  # frozen_detection defaults off
    static = _jpeg((80, 80, 80))
    frames = [static]
    _rig(monkeypatch, sensor, frames)

    asyncio.run(sensor.async_update())
    assert sensor.lux is not None  # measured immediately, no second pull
    assert frames == []  # exactly ONE fetch consumed

    frames[:] = [static]  # same bytes next cycle: still fine
    asyncio.run(sensor.async_update())
    assert sensor.lux is not None
