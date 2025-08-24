import asyncio
from types import SimpleNamespace

from custom_components.cameralux.config_flow import CameraLuxOptionsFlow, UI_ROI_ENABLED
from custom_components.cameralux.const import (
    CONF_BRIGHTNESS_ROI,
    CONF_CALIBRATION_FACTOR,
    CONF_ENTITY_ID,
    CONF_HEIGHT,
    CONF_ROI_ENABLED,
    CONF_SOURCE,
    CONF_UNAVAILABLE_ABOVE,
    CONF_UNAVAILABLE_BELOW,
    CONF_UPDATE_INTERVAL,
    CONF_WIDTH,
    CONF_X,
    CONF_Y,
    SOURCE_CAMERA,
)


class DummyEntry(SimpleNamespace):
    pass


def run_flow(flow, user_input):
    return asyncio.run(flow.async_step_edit(user_input))


def test_unavailable_thresholds_allow_zero_and_none():
    entry = DummyEntry(
        data={
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_ENTITY_ID: "camera.test",
            CONF_CALIBRATION_FACTOR: 2000.0,
            CONF_UPDATE_INTERVAL: 30,
            CONF_BRIGHTNESS_ROI: {CONF_ROI_ENABLED: False},
            CONF_UNAVAILABLE_BELOW: 5.0,
            CONF_UNAVAILABLE_ABOVE: 100.0,
        },
        options={},
        entry_id="1",
    )

    flow = CameraLuxOptionsFlow(entry)
    result = run_flow(
        flow,
        {
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_ENTITY_ID: "camera.test",
            CONF_CALIBRATION_FACTOR: 2000,
            CONF_UPDATE_INTERVAL: 30,
            UI_ROI_ENABLED: False,
            CONF_UNAVAILABLE_BELOW: "",
            CONF_UNAVAILABLE_ABOVE: "",
        },
    )
    data = result["data"]
    assert CONF_UNAVAILABLE_BELOW not in data
    assert CONF_UNAVAILABLE_ABOVE not in data

    entry.options = data
    flow = CameraLuxOptionsFlow(entry)
    result = run_flow(
        flow,
        {
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_ENTITY_ID: "camera.test",
            CONF_CALIBRATION_FACTOR: 2000,
            CONF_UPDATE_INTERVAL: 30,
            UI_ROI_ENABLED: False,
            CONF_UNAVAILABLE_BELOW: 0,
            CONF_UNAVAILABLE_ABOVE: 0,
        },
    )
    data = result["data"]
    assert data[CONF_UNAVAILABLE_BELOW] == 0
    assert data[CONF_UNAVAILABLE_ABOVE] == 0

    entry.options = data
    flow = CameraLuxOptionsFlow(entry)
    result = run_flow(
        flow,
        {
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_ENTITY_ID: "camera.test",
            CONF_CALIBRATION_FACTOR: 2000,
            CONF_UPDATE_INTERVAL: 30,
            UI_ROI_ENABLED: False,
            CONF_UNAVAILABLE_BELOW: "",
            CONF_UNAVAILABLE_ABOVE: "",
        },
    )
    data = result["data"]
    assert data[CONF_UNAVAILABLE_BELOW] == 0
    assert data[CONF_UNAVAILABLE_ABOVE] == 0


def test_roi_fields_allow_zero_and_none():
    entry = DummyEntry(
        data={
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_ENTITY_ID: "camera.test",
            CONF_CALIBRATION_FACTOR: 2000.0,
            CONF_UPDATE_INTERVAL: 30,
            CONF_BRIGHTNESS_ROI: {
                CONF_ROI_ENABLED: True,
                CONF_X: 1,
                CONF_Y: 2,
                CONF_WIDTH: 3,
                CONF_HEIGHT: 4,
            },
        },
        options={},
        entry_id="1",
    )

    flow = CameraLuxOptionsFlow(entry)
    result = run_flow(
        flow,
        {
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_ENTITY_ID: "camera.test",
            CONF_CALIBRATION_FACTOR: 2000,
            CONF_UPDATE_INTERVAL: 30,
            UI_ROI_ENABLED: True,
            CONF_X: "",
            CONF_Y: "",
            CONF_WIDTH: "",
            CONF_HEIGHT: "",
        },
    )
    roi = result["data"][CONF_BRIGHTNESS_ROI]
    assert CONF_X not in roi
    assert CONF_Y not in roi
    assert CONF_WIDTH not in roi
    assert CONF_HEIGHT not in roi

    entry.options = result["data"]
    flow = CameraLuxOptionsFlow(entry)
    result = run_flow(
        flow,
        {
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_ENTITY_ID: "camera.test",
            CONF_CALIBRATION_FACTOR: 2000,
            CONF_UPDATE_INTERVAL: 30,
            UI_ROI_ENABLED: True,
            CONF_X: 0,
            CONF_Y: 0,
            CONF_WIDTH: 0,
            CONF_HEIGHT: 0,
        },
    )
    roi = result["data"][CONF_BRIGHTNESS_ROI]
    assert roi[CONF_X] == 0
    assert roi[CONF_Y] == 0
    assert roi[CONF_WIDTH] == 0
    assert roi[CONF_HEIGHT] == 0

    entry.options = result["data"]
    flow = CameraLuxOptionsFlow(entry)
    result = run_flow(
        flow,
        {
            CONF_SOURCE: SOURCE_CAMERA,
            CONF_ENTITY_ID: "camera.test",
            CONF_CALIBRATION_FACTOR: 2000,
            CONF_UPDATE_INTERVAL: 30,
            UI_ROI_ENABLED: True,
            CONF_X: "",
            CONF_Y: "",
            CONF_WIDTH: "",
            CONF_HEIGHT: "",
        },
    )
    roi = result["data"][CONF_BRIGHTNESS_ROI]
    assert CONF_X not in roi
    assert CONF_Y not in roi
    assert CONF_WIDTH not in roi
    assert CONF_HEIGHT not in roi
