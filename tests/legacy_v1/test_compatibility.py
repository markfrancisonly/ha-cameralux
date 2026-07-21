"""Tests contained within the severable Camera Lux 1.x boundary."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("homeassistant", reason="legacy sensor imports Home Assistant")

from custom_components.cameralux.config_flow import (  # noqa: E402
    STEP_SCHEMA,
    _parse_step,
)
from custom_components.cameralux.const import (  # noqa: E402
    ALGORITHM_HUMAN,
    ALGORITHM_LINEAR,
    CONF_ALGORITHM,
    CONF_ENTITY_ID,
    CONF_FULL_SCALE_LUX,
    CONF_IR_ALGORITHM,
    CONF_SOURCE,
    SOURCE_CAMERA,
)
from custom_components.cameralux.legacy_v1 import LEGACY_ALGORITHM  # noqa: E402
from custom_components.cameralux.legacy_v1.sensor import (  # noqa: E402
    V1CameraLuxSensor,
)
from custom_components.cameralux.sensor import (  # noqa: E402
    CameraLuxSensor,
    async_setup_entry,
)

BASE = {CONF_SOURCE: SOURCE_CAMERA, CONF_ENTITY_ID: "camera.x"}


def make_legacy_sensor(**config):
    return V1CameraLuxSensor(None, "T", {**BASE, **config}, "uid")


def test_legacy_transfer_is_preserved():
    sensor = make_legacy_sensor(**{CONF_FULL_SCALE_LUX: 2000.0})
    assert sensor.algorithm == LEGACY_ALGORITHM
    assert sensor.color_trim_fraction == 0.0

    for average in (0.0, 0.043, 0.5, 1.0):
        assert math.isclose(sensor._perceive(average), math.log1p(average))
    assert math.isclose(sensor.max_lux, math.log1p(1.0) * 2000.0)
    assert "perception_exponent" not in sensor.extra_state_attributes


def test_legacy_ir_transfer_and_modern_ir_exit():
    sensor = make_legacy_sensor(
        **{
            CONF_IR_ALGORITHM: LEGACY_ALGORITHM,
            CONF_FULL_SCALE_LUX: 2000.0,
        }
    )
    assert sensor.ir_algorithm == LEGACY_ALGORITHM
    assert sensor._ir_measures()
    for average in (0.0, 0.2, 1.0):
        assert math.isclose(sensor._ir_perceive(average), math.log1p(average))

    modern_ir = make_legacy_sensor(**{CONF_IR_ALGORITHM: ALGORITHM_LINEAR})
    assert modern_ir._ir_perceive(0.25) == 0.25


def test_current_config_flow_does_not_offer_or_accept_legacy():
    _, errors = _parse_step({**BASE, CONF_ALGORITHM: LEGACY_ALGORITHM})
    assert errors[CONF_ALGORITHM] == "invalid_input"

    algorithm_field = next(
        value
        for key, value in STEP_SCHEMA.schema.items()
        if getattr(key, "schema", None) == CONF_ALGORITHM
    )
    offered = {option["value"] for option in algorithm_field.config["options"]}
    assert LEGACY_ALGORITHM not in offered


def test_setup_dispatches_only_migrated_entries_to_legacy_sensor():
    added: list[CameraLuxSensor] = []

    def build(algorithm: str) -> CameraLuxSensor:
        added.clear()
        entry = SimpleNamespace(
            data={**BASE, "name": "T", CONF_ALGORITHM: algorithm},
            options={},
            entry_id="uid",
        )
        asyncio.run(
            async_setup_entry(None, entry, lambda entities: added.extend(entities))
        )
        return added[0]

    assert type(build(LEGACY_ALGORITHM)) is V1CameraLuxSensor
    assert type(build(ALGORITHM_HUMAN)) is CameraLuxSensor
