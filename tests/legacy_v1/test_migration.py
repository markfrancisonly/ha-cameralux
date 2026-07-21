"""Migration tests contained with the Camera Lux 1.x compatibility code."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from custom_components.cameralux import async_migrate_entry
from custom_components.cameralux.const import (
    CONF_ALGORITHM,
    CONF_IR_ALGORITHM,
    CONF_IR_FIXED_LUX,
    CONFIG_ENTRY_VERSION,
    DEFAULT_IR_FIXED_LUX,
)
from custom_components.cameralux.legacy_v1 import LEGACY_ALGORITHM

V1_FULL_SCALE = "calibration_factor"
FULL_SCALE = "full_scale_lux"


class FakeConfigEntries:
    def async_update_entry(self, entry, *, data=None, options=None, version=None):
        if data is not None:
            entry.data = data
        if options is not None:
            entry.options = options
        if version is not None:
            entry.version = version


def make_entry(version, data, options=None):
    return SimpleNamespace(
        version=version,
        data=dict(data),
        options=dict(options or {}),
        title="Cam",
    )


def migrate(entry):
    hass = SimpleNamespace(config_entries=FakeConfigEntries())
    return asyncio.run(async_migrate_entry(hass, entry))


def merged(entry):
    return {**entry.data, **entry.options}


def test_rename_in_data():
    entry = make_entry(1, {V1_FULL_SCALE: 2000.0})
    assert migrate(entry) is True
    assert entry.version == CONFIG_ENTRY_VERSION
    assert merged(entry)[FULL_SCALE] == 2000.0
    assert V1_FULL_SCALE not in entry.data


def test_rename_in_options_which_shadows_data():
    entry = make_entry(1, {V1_FULL_SCALE: 2000.0}, {V1_FULL_SCALE: 500.0})
    migrate(entry)
    assert V1_FULL_SCALE not in entry.data
    assert V1_FULL_SCALE not in entry.options
    assert merged(entry)[FULL_SCALE] == 500.0


def test_v1_entry_lands_on_the_compatibility_curve():
    entry = make_entry(1, {V1_FULL_SCALE: 2000.0})
    migrate(entry)
    config = merged(entry)
    assert config[CONF_ALGORITHM] == LEGACY_ALGORITHM
    assert config[CONF_IR_ALGORITHM] == LEGACY_ALGORITHM
    assert config[CONF_IR_FIXED_LUX] == DEFAULT_IR_FIXED_LUX
    assert config["ir_enabled"] is False


def test_explicit_current_algorithm_survives_migration():
    entry = make_entry(
        1,
        {V1_FULL_SCALE: 2000.0, CONF_ALGORITHM: "human_brightness"},
    )
    migrate(entry)
    config = merged(entry)
    assert config[CONF_ALGORITHM] == "human_brightness"
    assert CONF_IR_ALGORITHM not in config


def test_abandoned_prerelease_ir_keys_are_dropped():
    entry = make_entry(
        1,
        {
            V1_FULL_SCALE: 500.0,
            "ir_enabled": True,
            "ir_calibration_factor": 20.0,
            "ir_exponent": 2.0,
            "ir_algorithm": "ir_ambient",
            "ir_unavailable_above": 40.0,
        },
        {"ir_full_scale_lux": 33.0},
    )
    migrate(entry)
    config = merged(entry)
    for key in (
        "ir_calibration_factor",
        "ir_full_scale_lux",
        "ir_exponent",
        "ir_unavailable_above",
    ):
        assert key not in config
    assert config["ir_enabled"] is True
    assert config[CONF_IR_ALGORITHM] == LEGACY_ALGORITHM


def test_ir_unavailable_bounds_are_dropped():
    entry = make_entry(1, {V1_FULL_SCALE: 500.0, "ir_unavailable_above": 40.0})
    migrate(entry)
    assert "ir_unavailable_above" not in merged(entry)


def test_current_version_is_a_noop():
    entry = make_entry(CONFIG_ENTRY_VERSION, {FULL_SCALE: 1000})
    before = merged(entry)
    assert migrate(entry) is True
    assert merged(entry) == before


def test_future_version_is_refused_without_mutation():
    entry = make_entry(CONFIG_ENTRY_VERSION + 1, {FULL_SCALE: 1000})
    assert migrate(entry) is False
    assert entry.version == CONFIG_ENTRY_VERSION + 1


def test_migration_is_idempotent():
    entry = make_entry(1, {"ir_enabled": True, V1_FULL_SCALE: 500.0})
    migrate(entry)
    first = merged(entry)
    assert migrate(entry) is True
    assert merged(entry) == first
