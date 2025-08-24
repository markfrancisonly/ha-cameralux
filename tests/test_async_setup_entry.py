import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys

# Ensure the custom component is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from custom_components.cameralux.const import CONF_ENTITY_ID
from custom_components.cameralux.sensor import async_setup_entry


class FakeConfigEntry(SimpleNamespace):
    pass


def test_async_setup_entry_with_camera_only():
    """Ensure async_setup_entry works when only a camera entity is provided."""
    entry = FakeConfigEntry(data={CONF_ENTITY_ID: "camera.test"}, options={}, entry_id="1")
    add_entities = MagicMock()

    # Should not raise and should call async_add_entities once
    asyncio.run(async_setup_entry(None, entry, add_entities))
    add_entities.assert_called_once()
