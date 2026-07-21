# Camera Lux 1.x compatibility boundary

Everything specific to persisted Camera Lux 1.x entries lives in this
directory: frozen field names, the v1-to-v2 migration, the hidden legacy curve
identifier, and the compatibility sensor subclass.

Current config entries do not import this package during setup, and the current
config flow does not expose its algorithm.

When 1.x entries are no longer supported:

1. Delete this directory and `tests/legacy_v1/`.
2. Remove `async_migrate_entry` from the integration root.
3. Remove the conditional compatibility factory call from `sensor.py`.
4. Remove the 1.x upgrade note from the root README.
