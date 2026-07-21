# Camera Lux

Camera Lux is a Home Assistant custom integration that estimates ambient light
from a camera entity or image URL.

Current configuration is entirely UI-driven and provides:

- Human-brightness, highlight-resistant, and linear measurement algorithms.
- One full-scale lux calibration control.
- Optional region-of-interest measurement.
- Optional per-camera IR detection with measured, fixed-lux, or unavailable
  policies.
- Optional lower and upper availability bounds.

For configuration guidance, algorithm selection, migration notes, state
attributes, and troubleshooting, see the
[complete Camera Lux documentation](https://github.com/markfrancisonly/ha-cameralux#readme).
