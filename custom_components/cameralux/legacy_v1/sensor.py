"""Camera Lux 1.x response curve, isolated for eventual removal.

The compatibility entity preserves ``lux = log1p(mean linear Y) * scale`` for
migrated entries. Current entries never use or offer this curve.
"""

from __future__ import annotations

import math
from typing import Any

from ..sensor import CameraLuxSensor
from . import LEGACY_ALGORITHM


class V1CameraLuxSensor(CameraLuxSensor):
    """Sensor retaining the Camera Lux 1.x transfer function."""

    def _resolve_curve(self, algorithm: str | None) -> tuple[str, float, float]:
        return LEGACY_ALGORITHM, 0.0, 1.0

    def _perceive(self, avg_luminance: float) -> float:
        return math.log1p(max(0.0, avg_luminance))

    def _curve_description(self) -> str:
        return "log1p transfer (1.x compatibility)"

    def _resolve_ir(self, ir_algorithm: str | None) -> tuple[str, float, float]:
        if ir_algorithm == LEGACY_ALGORITHM:
            return LEGACY_ALGORITHM, 0.0, 1.0
        return super()._resolve_ir(ir_algorithm)

    def _ir_perceive(self, avg_luminance: float) -> float:
        if self.ir_algorithm == LEGACY_ALGORITHM:
            return self._perceive(avg_luminance)
        return super()._ir_perceive(avg_luminance)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        attrs = dict(super().extra_state_attributes)
        legacy_curve_active = (
            self.ir_algorithm == LEGACY_ALGORITHM if self.ir_active else True
        )
        if legacy_curve_active:
            attrs.pop("perception_exponent", None)
        return attrs
