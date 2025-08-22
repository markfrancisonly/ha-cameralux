import math
from pathlib import Path
import sys

import pytest
from PIL import Image

# Ensure the custom component is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from custom_components.cameralux.helpers import (  # noqa: E402
    calculate_image_luminance,
    coerce_float,
    crop_image,
    get_bounded_int,
    resize_image,
)


def test_get_bounded_int():
    assert get_bounded_int("5", default=0, min_value=0, max_value=10) == 5
    assert get_bounded_int("bad", default=3) == 3
    assert get_bounded_int(100, default=0, max_value=50) == 50
    assert get_bounded_int(-5, default=0, min_value=-3) == -3


def test_coerce_float():
    assert coerce_float("2.5", 0.0) == 2.5
    assert coerce_float("oops", 1.2) == 1.2


def test_crop_and_resize_image():
    image = Image.new("RGB", (100, 100), color="white")
    roi = {"x": 10, "y": 10, "width": 50, "height": 50}
    cropped = crop_image(image, roi)
    assert cropped.size == (50, 50)

    roi_out = {"x": 90, "y": 90, "width": 20, "height": 20}
    cropped2 = crop_image(image, roi_out)
    assert cropped2.size == (10, 10)

    big = Image.new("RGB", (1000, 1000), color="white")
    resized = resize_image(big, max_pixels=10_000)
    assert resized.size == (100, 100)


def test_calculate_image_luminance():
    white = Image.new("RGB", (10, 10), color=(255, 255, 255))
    avg, perceived = calculate_image_luminance(white)
    assert avg == pytest.approx(1.0, rel=1e-3)
    assert perceived == pytest.approx(math.log1p(1.0), rel=1e-3)

    black = Image.new("RGB", (10, 10), color=(0, 0, 0))
    avg2, perceived2 = calculate_image_luminance(black)
    assert avg2 == pytest.approx(0.0, abs=1e-6)
    assert perceived2 == pytest.approx(0.0, abs=1e-6)
