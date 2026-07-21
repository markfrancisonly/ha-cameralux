"""Imaging + coercion helpers. Needs only pytest, Pillow and numpy.

Scope: the non-obvious numerical invariants that break silently on a refactor.
Trivial pass-throughs are intentionally not tested.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from custom_components.cameralux.const import DEFAULT_IR_CHROMA_THRESHOLD
from custom_components.cameralux.helpers import (
    INVERSE_GAMMA_LUT,
    calculate_image_luminance,
    calculate_ir_image_luminance,
    coerce_float,
    compute_chroma,
    compute_gray_fraction,
    crop_image,
    float_if_value,
    process_image,
    resize_image,
)


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# --- coercion: only the branches other modules depend on ------------------


def test_coerce_float_none_path_is_used_by_the_sensor():
    assert coerce_float(None, 7.0) == 7.0  # sensor resolves absent IR values here


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("bad", None), ("2.5", 2.5), (0, 0.0)],
)
def test_float_if_value_distinguishes_blank_from_zero(value, expected):
    assert float_if_value(value) == expected


# --- gamma: the whole reason the LUT exists -------------------------------


def test_inverse_gamma_is_linear_light_not_gamma_encoded():
    assert INVERSE_GAMMA_LUT[0] == pytest.approx(0.0)
    assert INVERSE_GAMMA_LUT[255] == pytest.approx(1.0, rel=1e-6)
    # sRGB mid-grey linearizes to ~0.216, NOT 0.5 -- guards the LUT wiring
    assert INVERSE_GAMMA_LUT[128] == pytest.approx(0.2158, abs=1e-3)


def test_luminance_uses_rec709_weights():
    assert calculate_image_luminance(
        Image.new("RGB", (8, 8), (0, 255, 0))
    ) == pytest.approx(0.7152, abs=1e-3)


# --- luminance across the modes a camera can actually deliver -------------


@pytest.mark.parametrize("mode", ["RGB", "L", "P"])
def test_luminance_white_is_one(mode):
    white = Image.new("RGB", (8, 8), (255, 255, 255)).convert(mode)
    assert calculate_image_luminance(white) == pytest.approx(1.0, rel=1e-2)


def test_luminance_black_is_zero():
    assert calculate_image_luminance(
        Image.new("RGB", (8, 8), (0, 0, 0))
    ) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("mode", ["I;16", "I;16B"])
def test_luminance_handles_big_endian_16bit(mode):
    """Big-endian 16-bit must hit the 16-bit branch, not clip through RGB."""
    assert calculate_image_luminance(Image.new(mode, (4, 4), 65535)) == pytest.approx(
        1.0, rel=1e-3
    )


def test_luminance_never_exceeds_full_scale_ceiling():
    """'I'/'F' can hold values above full scale; max_lux relies on avg <= 1."""
    assert calculate_image_luminance(Image.new("I", (4, 4), 2_000_000)) <= 1.0
    assert calculate_image_luminance(Image.new("F", (4, 4), 5.0)) <= 1.0


# --- chroma: the IR detector, calibrated against a real threshold ---------


def test_chroma_zero_for_neutral_large_for_colour():
    assert compute_chroma(Image.new("RGB", (8, 8), (128, 128, 128))) == pytest.approx(
        0.0, abs=1e-6
    )
    assert compute_chroma(Image.new("RGB", (8, 8), (255, 0, 0))) > 20.0


def test_chroma_expands_palette_before_measuring():
    """A palette image carries colour; measuring it raw would report 0."""
    assert compute_chroma(Image.new("RGB", (8, 8), (255, 0, 0)).convert("P")) > 20.0


def test_chroma_straddles_the_default_ir_threshold():
    monochrome = compute_chroma(Image.new("RGB", (8, 8), (90, 90, 90)))
    colour = compute_chroma(Image.new("RGB", (8, 8), (200, 90, 40)))
    assert monochrome < DEFAULT_IR_CHROMA_THRESHOLD < colour


# --- crop / resize: the clamping rules -----------------------------------


def test_crop_clamps_and_treats_zero_as_full_extent():
    image = Image.new("RGB", (100, 100), "white")
    assert crop_image(image, {"x": 90, "y": 90, "width": 20, "height": 20}).size == (
        10,
        10,
    )  # overrun clamped
    assert crop_image(image, {"x": 0, "y": 0, "width": 0, "height": 0}).size == (
        100,
        100,
    )  # 0 = full


def test_crop_origin_outside_frame_returns_full_image():
    image = Image.new("RGB", (100, 100), "white")
    assert crop_image(image, {"x": 200, "y": 0, "width": 10, "height": 10}).size == (
        100,
        100,
    )


def test_resize_only_shrinks_and_preserves_aspect():
    assert resize_image(Image.new("RGB", (1000, 1000), "white"), 10_000).size == (
        100,
        100,
    )
    small = Image.new("RGB", (10, 10), "white")
    assert resize_image(small, 10_000) is small  # untouched


# --- process_image: the executor entry point + documented contract --------


def test_process_image_pure_white_is_full_scale_for_any_exponent():
    avg, chroma, ir_frame, effective_roi = process_image(
        _png(Image.new("RGB", (20, 20), "white")), None, 250_000
    )
    assert avg == pytest.approx(1.0, rel=1e-3)
    assert chroma == pytest.approx(0.0, abs=1e-6)
    assert ir_frame is False  # no threshold given -> color path
    assert effective_roi is None
    # max_lux invariant: 1.0 ** exponent == 1.0
    for exponent in (0.1, 0.5, 2.0, 4.0):
        assert avg**exponent == pytest.approx(1.0, rel=1e-3)


def test_process_image_roi_scopes_luminance_but_chroma_uses_full_frame():
    frame = Image.new("RGB", (40, 20), (255, 0, 0))  # right half red
    frame.paste(Image.new("RGB", (20, 20), (255, 255, 255)), (0, 0))  # left half white
    avg, chroma, _, effective_roi = process_image(
        _png(frame), {"x": 10, "y": 0, "width": 20, "height": 40}, 250_000
    )
    assert 0.0 < avg < 1.0  # ROI spans both the white and red halves
    assert chroma > 10.0  # chroma still saw the red half
    assert effective_roi == {"x": 10, "y": 0, "width": 20, "height": 20}


def test_process_image_reports_full_frame_when_roi_origin_is_invalid():
    _, _, _, effective_roi = process_image(
        _png(Image.new("RGB", (40, 20), "white")),
        {"x": 100, "y": 0, "width": 10, "height": 10},
        250_000,
    )
    assert effective_roi == {"x": 0, "y": 0, "width": 40, "height": 20}


def test_process_image_dispatches_ir_method_below_threshold():
    grey = _png(Image.new("RGB", (20, 20), (90, 90, 90)))  # chroma ~0, all gray
    _, _, ir, _ = process_image(grey, None, 250_000, ir_chroma_threshold=6.0)
    assert ir is True  # chroma below threshold -> IR method
    _, _, ir, _ = process_image(grey, None, 250_000, ir_chroma_threshold=None)
    assert ir is False  # profile disabled -> color method
    colour = _png(Image.new("RGB", (20, 20), (200, 90, 40)))
    _, _, ir, _ = process_image(colour, None, 250_000, ir_chroma_threshold=6.0)
    assert ir is False  # colour frame stays on the color method


def _noisy_dark_colour_frame() -> Image.Image:
    """Dark COLOR frame: mean chroma is low, but pixels are not gray —
    the signature of a no-IR camera (tablet) in a dark room."""
    img = Image.new("RGB", (20, 20))
    px = [(10, 14, 8), (13, 9, 12), (8, 12, 15), (14, 10, 9)]
    img.putdata([px[i % 4] for i in range(400)])
    return img


def test_gray_guard_rejects_dark_noisy_colour_frames():
    frame = _noisy_dark_colour_frame()
    assert compute_chroma(frame) < 6.0  # mean chroma alone would say IR
    assert compute_gray_fraction(frame) < 0.85  # but pixels are not gray
    _, _, ir, _ = process_image(_png(frame), None, 250_000, ir_chroma_threshold=6.0)
    assert ir is False  # guard keeps the color method


def test_ir_luminance_trims_hotspots():
    """96% dark + 4% white hotspot: the IR statistic drops the hotspot."""
    img = Image.new("RGB", (20, 20), (0, 0, 0))
    img.paste(Image.new("RGB", (4, 4), (255, 255, 255)), (0, 0))  # 16/400 px
    assert calculate_image_luminance(img) == pytest.approx(0.04, abs=0.005)
    assert calculate_ir_image_luminance(img) == pytest.approx(0.0, abs=1e-4)
    # the color path trims too when asked (highlight-resistant algorithms)
    assert calculate_image_luminance(img, trim_top_fraction=0.05) == pytest.approx(
        0.0, abs=1e-4
    )
    # IR statistic dispatch: median floors the hotspot, mean keeps it
    assert calculate_ir_image_luminance(img, "median") == pytest.approx(0.0, abs=1e-4)
    assert calculate_ir_image_luminance(img, "mean") == pytest.approx(0.04, abs=0.005)


def test_process_image_rejects_garbage_bytes():
    with pytest.raises(OSError):
        process_image(b"not an image", None, 250_000)
