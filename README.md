[![GitHub Release](https://img.shields.io/github/release/markfrancisonly/ha-cameralux.svg?style=flat-square)](https://github.com/markfrancisonly/ha-cameralux/releases)
[![License](https://img.shields.io/github/license/markfrancisonly/ha-cameralux.svg?style=flat-square)](https://github.com/markfrancisonly/ha-cameralux/blob/master/LICENSE)
[![hacs](https://img.shields.io/badge/HACS-default-orange.svg?style=flat-square)](https://hacs.xyz)

# Camera Lux

**Camera Lux** is a custom Home Assistant integration that utilizes a camera feed or direct image URL to estimate ambient light levels (lux) in your environment. By analyzing images from your cameras, `cameralux` provides virtual lux readings that can be used to automate lighting, monitor lighting conditions, and enhance your smart home setup.



<img width="600"  alt="image" src="https://github.com/user-attachments/assets/5832f3c8-ce1b-46d8-b71a-c6e8137ada0b" />


## Features

- **Real-Time Lux Monitoring:** Continuously estimate ambient light levels using your existing cameras.
- **Configurable Update Intervals:** Customize how frequently each sensor updates its lux readings.
- **Region of Interest (ROI):** Focus on specific areas within the camera's field of view for more accurate measurements.
- **Named Algorithms:** Pick how brightness becomes lux by camera type — no power-law tuning required.
- **Full-Scale Lux:** One calibration knob that raises or lowers all readings.
- **IR Detection:** A camera in IR mode measures its own illuminator, not ambient light — enable detection (per-camera chroma threshold + per-pixel gray verification) and declare what to report when an IR frame is seen: a measurement with a chosen curve, a fixed lux, or unavailable.
- **Unavailable Lux Thresholds:** Optionally mark the sensor unavailable when lux falls below or rises above configured thresholds — go offline rather than publish a reading outside the trustworthy envelope.
- **Frozen-Feed Detection (opt-in):** When enabled, each cycle fetches the frame twice, a moment apart, and compares bytes. A live camera never produces identical bytes twice (sensor noise); only a stuck or substituted frame — an NVR's "camera offline" placeholder, a wedged stream — repeats exactly. Such frames are never measured: the sensor goes unavailable until real frames return, so downstream automations hold instead of reacting to a picture nobody took. Leave it off for sources that legitimately serve unchanged bytes between updates (infrequently refreshed snapshot URLs), where identical pairs are normal.
- **Compatibility:** Works with any camera integrated into Home Assistant, supporting both camera entities and direct image URLs.

## Configuration Options via UI

- **`entity_id`**: The camera entity from which to fetch images.
- **`image_url`** _(optional)_: Direct HTTP URL pointing to an image file (e.g., JPEG, PNG).
- **`full_scale_lux`** _(optional)_: The lux reported for a pure white image. Because the measured luminance is normalized to `[0, 1]`, this is simultaneously the calibration knob and the sensor's maximum reading — published as the `max_lux` attribute. Defaults to 1000.
- **`algorithm`** _(optional)_: How measured brightness becomes lux — see [Choosing an algorithm](#choosing-an-algorithm). Defaults to Human brightness.
- **`update_interval`** _(optional)_: Seconds between updates. Default is 30 seconds.
- **`unavailable_below`** _(optional)_: If the calculated lux is below this value, the sensor is marked unavailable until it rises.
- **`unavailable_above`** _(optional)_: If the calculated lux is above this value, the sensor is marked unavailable until it falls.
- **`ir_enabled`** _(optional)_: Enable IR detection. Off by default — leave off for cameras with no IR mode (tablets, wall panels, fixed-exposure).
- **`ir_chroma_threshold`** _(optional)_: Frames below this color level count as IR. Cameras flip to IR at different chroma levels, so tune per camera (IR/monochrome ≈ 0–4, color scenes 7+; default 6).
- **`ir_algorithm`** _(optional)_: What to do when an IR frame is detected — see [IR mode](#ir-mode). Defaults to Fixed brightness.
- **`ir_fixed_lux`** _(optional)_: The lux reported by Fixed brightness. Defaults to 0.
- **`roi_enabled`** _(optional)_: Defines a rectangular region within the image for focused brightness assessment.
  - **`x`**: The x-coordinate of the top-left corner of the ROI.
  - **`y`**: The y-coordinate of the top-left corner of the ROI.
  - **`width`**: The width of the ROI.
  - **`height`**: The height of the ROI.

**Note:** Each sensor must have either `entity_id` or `image_url` configured, but not both.

## IR mode

When a camera switches to infrared mode, it stops measuring ambient light:
the image shows the camera's **own IR illuminator** reflecting off the scene,
shaped by surface distances and auto-gain. Enable IR detection and the IR
algorithm is the policy applied **after** a frame is detected as IR:

- **Fixed brightness** _(default)_ — report `ir_fixed_lux` (default 0). A
  declared policy value, not a measurement, so the unavailable thresholds
  don't judge it.
- **Human brightness / Human brightness, highlight-resistant / Linear** —
  measure the IR frame with that curve (sharing the one `full_scale_lux`),
  with the normal unavailable thresholds in force. For cameras whose IR
  image is still meaningful.
- **Unavailable** — take the sensor offline rather than report a number
  nobody measured.

A frame counts as IR only when its chroma falls below the per-camera
threshold **and** nearly every pixel is individually gray (a true monochrome
frame) — dark or muted color scenes do not trip it. The live `chroma`
attribute shows what the detector sees; `ir_detected` shows the verdict each
cycle.

## Choosing an algorithm

Pick by camera type — you don't need to understand the math:

- **Human brightness** _(default)_ — for regular cameras **with infrared night
  mode** (most security cameras). Readings track how bright the scene *looks*.
  Pair it with IR detection and the default Fixed policy: monochrome IR
  frames report the fixed value, so this algorithm only ever handles daylight
  and lit scenes, where it performs best.
- **Human brightness, highlight-resistant** — same response, but the brightest
  ~5% of pixels are ignored. Choose this when direct sun patches, windows, or a
  lamp in frame make readings jump while the room's overall light hasn't
  changed.
- **Linear** — for **fixed-exposure or no-IR cameras**: tablets, wall panels,
  and starlight/"color night vision" cameras that never switch to monochrome.
  These cameras see true darkness on the normal path, and linear is the only
  response that keeps dark readings honest (a dark room reads a few lux;
  compressive curves would report 40–140 lx). Leave IR detection off.

**Algorithms set the shape; full-scale sets the size.** No algorithm ever
changes your scale knob: `full_scale_lux` raises or lowers all measured
readings. Calibrate against a reference light level
(`new = current × target_lux / reported_lux`).

## State Attributes

Listed in the order they appear on the entity:

- **`chroma`**: Mean chroma deviation from neutral, used for IR detection.
- **`ir_detected`**: **Key attribute.** Present when IR detection is enabled; `true` when the frame reads as IR/monochrome, so the IR policy is in force.
- **`avg_luminance`**: Mean linear-light luminance of the sampled region, in `[0, 1]`.
- **`effective_roi`**: Present when ROI is enabled after the first successful frame. The actual source-image rectangle measured after defaults and boundary clamping; reports the full frame if an invalid ROI falls back to it.
- **`perceived_luminance`**: The perception transform of `avg_luminance`. Multiplied by `full_scale_lux` this gives the state. `null` while a fixed IR value or unavailable policy is in force.
- **`algorithm`**: The policy in force this cycle — the color curve (`human_brightness`, `human_brightness_hr`, `linear`) or, on an IR frame, the IR selection (a curve name, `fixed`, `unavailable`). Migrated 1.x entries report their compatibility curve until a current algorithm is selected in Options.
- **`perception_exponent`**: The exponent that policy applies (absent under the legacy log curve).
- **`max_lux`**: The reading a pure white image would produce.
- **`unavailable_above_lux`** / **`unavailable_below_lux`**: The configured thresholds. Shown only when set; never applied to a fixed IR value.

## Breaking Changes: Upgrading from 1.x

Existing 1.x entries retain their previous response curve automatically, so an
upgrade does not force an immediate recalibration. However, the legacy compatibility 
curve is *not* offered for new configuration or reinstallation. Open the entry's Options, 
choose one of the current algorithms, and save when you are ready to move to the current model.


## Installation

### HACS (Recommended)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=markfrancisonly&repository=ha-cameralux&category=integration)

Or manually add the custom repository:

<details>
<summary>Step-by-step HACS installation</summary>

1. Open **HACS** in your Home Assistant dashboard
2. Click the **⋮** menu (top right) → **Custom repositories**
3. Add this URL and set the category to **Integration**, then click **Add**:
   ```
   https://github.com/markfrancisonly/ha-cameralux
   ```
4. The repository now appears in the custom repositories list. Close the dialog.
5. Back in HACS, search for **Video Call** and open the result
6. Click **Download** (or **Install**) and confirm
7. **Restart Home Assistant**

</details>

### Manual Installation

1. Download the [latest release](https://github.com/markfrancisonly/ha-cameralux/releases)
2. Copy the contents into `custom_components/ha-cameralux/` inside your HA config directory
3. Restart Home Assistant
