# Camera Lux

**Camera Lux** is a custom Home Assistant integration that utilizes camera feeds to estimate ambient light levels (lux) in your environment. By analyzing images from your cameras, `cameralux` provides virtual lux readings that can be used to automate lighting, monitor lighting conditions, and enhance your smart home setup.

<img width="600"  alt="image" src="https://github.com/user-attachments/assets/51281257-7776-4a9e-a031-4f7a2e0d4e7a" />

## Features

- **Real-Time Lux Monitoring:** Continuously estimate ambient light levels using your existing cameras.
- **Configurable Update Intervals:** Customize how frequently each sensor updates its lux readings.
- **Region of Interest (ROI):** Focus on specific areas within the camera's field of view for more accurate measurements.
- **Calibration Factor:** Scale luminance to lux calculations to better align with real-world lighting conditions.
- **Compatibility:** Works with any camera integrated into Home Assistant, supporting both camera entities and direct image URLs.

## Configuration 

- **`entity_id`**: The camera entity from which to fetch images.
- **`image_url`** *(optional)*: Direct HTTP URL pointing to an image file (e.g., JPEG, PNG).
- **`update_interval`** *(optional)*: Seconds between updates. Default is 30 seconds.
- **`brightness_roi`** *(optional)*: Defines a rectangular region within the image for focused brightness assessment.
  - **`x`**: The x-coordinate of the top-left corner of the ROI.
  - **`y`**: The y-coordinate of the top-left corner of the ROI.
  - **`width`**: The width of the ROI.
  - **`height`**: The height of the ROI.
- **`calibration_factor`** *(optional)*: A float value to calibrate the perceived luminance to lux. Defaults to 2000 if not specified.

**Note:** Each sensor must have either `entity_id` or `image_url` configured, but not both.
 
## Installation

1. **Using HACS (Recommended):**
   - Open Home Assistant.
   - Navigate to **HACS > Integrations**.
   - Click on the **"+"** button.
   - Search for **"Camera Lux Sensor"** and install it.

2. **Manual Installation:**
   - Download the `cameralux` repository from [GitHub](https://github.com/markfrancisonly/ha-cameralux).
   - Place the `cameralux` folder inside your `custom_components` directory.
   - Restart Home Assistant.


