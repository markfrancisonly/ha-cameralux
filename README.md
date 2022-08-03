
# Camera Lux
cameralux is a Home Assistant sensor custom component that emulates an ambient light level lux sensor from a [camera](https://www.home-assistant.io/components/camera/). The cameralux sensor has a state between 0 and 255 that represents the perceived brightness of the associated camera image.

Configure via yaml using the config format `lux sensor friendly_name: camera.entity_id`

## Example configuration.yaml

```yaml
sensor:
  - platform: cameralux
    sensors:
      Doorbell lux: camera.doorbell
      Family lux: camera.family
      Kitchen lux: camera.kitchen
```

## Roadmap

Missing is a configuration method to focus on a specific area of the camera image and calibrate the relationship between camera image brightness and lux value output. Ideally the mathematical relationship between lux and image brightness would auto-calibrate using the combined affect of time, season, and sun position to denote ambient lighting markers in camera image brightness. 
