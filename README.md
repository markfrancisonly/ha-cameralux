
# Camera Lux
cameralux is a Home Assistant sensor custom component that creates emulate a lux sensors that can be used to determine ambient light level from a camera images.  

The cameralux sensor returns a lux unit of measure between 0 and 255 derived from the raw average image brightness of a Home Assistant [camera](https://www.home-assistant.io/components/camera/).

## Example configuration.yaml

```yaml
sensor:
  - platform: cameralux
    sensors:
      Doorbell lux: camera.doorbell
      Family lux: camera.family
      Kitchen lux: camera.kitchen
```
The yaml above creates three light sensors `sensor.doorbell_lux`,`sensor.family_lux` and `sensor.kitchen_lux` using the config format `lux sensor friendly_name: camera.entity_id`

## Roadmap

Missing is a configuration method to focus on a specific area of the camera image and calibrate the relationship between camera image brightness and lux value output. Ideally the mathematical relationship between lux and image brightness would auto-calibrate using the combined affect of time, season, and sun position to denote ambient lighting markers in camera image brightness. 
