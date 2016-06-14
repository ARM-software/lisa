LISA provide integration for some types of energy probes.

Currently supported instruments are:
* `HWMON`: Linux kernel hardware monitoring
* `AEP`: ARM Energy Probe
* `iiocapture`: BayLibre ACME Cape

Instruments need to be specified either in `target.config` or in a configuration dictionary to be passed to `TestEnv` when creating the test environment object.

## Linux HWMON

## ARM Energy Probe (AEP)

## iiocapture

The `iiocapture` instrument exploits the [BayLibre ACME](http://baylibre.com/acme/) solution for measuring power.

### Equipment
To use this instrument you need the following equipment:

* A [BeagleBone Black](https://beagleboard.org/black)
* An [ACME Cape](http://sigrok.org/wiki/BayLibre_ACME)
* Power probes for the ACME Cape
* The `iio-capture` tool installed in your host ([installation instructions here](https://github.com/BayLibre/iio-capture))

### Build the software suite

Next step is to build the ACME software suite by following the instructions on [Building the software with iio](http://wiki.baylibre.com/doku.php?id=acme:start#building_the_software_with_iio).

### LISA Target Configuration

The target configuration for this instrument is:

```python
my_conf = {
    "emeter" : {
        "instrument" : "iiocapture",
        "conf" : {
            # List of iio devices (i.e. probes) attached to the ACME Cape
            # The ACME Cape 8 probe slots numbered 1 to 8. iio:device<n> is
            # the n-th discovered probe and they are discovered in ascending order.
            # For example, if you have 2 probes attached to PROBE2 and PROBE7, then
            # PROBE2 will be iio:device0 and PROBE7 will be iio:device1
            'iiodevice'     : ['iio:device0'],
            # Absolute path to the iio-capture binary on the host
            'iiocapturebin' : '<PATH_TO_iio-capture>/iio-capture',
            # IP of the BeagleBone Black
            'device_ip'     : '192.168.7.2',
        }
    },
}
```

You can also verify that the probes are correctly detected by the `iio daemon` running on the BeagleBone by running `iio_info` which is part of the `iio-capture`:

```bash
$ iio_info -n <DEVICE_IP>
```