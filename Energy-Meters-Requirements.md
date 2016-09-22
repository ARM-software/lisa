LISA provide integration for some types of energy probes.

Currently supported instruments are:
* [`HWMON`](https://github.com/ARM-software/lisa/wiki/Energy-Meters-Requirements#linux-hwmon): Linux kernel hardware monitoring
* [`AEP`](https://github.com/ARM-software/lisa/wiki/Energy-Meters-Requirements#arm-energy-probe-aep): ARM Energy Probe
* [`iiocapture`](https://github.com/ARM-software/lisa/wiki/Energy-Meters-Requirements#iiocapture---baylibre-acme-cape): BleagleBone Black + BayLibre ACME Cape

Instruments need to be specified either in `target.config` or in aconfiguration dictionary to be passed to `TestEnv` when creating the test environment object.

## Using Energy Meters

In general, energy meters provide the following methods:

* `reset()` - reset the energy meters
* `report()` - get total energy consumption since last reset
* `sample()` (optional) - get a sample from the energy meters

Assuming you declare a target configuration dictionary called `my_conf`, the following code snippet shows how to use the `EnergyMeter` class to measure energy consumption of the target while running a workload.

```python
my_conf = {
    ...
    "emeter" : {
        # Energy meter configuration. You need to specify an instrument here.
        "instrument" : "instrument_name",
        "conf" : {
            ...
        }
        "channel_map" : {
            ...
        }
    },
    ...
}

te = TestEnv(target_conf=my_conf)

# Reset energy meter
te.emeter.reset()
# Start workload
wload.run()
# Stop energy measurement and report results in a specific output directory
nrg_data, nrg_file = te.emeter.report(te.res_dir)
```

The `report()` method returns a tuple containing an energy data dictionary (`nrg_data`) with the following format:

```python
nrg_data = {
    "channel_1" : {
        # Collected data for channel_1
    },
    ...
    "channel_n" : {
        # Collected data for channel_n
    }
}
```

and the name of the output file where collected energy data is stored (`nrg_file`).

***

## Linux HWMON

The `hwmon` is a generic Linux kernel subsystem, providing access to hardware monitoring components like temperature or voltage/current sensors.

#### LISA Target Configuration

Energy sampling with `hwmon` requires the HWMON module to be enabled in `devlib` be specifying it in the target configuration.

```python
target_conf = {
    # Enable hwmon module in devlib
    "modules" = ['hwmon'],

    "emeter" = {
        "instrument" : "hwmon",
        "conf" : {
            # Prefixes of the HWMon labels
            'sites' : ['a53', 'a57'],
            # Type of hardware monitor to be used
            'kinds' : ['energy']
        },
        # Mapping between sites and user-defined channel names
        "channel_map" : {
            'little' : 'a53',
            'big' : 'a57'
        }
    },
}
```

HWMON are the default energy meter in LISA. Therefore, enabling them as a `devlib` module is enough to get them as energy meters.

***

## ARM Energy Probe (AEP)

ARM Energy Probes are lightweight power measurement tools for software developers. They can monitor up to three voltage rails simultaneously.

#### Equipment

The required equipment is the following:

* An ARM Energy Probe
* A shunt resistor to be connected between the voltage rail and the probe. The voltage drop
  on the resistor must be at most 165 mv. Therefore depending on the maximum current required
  by the load, one can properly select the value of the shunt resistor
* Install `caiman` required libraries:

```bash
sudo apt-get instal libudev
```

```bash
git clone https://github.com/ARM-software/caiman.git
cd caiman &&  && cd -
```
tool ([installation instructions](https://github.com/ARM-software/caiman))

![ARM Energy Probe](https://developer.arm.com/-/media/developer/products/software-tools/ds-5-development-studio/images/ARM%20Energy%20Probe/ARM_Energy_Probe_4.png?h=378&w=416&hash=90D98087E80D9178CCC28026C1C8E476A6736D09&hash=90D98087E80D9178CCC28026C1C8E476A6736D09&la=en)

#### LISA Target Configuration

```python
target_conf = {
     "emeter" : {
         "instrument" : "aep",
         "conf" : {
             # Value of the shunt resistor in Ohm
             'resistor_values' : [0.099],
             # Device entry assigned to the probe on the host
             'device_entry'    : '/dev/ttyACM0',
         },
         "channel_map" : {
             # <User-defined channel name> : <channel label>
             "BAT" : "BAT"
         }
     },
}
```
For this instrument, `channel_map` keys and values are usually set to the same value. In fact, it would be enough to use a list of strings for specifying channel names, but to make the `EnergyMeter` API more generic, it is better to be consistent and use a dictionary also in this case. 

***

## iiocapture - Baylibre ACME Cape

The `iiocapture` instrument exploits the [BayLibre ACME](http://baylibre.com/acme/) solution for measuring power.

#### Equipment
To use this instrument you need the following equipment:

* A [BeagleBone Black](https://beagleboard.org/black)
* An [ACME Cape](http://sigrok.org/wiki/BayLibre_ACME)
* Power probes for the ACME Cape
* The `iio-capture` tool installed in your host ([installation instructions here](https://github.com/BayLibre/iio-capture))

#### Build the software suite

Next step is to get an *IIO version* of the ACME BeagleBone black image. The recommended way of using ACME is to use the pre-built image provided by BayLibre:

* ACME Image (this is a beta release at the moment): https://github.com/baylibre-acme/ACME/releases/tag/b0

#### LISA Target Configuration

The target configuration for this instrument is:

```python
target_conf = {
    "emeter" : {
        "instrument" : "iiocapture",
        "conf" : {
            # Absolute path to the iio-capture binary on the host
            'iio-capture' : '<PATH_TO_iio-capture>/iio-capture',
            # Default host name of the BeagleBone Black
            'ip_address'     : 'baylibre-acme.local',
        },
        "channel_map" : {
            "Device0" : 0, # iio:device0
            ...
            "DeviceN" : N, # iio:deviceN
        }
    },
}
```

The ACME Cape 8 probe slots numbered 1 to 8. `iio:device<n>` is the n-th discovered probe and they are discovered in ascending order. For example, if you have 2 probes attached to PROBE2 and PROBE7, then PROBE2 will be `iio:device0` and PROBE7 will be `iio:device1`


You can also verify that the probes are correctly detected by the `iio daemon` running on the BeagleBone by running `iio_info` which is part of the `iio-capture` infrastructure:

```bash
$ iio_info -n IP_ADDRESS
```