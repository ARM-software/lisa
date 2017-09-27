Instrumentation
===============

The ``Instrument`` API provide a consistent way of collecting measurements from
a target. Measurements are collected via an instance of a class derived from
:class:`Instrument`. An ``Instrument`` allows collection of measurement from one
or more channels. An ``Instrument`` may support ``INSTANTANEOUS`` or
``CONTINUOUS`` collection, or both.

Example
-------

The following example shows how to use an instrument to read temperature from an
Android target.

.. code-block:: ipython

    # import and instantiate the Target and the instrument
    # (note: this assumes exactly one android target connected
    #  to the host machine).
    In [1]: from devlib import AndroidTarget, HwmonInstrument

    In [2]: t = AndroidTarget()

    In [3]: i = HwmonInstrument(t)

    # Set up the instrument on the Target. In case of HWMON, this is
    # a no-op, but is included here for completeness.
    In [4]: i.setup()

    # Find out what the instrument is capable collecting from the
    # target.
    In [5]: i.list_channels()
    Out[5]:
    [CHAN(battery/temp1, battery_temperature),
     CHAN(exynos-therm/temp1, exynos-therm_temperature)]

    # Set up a new measurement session, and specify what is to be
    # collected.
    In [6]: i.reset(sites=['exynos-therm'])

    # HWMON instrument supports INSTANTANEOUS collection, so invoking
    # take_measurement() will return a list of measurements take from
    # each of the channels configured during reset()
    In [7]: i.take_measurement()
    Out[7]: [exynos-therm_temperature: 36.0 degrees]

API
---

Instrument
~~~~~~~~~~

.. class:: Instrument(target, **kwargs)

   An ``Instrument`` allows collection of measurement from one or more
   channels. An ``Instrument`` may support ``INSTANTANEOUS`` or ``CONTINUOUS``
   collection, or both.

.. attribute:: Instrument.mode

   A bit mask that indicates collection modes that are supported by this
   instrument. Possible values are:

   :INSTANTANEOUS: The instrument supports taking a single sample via
                   ``take_measurement()``.
   :CONTINUOUS: The instrument supports collecting measurements over a
                period of time via ``start()``, ``stop()``, ``get_data()``,
		and (optionally) ``get_raw`` methods.

   .. note:: It's possible for one instrument to support more than a single
             mode.

.. attribute:: Instrument.active_channels

   Channels that have been activated via ``reset()``. Measurements will only be
   collected for these channels.

.. method:: Instrument.list_channels()

   Returns a list of :class:`InstrumentChannel` instances that describe what
   this instrument can measure on the current target. A channel is a combination
   of a ``kind`` of measurement (power, temperature, etc) and a ``site`` that
   indicates where on the target the measurement will be collected from.

.. method:: Instrument.get_channels(measure)

   Returns channels for a particular ``measure`` type. A ``measure`` can be
   either a string (e.g. ``"power"``) or a :class:`MeasurmentType` instance.

.. method::  Instrument.setup(*args, **kwargs)

   This will set up the instrument on the target. Parameters this method takes
   are particular to subclasses (see documentation for specific instruments
   below).  What actions are performed by this method are also
   instrument-specific.  Usually these will be things like  installing
   executables, starting services, deploying assets, etc. Typically, this method
   needs to be invoked at most once per reboot of the target (unless
   ``teardown()`` has been called), but see documentation for the instrument
   you're interested in.

.. method:: Instrument.reset(sites=None, kinds=None, channels=None)

   This is used to configure an instrument for collection. This must be invoked
   before ``start()`` is called to begin collection. This methods sets the
   ``active_channels`` attribute of the ``Instrument``.

   If ``channels`` is provided, it is a list of names of channels to enable and
   ``sites`` and ``kinds`` must both be ``None``.

   Otherwise, if one of ``sites`` or ``kinds`` is provided, all channels
   matching the given sites or kinds are enabled. If both are provided then all
   channels of the given kinds at the given sites are enabled.

   If none of ``sites``, ``kinds`` or ``channels`` are provided then all
   available channels are enabled.

.. method:: Instrument.take_measurment()

   Take a single measurement from ``active_channels``. Returns a list of
   :class:`Measurement` objects (one for each active channel).

   .. note:: This method is only implemented by :class:`Instrument`\ s that
             support ``INSTANTANEOUS`` measurement.

.. method:: Instrument.start()

   Starts collecting measurements from ``active_channels``.

   .. note:: This method is only implemented by :class:`Instrument`\ s that
             support ``CONTINUOUS`` measurement.

.. method:: Instrument.stop()

   Stops collecting measurements from ``active_channels``. Must be called after
   :func:`start()`.

   .. note:: This method is only implemented by :class:`Instrument`\ s that
             support ``CONTINUOUS`` measurement.

.. method:: Instrument.get_data(outfile)

   Write collected data into ``outfile``. Must be called after :func:`stop()`.
   Data will be written in CSV format with a column for each channel and a row
   for each sample. Column heading will be channel, labels in the form
   ``<site>_<kind>`` (see :class:`InstrumentChannel`). The order of the columns
   will be the same as the order of channels in ``Instrument.active_channels``.

   If reporting timestamps, one channel must have a ``site`` named ``"timestamp"``
   and a ``kind`` of a :class:`MeasurmentType` of an appropriate time unit which will
   be used, if appropriate, during any post processing.

   .. note:: Currently supported time units are seconds, milliseconds and
             microseconds, other units can also be used if an appropriate
             conversion is provided.

   This returns a :class:`MeasurementCsv` instance associated with the outfile
   that can be used to stream :class:`Measurement`\ s lists (similar to what is
   returned by ``take_measurement()``.

   .. note:: This method is only implemented by :class:`Instrument`\ s that
             support ``CONTINUOUS`` measurement.

.. method:: Instrument.get_raw()

   Returns a list of paths to files containing raw output from the underlying
   source(s) that is used to produce the data CSV. If now raw output is
   generated or saved, an empty list will be returned. The format of the
   contents of the raw files is entirely source-dependent.

.. attribute:: Instrument.sample_rate_hz

   Sample rate of the instrument in Hz. Assumed to be the same for all channels.

   .. note:: This attribute is only provided by :class:`Instrument`\ s that
             support ``CONTINUOUS`` measurement.

Instrument Channel
~~~~~~~~~~~~~~~~~~

.. class:: InstrumentChannel(name, site, measurement_type, **attrs)

   An :class:`InstrumentChannel` describes a single type of measurement that may
   be collected by an :class:`Instrument`. A channel is primarily defined by a
   ``site`` and a ``measurement_type``.

   A ``site`` indicates where  on the target a measurement is collected from
   (e.g. a voltage rail or location of a sensor).

   A ``measurement_type`` is an instance of :class:`MeasurmentType` that
   describes what sort of measurement this is (power, temperature, etc). Each
   measurement type has a standard unit it is reported in, regardless of an
   instrument used to collect it.

   A channel (i.e. site/measurement_type combination) is unique per instrument,
   however there may be more than one channel associated with one site (e.g. for
   both voltage and power).

   It should not be assumed that any site/measurement_type combination is valid.
   The list of available channels can queried with
   :func:`Instrument.list_channels()`.

.. attribute:: InstrumentChannel.site

   The name of the "site" from which the measurements are collected (e.g. voltage
   rail, sensor, etc).

.. attribute:: InstrumentChannel.kind

   A string indicating the type of measurement that will be collected. This is
   the ``name`` of the :class:`MeasurmentType` associated with this channel.

.. attribute:: InstrumentChannel.units

   Units in which measurement will be reported. this is determined by the
   underlying :class:`MeasurmentType`.

.. attribute:: InstrumentChannel.label

   A label that can be attached to measurements associated with with channel.
   This is constructed with ::

       '{}_{}'.format(self.site, self.kind)


Measurement Types
~~~~~~~~~~~~~~~~~

In order to make instruments easer to use, and to make it easier to swap them
out when necessary (e.g. change method of collecting power), a number of
standard measurement types are defined. This way, for example, power will always
be reported as "power" in Watts, and never as "pwr" in milliWatts. Currently
defined measurement types are


+-------------+-------------+---------------+
| name        | units       | category      |
+=============+=============+===============+
| count       | count       |               |
+-------------+-------------+---------------+
| percent     | percent     |               |
+-------------+-------------+---------------+
| time_us     | microseconds|  time         |
+-------------+-------------+---------------+
| time_ms     | milliseconds|  time         |
+-------------+-------------+---------------+
| temperature | degrees     |  thermal      |
+-------------+-------------+---------------+
| power       | watts       | power/energy  |
+-------------+-------------+---------------+
| voltage     | volts       | power/energy  |
+-------------+-------------+---------------+
| current     | amps        | power/energy  |
+-------------+-------------+---------------+
| energy      | joules      | power/energy  |
+-------------+-------------+---------------+
| tx          | bytes       | data transfer |
+-------------+-------------+---------------+
| rx          | bytes       | data transfer |
+-------------+-------------+---------------+
| tx/rx       | bytes       | data transfer |
+-------------+-------------+---------------+


.. instruments:

Available Instruments
---------------------

This section lists instruments that are currently part of devlib.

TODO
