Derived Measurements
=====================


The ``DerivedMeasurements`` API provides a consistent way of performing post
processing on a provided :class:`MeasurementCsv` file.

Example
-------

The following example shows how to use an implementation of a
:class:`DerivedMeasurement` to obtain a list of calculated ``DerivedMetric``'s.

.. code-block:: ipython

    # Import the relevant derived measurement module
    # in this example the derived energy module is used.
    In [1]: from devlib import DerivedEnergyMeasurements

    # Obtain a MeasurementCsv file from an instrument or create from
    # existing .csv file. In this example an existing csv file is used which was
    # created with a sampling rate of 100Hz
    In [2]: from devlib import MeasurementsCsv
    In [3]: measurement_csv = MeasurementsCsv('/example/measurements.csv', sample_rate_hz=100)

    # Process the file and obtain a list of the derived measurements
    In [4]: derived_measurements = DerivedEnergyMeasurements.process(measurement_csv)

    In [5]: derived_measurements
    Out[5]: [device_energy: 239.1854075 joules, device_power: 5.5494089227 watts]

API
---

Derived Measurements
~~~~~~~~~~~~~~~~~~~~

.. class:: DerivedMeasurements()

   The ``DerivedMeasurements`` class is an abstract base for implementing
   additional classes to calculate various metrics.

.. method:: DerivedMeasurements.process(measurement_csv)

   Returns a list of :class:`DerivedMetric` objects that have been calculated.


Derived Metric
~~~~~~~~~~~~~~

.. class:: DerivedMetric

  Represents a metric derived from previously collected ``Measurement``s.
  Unlike, a ``Measurement``, this was not measured directly from the target.


.. attribute:: DerivedMetric.name

   The name of the derived metric. This uniquely defines a metric -- two
   ``DerivedMetric`` objects with the same ``name`` represent to instances of
   the same metric (e.g. computed from two different inputs).

.. attribute:: DerivedMetric.value

   The ``numeric`` value of the metric that has been computed for a particular
   input.

.. attribute:: DerivedMetric.measurement_type

   The ``MeasurementType`` of the metric. This indicates which conceptual
   category the metric falls into, its units, and conversions to other
   measurement types.

.. attribute:: DerivedMetric.units

   The units in which the metric's value is expressed.


Available Derived Measurements
-------------------------------
.. class:: DerivedEnergyMeasurements()

  The ``DerivedEnergyMeasurements`` class is used to calculate average power and
  cumulative energy for each site if the required data is present.

  The calculation of cumulative energy can occur in 3 ways. If a
  ``site`` contains ``energy`` results, the first and last measurements are extracted
  and the delta calculated. If not, a ``timestamp`` channel will be used to calculate
  the energy from the power channel, failing back to using the sample rate attribute
  of the :class:`MeasurementCsv` file if timestamps are not available. If neither
  timestamps or a sample rate are available then an error will be raised.


.. method:: DerivedEnergyMeasurements.process(measurement_csv)

  Returns a list of :class:`DerivedMetric` objects that have been calculated for
  the average power and cumulative energy for each site.


