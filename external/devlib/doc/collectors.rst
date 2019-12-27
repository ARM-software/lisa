.. _collector:

Collectors
==========

The ``Collector`` API provide a consistent way of collecting arbitrary data from
a target. Data is collected via an instance of a class derived from
:class:`CollectorBase`.


Example
-------

The following example shows how to use a collector to read the logcat output
from an Android target.

.. code-block:: python

    # import and instantiate the Target and the collector
    # (note: this assumes exactly one android target connected
    #  to the host machine).
    In [1]: from devlib import AndroidTarget, LogcatCollector

    In [2]: t = AndroidTarget()

    # Set up the collector on the Target.

    In [3]: collector = LogcatCollector(t)

    # Configure the output file path for the collector to use.
    In [4]: collector.set_output('adb_log.txt')

    # Reset the Collector to preform any required configuration or preparation.
    In [5]: collector.reset()

    # Start Collecting
    In [6]: collector.start()

    # Wait for some output to be generated
    In [7]: sleep(10)

    # Stop Collecting
    In [8]: collector.stop()

    # Retrieved the collected data
    In [9]: output = collector.get_data()

    # Display the returned ``CollectorOutput`` Object.
    In [10]: output
    Out[10]: [<adb_log.txt (file)>]

    In [11] log_file = output[0]

    # Get the path kind of the the returned CollectorOutputEntry.
    In [12]: log_file.path_kind
    Out[12]: 'file'

    # Get the path of the returned CollectorOutputEntry.
    In [13]: log_file.path
    Out[13]: 'adb_log.txt'

    # Find the full path to the log file.
    In [14]: os.path.join(os.getcwd(), logfile)
    Out[14]: '/tmp/adb_log.txt'


API
---
.. collector:

.. module:: devlib.collector


CollectorBase
~~~~~~~~~~~~~

.. class:: CollectorBase(target, \*\*kwargs)

   A ``CollectorBase`` is the the base class and API that should be
   implemented to allowing collecting various data from a traget e.g. traces,
   logs etc.

.. method::  Collector.setup(\*args, \*\*kwargs)

   This will set up the collector on the target. Parameters this method takes
   are particular to subclasses (see documentation for specific collectors
   below).  What actions are performed by this method are also
   collector-specific.  Usually these will be things like  installing
   executables, starting services, deploying assets, etc. Typically, this method
   needs to be invoked at most once per reboot of the target (unless
   ``teardown()`` has been called), but see documentation for the collector
   you're interested in.

.. method:: CollectorBase.reset()

   This can be used to configure a collector for collection. This must be invoked
   before ``start()`` is called to begin collection.

.. method:: CollectorBase.start()

   Starts collecting from the target.

.. method:: CollectorBase.stop()

   Stops collecting from target. Must be called after
   :func:`start()`.


.. method:: CollectorBase.set_output(output_path)

   Configure the output path for the particular collector. This will be either
   a directory or file path which will be used when storing the data. Please see
   the individual Collector documentation for more information.


.. method:: CollectorBase.get_data()

    The collected data will be return via the previously specified output_path.
    This method will return a ``CollectorOutput`` object which is a subclassed
    list object containing individual ``CollectorOutputEntry`` objects with details
    about the individual output entry.


CollectorOutputEntry
~~~~~~~~~~~~~~~~~~~~

This object is designed to allow for the output of a collector to be processed
generically. The object will behave as a regular string containing the path to
underlying output path and can be used directly in ``os.path`` operations.

.. attribute:: CollectorOutputEntry.path

    The file path for the corresponding output item.

.. attribute:: CollectorOutputEntry.path_kind

    The type of output the is specified in the ``path`` attribute. Current valid
    kinds are: ``file`` and ``directory``.

.. method:: CollectorOutputEntry.__init__(path, path_kind)

    Initialises a ``CollectorOutputEntry`` object with the desired file path and
    kind of file path specified.


.. collectors:

Available Collectors
---------------------

This section lists collectors that are currently part of devlib.

.. todo:: Add collectors
