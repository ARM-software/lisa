Revent Recordings
=================

Convention for Naming revent Files for Revent Workloads
-------------------------------------------------------------------------------

There is a convention for naming revent files which you should follow if you
want to record your own revent files. Each revent file must be called (case sensitive)
``<device name>.<stage>.revent``,
where ``<device name>`` is the name of your device (as defined by the model
name of your device which can be retrieved with
``adb shell getprop ro.product.model`` or by the ``name`` attribute of your
customized device class), and ``<stage>`` is one of the following currently
supported stages:

        :setup: This stage is where the application is loaded (if present). It is
                a good place to record an revent here to perform any tasks to get
                ready for the main part of the workload to start.
        :run: This stage is where the main work of the workload should be performed.
              This will allow for more accurate results if the revent file for this
              stage only records the main actions under test.
        :extract_results: This stage is used after the workload has been completed
                          to retrieve any metrics from the workload e.g. a score.
        :teardown: This stage is where any final actions should be performed to
                   clean up the workload.

Only the run stage is mandatory, the remaining stages will be replayed if a
recording is present otherwise no actions will be performed for that particular
stage.

All your custom revent files should reside at
``'$WA_USER_DIRECTORY/dependencies/WORKLOAD NAME/'``. So
typically to add a custom revent files for a device named "mydevice" and a
workload name "myworkload", you would need to add the revent files to the
directory ``~/.workload_automation/dependencies/myworkload/revent_files``
creating the directory structure if necessary. ::

    mydevice.setup.revent
    mydevice.run.revent
    mydevice.extract_results.revent
    mydevice.teardown.revent

Any revent file in the dependencies will always overwrite the revent file in the
workload directory. So for example it is possible to just provide one revent for
setup in the dependencies and use the run.revent that is in the workload directory.


File format of revent recordings
--------------------------------

You do not need to understand recording format in order to use revent. This
section is intended for those looking to extend revent in some way, or to
utilize revent recordings for other purposes.

Format Overview
~~~~~~~~~~~~~~~

Recordings are stored in a binary format. A recording consists of three
sections::

    +-+-+-+-+-+-+-+-+-+-+-+
    |       Header        |
    +-+-+-+-+-+-+-+-+-+-+-+
    |                     |
    |  Device Description |
    |                     |
    +-+-+-+-+-+-+-+-+-+-+-+
    |                     |
    |                     |
    |     Event Stream    |
    |                     |
    |                     |
    +-+-+-+-+-+-+-+-+-+-+-+

The header contains metadata describing the recording. The device description
contains information about input devices involved in this recording. Finally,
the event stream contains the recorded input events.

All fields are either fixed size or prefixed with their length or the number of
(fixed-sized) elements.

.. note:: All values below are little endian


Recording Header
~~~~~~~~~~~~~~~~

An revent recoding header has the following structure

 * It starts with the "magic" string ``REVENT`` to indicate that this is an
   revent recording.
 * The magic is followed by a 16 bit version number. This indicates the format
   version of the recording that follows. Current version is ``2``.
 * The next 16 bits indicate the type of the recording. This dictates the
   structure of the Device Description section. Valid values are:

        ``0``
                This is a general input event recording. The device description
                contains a list of paths from which the events where recorded.
        ``1``
                This a gamepad recording. The device description contains the
                description of the gamepad used to create the recording.

 * The header is zero-padded to 128 bits.

::

     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |      'R'      |      'E'      |      'V'      |      'E'      |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |      'N'      |      'T'      |            Version            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |             Mode              |            PADDING            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                            PADDING                            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


Device Description
~~~~~~~~~~~~~~~~~~

This section describes the input devices used in the recording. Its structure is
determined by the value of ``Mode`` field in the header.

General Recording
~~~~~~~~~~~~~~~~~

.. note:: This is the only format supported prior to version ``2``.

The recording has been made from all available input devices. This section
contains the list of ``/dev/input`` paths for the devices, prefixed with total
number of the devices recorded.

::

     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                       Number of devices                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    |             Device paths              +-+-+-+-+-+-+-+-+-+-+-+-+
    |                                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


Similarly, each device path is a length-prefixed string. Unlike C strings, the
path is *not* NULL-terminated.

::

     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                     Length of device path                     |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    |                          Device path                          |
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


Gamepad Recording
~~~~~~~~~~~~~~~~~

The recording has been made from a specific gamepad. All events in the stream
will be for that device only. The section describes the device properties that
will be used to create a virtual input device using ``/dev/uinput``. Please
see ``linux/input.h`` header in the Linux kernel source for more information
about the fields in this section.

::

     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |            bustype            |             vendor            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |            product            |            version            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                         name_length                           |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    |                             name                              |
    |                                                               |
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                            ev_bits                            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    |                                                               |
    |                       key_bits (96 bytes)                     |
    |                                                               |
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    |                                                               |
    |                       rel_bits (96 bytes)                     |
    |                                                               |
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    |                                                               |
    |                       abs_bits (96 bytes)                     |
    |                                                               |
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                          num_absinfo                          |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    |                                                               |
    |                                                               |
    |                                                               |
    |                        absinfo entries                        |
    |                                                               |
    |                                                               |
    |                                                               |
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


Each ``absinfo`` entry consists of six 32 bit values. The number of entries is
determined by the ``abs_bits`` field.


::

     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                            value                              |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                           minimum                             |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                           maximum                             |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                             fuzz                              |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                             flat                              |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                          resolution                           |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


Event Stream
~~~~~~~~~~~~

The majority of an revent recording will be made up of the input events that were
recorded. The event stream is prefixed with the number of events in the stream,
and start and end times for the recording.

::

     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                        Number of events                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                  Number of events (cont.)                     |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                      Start Time Seconds                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                  Start Time Seconds (cont.)                   |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                    Start Time Microseconds                    |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |              Start Time Microseconds (cont.)                  |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                        End Time Seconds                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                    End Time Seconds (cont.)                   |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                      End Time Microseconds                    |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                End Time Microseconds (cont.)                  |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                               |
    |                                                               |
    |             Events                                            |
    |                                                               |
    |                                                               |
    |                                       +-+-+-+-+-+-+-+-+-+-+-+-+
    |                                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


Event Structure
~~~~~~~~~~~~~~~

Each event entry structured as follows:

 * An unsigned short integer representing which device from the list of device paths
   this event is for (zero indexed). E.g. Device ID = 3 would be the 4th
   device in the list of device paths.
 * A unsigned long integer representing the number of seconds since "epoch" when
   the event was recorded.
 * A unsigned long integer representing the microseconds part of the timestamp.
 * An unsigned integer representing the event type
 * An unsigned integer representing the event code
 * An unsigned integer representing the event value

For more information about the event type, code and value please read:
https://www.kernel.org/doc/Documentation/input/event-codes.txt

::

     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |           Device ID           |        Timestamp Seconds      |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                       Timestamp Seconds (cont.)               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |   Timestamp Seconds (cont.)   |        stamp Micoseconds      |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |              Timestamp Micoseconds (cont.)                    |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    | Timestamp Micoseconds (cont.) |          Event Type           |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |          Event Code           |          Event Value          |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |       Event Value (cont.)     |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


Parser
~~~~~~

WA has a parser for revent recordings. This can be used to work with revent
recordings in scripts. Here is an example:

.. code:: python

    from wa.utils.revent import ReventRecording

    with ReventRecording('/path/to/recording.revent') as recording:
        print("Recording: {}".format(recording.filepath))
        print("There are {} input events".format(recording.num_events))
        print("Over a total of {} seconds".format(recording.duration))
