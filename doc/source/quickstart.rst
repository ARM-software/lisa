==========
Quickstart
==========

This sections will show you how to quickly start running workloads using
Workload Automation 2.


Install
=======

.. note:: This is a quick summary. For more detailed instructions, please see
          the :doc:`installation` section.

Make sure you have Python 2.7 and a recent Android SDK with API level 18 or above
installed on your system. For the SDK, make sure that either ``ANDROID_HOME``
environment variable is set, or that ``adb`` is in your ``PATH``.

.. note:: A complete install of the Android SDK is required, as WA uses a
          number of its utilities, not just adb.

In addition to the base Python 2.7 install, you will also need to have ``pip``
(Python's package manager) installed as well. This is usually a separate package.

Once you have the pre-requisites and a tarball with the workload automation package,
you can install it with pip::

        sudo pip install wlauto-2.2.0dev.tar.gz

This will install Workload Automation on your system, along with the Python
packages it depends on.

(Optional) Verify installation
-------------------------------

Once the tarball has been installed, try executing ::

        wa -h

You should see a help message outlining available subcommands.


(Optional) APK files
--------------------

A large number of WA workloads are installed as APK files. These cannot be
distributed with WA and so you will need to obtain those separately. 

For more details, please see the :doc:`installation` section.


Configure Your Device
=====================

Out of the box, WA is configured to work with a generic Android device through
``adb``. If you only have one device listed when you execute ``adb devices``,
and your device has a standard Android configuration, then no extra configuration
is required (if your device is connected via network, you will have to manually execute
``adb connect <device ip>`` so that it appears in the device listing).

If you have  multiple devices connected, you will need to tell WA which one you
want it to use. You can do that by setting ``adb_name`` in device configuration inside
``~/.workload_automation/config.py``\ , e.g.

.. code-block:: python

        # ...

        device_config = dict(
                adb_name = 'abcdef0123456789',
                # ...
        )

        # ...

This should give you basic functionality. If your device has non-standard
Android configuration (e.g. it's a development board) or your need some advanced
functionality (e.g. big.LITTLE tuning parameters), additional configuration may
be required. Please see the :doc:`device_setup` section for more details.


Running Your First Workload
===========================

The simplest way to run a workload is to specify it as a parameter to WA ``run``
sub-command::

        wa run dhrystone

You will see INFO output from WA as it executes each stage of the run. A
completed run output should look something like this::

        INFO     Initializing
        INFO     Running workloads
        INFO     Connecting to device
        INFO     Initializing device
        INFO     Running workload 1 dhrystone (iteration 1)
        INFO            Setting up
        INFO            Executing
        INFO            Processing result
        INFO            Tearing down
        INFO     Processing overall results
        INFO     Status available in wa_output/status.txt
        INFO     Done.
        INFO     Ran a total of 1 iterations: 1 OK
        INFO     Results can be found in wa_output

Once the run has completed, you will find a directory called ``wa_output``
in the location where you have invoked ``wa run``. Within this directory,
you will find a "results.csv" file which will contain results obtained for
dhrystone, as well as a "run.log" file containing detailed log output for
the run. You will also find a sub-directory called 'drystone_1_1' that
contains the results for that iteration. Finally, you will find a copy of the
agenda file in the ``wa_output/__meta`` subdirectory. The contents of
iteration-specific subdirectories will vary from workload to workload, and,
along with the contents of the main output directory, will depend on the
instrumentation and result processors that were enabled for that run.

The ``run`` sub-command takes a number of options that control its behavior,
you can view those by executing ``wa run -h``. Please see the :doc:`invocation`
section for details.


Create an Agenda
================

Simply running a single workload is normally of little use. Typically, you would
want to specify several workloads, setup the device state and, possibly, enable
additional instrumentation. To do this, you would need to create an "agenda" for
the run that outlines everything you want WA to do.

Agendas are written using YAML_ markup language. A simple agenda might look
like this:

.. code-block:: yaml

        config:
                instrumentation: [~execution_time]
                result_processors: [json]
        global:
                iterations: 2
        workloads:
                - memcpy
                - name: dhrystone
                  params:
                        mloops: 5
                        threads: 1

This agenda

- Specifies two workloads: memcpy and dhrystone.
- Specifies that dhrystone should run in one thread and execute five million loops.
- Specifies that each of the two workloads should be run twice.
- Enables json result processor, in addition to the result processors enabled in
  the config.py.
- Disables execution_time instrument, if it is enabled in the config.py

There is a lot more that could be done with an agenda. Please see :doc:`agenda`
section for details.

.. _YAML: http://en.wikipedia.org/wiki/YAML

