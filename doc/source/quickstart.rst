==========
Quickstart
==========

This guide will show you how to quickly start running workloads using
Workload Automation 2.


Install
=======

.. note:: This is a quick summary. For more detailed instructions, please see
          the :doc:`installation` section.

Make sure you have Python 2.7 and a recent Android SDK with API level 18 or above
installed on your system. A complete install of the Android SDK is required, as
WA uses a number of its utilities, not just adb. For the SDK, make sure that either
``ANDROID_HOME`` environment variable is set, or that ``adb`` is in your ``PATH``.

.. Note:: If you plan to run Workload Automation on Linux devices only, SSH is required,
          and Android SDK is optional if you wish to run WA on Android devices at a
          later time.

          However, you would be starting off with a limited number of workloads that
          will run on Linux devices.

In addition to the base Python 2.7 install, you will also need to have ``pip``
(Python's package manager) installed as well. This is usually a separate package.

Once you have those, you can install WA with::

        sudo -H pip install wlauto

This will install Workload Automation on your system, along with its mandatory 
dependencies.

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

Locate the device configuration file, config.py, under the
~/.workload_automation directory. Then adjust the device 
configuration settings accordingly to the device you are using.

Android
-------

By default, the device is set to 'generic_android'. WA is configured to work 
with a generic Android device through ``adb``. If you only have one device listed 
when you execute ``adb devices``, and your device has a standard Android 
configuration, then no extra configuration is required.

However, if your device is connected via network, you will have to manually execute
``adb connect <device ip>`` so that it appears in the device listing.

If you have multiple devices connected, you will need to tell WA which one you
want it to use. You can do that by setting ``adb_name`` in device_config section.

.. code-block:: python

        # ...

        device_config = dict(
                adb_name = 'abcdef0123456789',
                # ...
        )

        # ...

Linux
-----

First, set the device to 'generic_linux'

.. code-block:: python

        # ...
          device = 'generic_linux'
        # ...

Find the device_config section and add these parameters

.. code-block:: python

        # ...

        device_config = dict(
                host = '192.168.0.100',
                username = 'root',
                password = 'password'
                # ...
        )

        # ...

Parameters:

- Host is the IP of your target Linux device
- Username is the user for the device
- Password is the password for the device

Enabling and Disabling Instrumentation
---------------------------------------

Some instrumentation tools are enabled after your initial install of WA.

.. note:: Some Linux devices may not be able to run certain instruments
          provided by WA (e.g. cpufreq is disabled or unsupported by the 
          device). 

As a start, keep the 'execution_time' instrument enabled while commenting out
the rest to disable them.

.. code-block:: python

        # ...

        Instrumentation = [
                # Records the time it took to run the workload
                'execution_time',

                # Collects /proc/interrupts before and after execution and does a diff.
                # 'interrupts',

                # Collects the contents of/sys/devices/system/cpu before and after execution and does a diff.
                # 'cpufreq',

                # ...
        )



This should give you basic functionality. If you are working with a development 
board or you need some advanced functionality (e.g. big.LITTLE tuning parameters), 
additional configuration may be required. Please see the :doc:`device_setup` 
section for more details.


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

An agenda can be created in a text editor and saved as a YAML file. Please make note of
where you have saved the agenda.

Please see :doc:`agenda` section for more options.

.. _YAML: http://en.wikipedia.org/wiki/YAML

Examples
========

These examples show some useful options with the ``wa run`` command.

To run your own agenda::
    
    wa run <path/to/agenda> (e.g. wa run ~/myagenda.yaml)

To redirect the output to a different directory other than wa_output::
    
    wa run dhrystone -d my_output_directory

To use a different config.py file::
    
    wa run -c myconfig.py dhrystone

To use the same output directory but override existing contents to
store new dhrystone results::
    
    wa run -f dhrystone

To display verbose output while running memcpy::

    wa run --verbose memcpy

Uninstall
=========

If you have installed Workload Automation via ``pip``, then run this command to
uninstall it::

    sudo pip uninstall wlauto


.. Note:: It will *not* remove any user configuration (e.g. the ~/.workload_automation 
          directory).

Upgrade
=======

To upgrade Workload Automation to the latest version via ``pip``, run::
    
    sudo pip install --upgrade --no-deps wlauto

