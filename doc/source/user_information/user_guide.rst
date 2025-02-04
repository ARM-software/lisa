.. _user-guide:

**********
User Guide
**********

This guide will show you how to quickly start running workloads using
Workload Automation 3.

.. contents:: Contents
   :depth: 2
   :local:

---------------------------------------------------------------


Install
=======

.. note:: This is a quick summary. For more detailed instructions, please see
          the :ref:`installation` section.

Make sure you have Python 3.5+ and a recent Android SDK with API
level 18 or above installed on your system. A complete install of the Android
SDK is required, as WA uses a number of its utilities, not just adb. For the
SDK, make sure that either ``ANDROID_HOME`` environment variable is set, or that
``adb`` is in your ``PATH``.

.. Note:: If you plan to run Workload Automation on Linux devices only, SSH is required,
          and Android SDK is optional if you wish to run WA on Android devices at a
          later time.

          However, you would be starting off with a limited number of workloads that
          will run on Linux devices.

In addition to the base Python install, you will also need to have ``pip``
(Python's package manager) installed as well. This is usually a separate package.

Once you have those, you can install WA with::

        sudo -H pip install wlauto

This will install Workload Automation on your system, along with its mandatory
dependencies.

Alternatively we provide a Dockerfile that which can be used to create a Docker
image for running WA along with its dependencies. More information can be found
in the :ref:`Installation <dockerfile>` section.

(Optional) Verify installation
-------------------------------

Once the tarball has been installed, try executing ::

        wa -h

You should see a help message outlining available subcommands.


(Optional) APK files
--------------------

A large number of WA workloads are installed as APK files. These cannot be
distributed with WA and so you will need to obtain those separately.

For more details, please see the :ref:`installation <apk_files>` section.


List Command
============

In order to get started with using WA we first we need to find
out what is available to use. In order to do this we can use the :ref:`list <list-command>`
command followed by the type of plugin that you wish to see.

For example to see what workloads are available along with a short description
of each you run::

    wa list workloads

Which will give an output in the format of:

.. code-block:: none

             adobereader:    The Adobe Reader workflow carries out the following typical
                             productivity tasks.
              androbench:    Executes storage performance benchmarks
          angrybirds_rio:    Angry Birds Rio game.
                  antutu:    Executes Antutu 3D, UX, CPU and Memory tests
               applaunch:    This workload launches and measures the launch time of applications
                             for supporting workloads.
             benchmarkpi:    Measures the time the target device takes to run and complete the
                             Pi calculation algorithm.
               dhrystone:    Runs the Dhrystone benchmark.
               exoplayer:    Android ExoPlayer
               geekbench:    Geekbench provides a comprehensive set of benchmarks engineered to
                             quickly and accurately measure
                             processor and memory performance.
            #..

The same syntax can be used to display ``commands``,
``energy_instrument_backends``, ``instruments``, ``output_processors``,
``resource_getters``, ``targets``. Once you have found the plugin you are
looking for you can use the :ref:`show <show-command>` command to display more
detailed information.  Alternatively please see the
:ref:`Plugin Reference <plugin-reference>` for an online version.

Show Command
============

If you want to learn more information about a particular plugin, such as the
parameters it supports, you can use the "show" command::

    wa show dhrystone

If you have ``pandoc`` installed on your system, this will display man
page-like description of the plugin, and the parameters it supports. If you do
not have ``pandoc``, you will instead see the same information as raw
restructured text.

Configure Your Device
=====================

There are multiple options for configuring your device depending on your
particular use case.

You can either add your configuration to the default configuration file
``config.yaml``, under the ``$WA_USER_DIRECTORY/`` directory or you can specify it in
the ``config`` section of your agenda directly.

Alternatively if you are using multiple devices, you may want to create separate
config files for each of your devices you will be using. This allows you to
specify which device you would like to use for a particular run and pass it as
an argument when invoking with the ``-c`` flag.
::

    wa run dhrystone -c my_device.yaml

By default WA will use the “most specific” configuration available for example
any configuration specified inside an agenda will override a passed
configuration file which will in turn overwrite the default configuration file.

.. note:: For a more information about configuring your
          device please see :ref:`Setting Up A Device <setting-up-a-device>`.

Android
-------

By default, the device WA will use is set to 'generic_android'. WA is configured
to work with a generic Android device through ``adb``. If you only have one
device listed when you execute ``adb devices``, and your device has a standard
Android configuration, then no extra configuration is required.

However, if your device is connected via network, you will have to manually
execute ``adb connect <device ip>`` (or specify this in your
:ref:`agenda <agenda>`) so that it appears in the device listing.

If you have multiple devices connected, you will need to tell WA which one you
want it to use. You can do that by setting ``device`` in the device_config section.

.. code-block:: yaml

        # ...

        device_config:
                device: 'abcdef0123456789'
                # ...
        # ...

Linux
-----

First, set the device to 'generic_linux'

.. code-block:: yaml

        # ...
          device: 'generic_linux'
        # ...

Find the device_config section and add these parameters

.. code-block:: yaml

        # ...

        device_config:
                host: '192.168.0.100'
                username: 'root'
                password: 'password'
                # ...
        # ...

Parameters:

- Host is the IP of your target Linux device
- Username is the user for the device
- Password is the password for the device

Enabling and Disabling Augmentations
---------------------------------------

Augmentations are the collective name  for  "instruments" and "output
processors" in WA3.

Some augmentations are enabled by default after your initial install of WA,
which are specified in the ``config.yaml`` file located in your
``WA_USER_DIRECTORY``, typically ``~/.workload_autoamation``.

.. note:: Some Linux devices may not be able to run certain augmentations
          provided by WA (e.g. cpufreq is disabled or unsupported by the
          device).

.. code-block:: yaml

        # ...

        augmentations:
            # Records the time it took to run the workload
            - execution_time

            # Collects /proc/interrupts before and after execution and does a diff.
            - interrupts

            # Collects the contents of/sys/devices/system/cpu before and after
            # execution and does a diff.
            - cpufreq

            # Generate a txt file containing general status information about
            # which runs failed and which were successful.
            - status

            # ...

If you only wanted to keep the 'execution_time' instrument enabled, you can comment out
the rest of the list augmentations to disable them.

This should give you basic functionality. If you are working with a development
board or you need some advanced functionality additional configuration may be required.
Please see the :ref:`device setup <setting-up-a-device>` section for more details.

.. note:: In WA2 'Instrumentation' and 'Result Processors' were divided up into their
          own sections in the agenda. In WA3 they now fall under the same category of
          'augmentations'. For compatibility the old naming structure is still valid
          however using the new entry names is recommended.



Running Your First Workload
===========================

The simplest way to run a workload is to specify it as a parameter to WA ``run``
:ref:`run <run-command>` sub-command::

        wa run dhrystone

You will see INFO output from WA as it executes each stage of the run. A
completed run output should look something like this::

        INFO     Creating output directory.
        INFO     Initializing run
        INFO     Connecting to target
        INFO     Setting up target
        INFO     Initializing execution context
        INFO     Generating jobs
        INFO         Loading job wk1 (dhrystone) [1]
        INFO     Installing instruments
        INFO     Installing output processors
        INFO     Starting run
        INFO     Initializing run
        INFO         Initializing job wk1 (dhrystone) [1]
        INFO     Running job wk1
        INFO         Configuring augmentations
        INFO         Configuring target for job wk1 (dhrystone) [1]
        INFO         Setting up job wk1 (dhrystone) [1]
        INFO         Running job wk1 (dhrystone) [1]
        INFO         Tearing down job wk1 (dhrystone) [1]
        INFO         Completing job wk1
        INFO     Job completed with status OK
        INFO     Finalizing run
        INFO         Finalizing job wk1 (dhrystone) [1]
        INFO     Done.
        INFO     Run duration: 9 seconds
        INFO     Ran a total of 1 iterations: 1 OK
        INFO     Results can be found in wa_output


Once the run has completed, you will find a directory called ``wa_output``
in the location where you have invoked ``wa run``. Within this directory,
you will find a "results.csv" file which will contain results obtained for
dhrystone, as well as a "run.log" file containing detailed log output for
the run. You will also find a sub-directory called 'wk1-dhrystone-1' that
contains the results for that iteration. Finally, you will find various additional
information in the ``wa_output/__meta`` subdirectory for example information
extracted from the target and a copy of the agenda file. The contents of
iteration-specific subdirectories will vary from workload to workload, and,
along with the contents of the main output directory, will depend on the
augmentations that were enabled for that run.

The ``run`` sub-command takes a number of options that control its behaviour,
you can view those by executing ``wa run -h``. Please see the :ref:`invocation`
section for details.


Create an Agenda
================

Simply running a single workload is normally of little use. Typically, you would
want to specify several workloads, setup the device state and, possibly, enable
additional augmentations. To do this, you would need to create an "agenda" for
the run that outlines everything you want WA to do.

Agendas are written using YAML_ markup language. A simple agenda might look
like this:

.. code-block:: yaml

        config:
                augmentations:
                    - ~execution_time
                    - targz
                iterations: 2
        workloads:
                - memcpy
                - name: dhrystone
                  params:
                        mloops: 5
                        threads: 1

This agenda:

- Specifies two workloads: memcpy and dhrystone.
- Specifies that dhrystone should run in one thread and execute five million loops.
- Specifies that each of the two workloads should be run twice.
- Enables the targz output processor, in addition to the output processors enabled in
  the config.yaml.
- Disables execution_time instrument, if it is enabled in the config.yaml

An agenda can be created using WA's ``create`` :ref:`command <using-the-create-command>`
or in a text editor and saved as a YAML file.

For more options please see the :ref:`agenda` documentation.

.. _YAML: http://en.wikipedia.org/wiki/YAML

.. _using-the-create-command:

Using the Create Command
-------------------------
The easiest way to create an agenda is to use the 'create' command. For more
in-depth information please see the :ref:`Create Command <create-command>` documentation.

In order to populate the agenda with relevant information you can supply all of
the plugins you wish to use as arguments to the command, for example if we want
to create an agenda file for running ``dhrystone`` on a `generic_android` device and we
want to enable the ``execution_time`` and ``trace-cmd`` instruments and display the
metrics using the ``csv`` output processor. We would use the following command::

    wa create agenda generic_android dhrystone execution_time trace-cmd csv -o my_agenda.yaml

This will produce a ``my_agenda.yaml`` file containing all the relevant
configuration for the specified plugins along with their default values as shown
below:

.. code-block:: yaml

        config:
            augmentations:
            - execution_time
            - trace-cmd
            - csv
            iterations: 1
            device: generic_android
            device_config:
                adb_server: null
                adb_port: null
                big_core: null
                core_clusters: null
                core_names: null
                device: null
                disable_selinux: true
                executables_directory: null
                load_default_modules: true
                logcat_poll_period: null
                model: null
                modules: null
                package_data_directory: /data/data
                shell_prompt: !<tag:wa:regex> '8:^.*(shell|root)@.*:/\S* [#$] '
                working_directory: null
            execution_time: {}
            trace-cmd:
                buffer_size: null
                buffer_size_step: 1000
                events:
                - sched*
                - irq*
                - power*
                - thermal*
                functions: null
                no_install: false
                report: true
                report_on_target: false
                mode: write-to-memory
            csv:
                extra_columns: null
                use_all_classifiers: false
        workloads:
        -   name: dhrystone
            params:
                cleanup_assets: true
                delay: 0
                duration: 0
                mloops: 0
                taskset_mask: 0
                threads: 4


Run Command
============
These examples show some useful options that can be used with WA's ``run`` command.

Once we have created an agenda to use it with WA we can pass it as a argument to
the run command e.g.::

    wa run <path/to/agenda> (e.g. wa run ~/myagenda.yaml)

By default WA will use the "wa_output" directory to stores its output however to
redirect the output to a different directory we can use::

    wa run dhrystone -d my_output_directory

We can also tell WA to use additional config files by supplying it with
the ``-c`` argument. One use case for passing additional config files is if you
have multiple devices you wish test with WA, you can store the relevant device
configuration in individual config files and then pass the file corresponding to
the device you wish to use for that particular test.

.. note:: As previously mentioned, any more specific configuration present in
          the agenda file will overwrite the corresponding config parameters
          specified in the config file(s).


::

    wa run -c myconfig.yaml ~/myagenda.yaml

To use the same output directory but override the existing contents to
store new dhrystone results we can specify the ``-f`` argument::

    wa run -f dhrystone

To display verbose output while running memcpy::

    wa run --verbose memcpy


.. _output_directory:

Output
======

The output directory will contain subdirectories for each job that was run,
which will in turn contain the generated metrics and artifacts for each job.
The directory will also contain a ``run.log`` file containing the complete log
output for the run, and a ``__meta`` directory with the configuration and
metadata for the run. Metrics are serialized inside ``result.json`` files inside
each job's subdirectory. There may also be a ``__failed`` directory containing
failed attempts for jobs that have been re-run.

Augmentations may add additional files at the run or job directory level. The
default configuration has ``status`` and ``csv`` augmentations enabled which
generate a ``status.txt`` containing status summary for the run and individual
jobs, and a ``results.csv`` containing metrics from all jobs in a CSV table,
respectively.

See :ref:`output_directory_structure` for more information.

In order to make it easier to access WA results from scripts, WA provides an API
that parses the contents of the output directory:


.. code-block:: pycon

    >>> from wa import RunOutput
    >>> ro = RunOutput('./wa_output')
    >>> for job in ro.jobs:
    ...     if job.status != 'OK':
    ...         print('Job "{}" did not complete successfully: {}'.format(job, job.status))
    ...         continue
    ...     print('Job "{}":'.format(job))
    ...     for metric in job.metrics:
    ...         if metric.units:
    ...             print('\t{}: {} {}'.format(metric.name, metric.value, metric.units))
    ...         else:
    ...             print('\t{}: {}'.format(metric.name, metric.value))
    ...
    Job "wk1-dhrystone-1":
            thread 0 score: 20833333
            thread 0 DMIPS: 11857
            thread 1 score: 24509804
            thread 1 DMIPS: 13950
            thread 2 score: 18011527
            thread 2 DMIPS: 10251
            thread 3 score: 26371308
            thread 3 DMIPS: 15009
            time: 1.001251 seconds
            total DMIPS: 51067
            total score: 89725972
            execution_time: 1.4834280014 seconds

See  :ref:`output_processing_api` for details.

Uninstall
=========

If you have installed Workload Automation via ``pip``, then run this command to
uninstall it::

    sudo pip uninstall wa


.. Note:: It will *not* remove any user configuration (e.g. the ~/.workload_automation
          directory).

Upgrade
=======

To upgrade Workload Automation to the latest version via ``pip``, run::

    sudo pip install --upgrade --no-deps wa

