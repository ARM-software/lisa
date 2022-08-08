.. _invocation:

Commands
========

Installing the wa package will add ``wa`` command to your system,
which you can run from anywhere. This has a number of sub-commands, which can
be viewed by executing ::

        wa -h

Individual sub-commands are discussed in detail below.

.. _run-command:

Run
---

The most common sub-command you will use is ``run``. This will run the specified
workload(s) and process its resulting output. This takes a single mandatory
argument which specifies what you want WA to run. This could be either a workload
name, or a path to an agenda" file that allows to specify multiple workloads as
well as a lot additional configuration (see :ref:`agenda` section for details).
Executing ::

        wa run -h

Will display help for this subcommand that will look something like this:

.. code-block:: none

        usage: wa run [-h] [-c CONFIG] [-v] [--version] [-d DIR] [-f] [-i ID]
              [--disable INSTRUMENT]
              AGENDA

        Execute automated workloads on a remote device and process the resulting
        output.

        positional arguments:
          AGENDA                Agenda for this workload automation run. This defines
                                which workloads will be executed, how many times, with
                                which tunables, etc. See example agendas in
                                /usr/local/lib/python3.X/dist-packages/wa for an
                                example of how this file should be structured.

        optional arguments:
          -h, --help            show this help message and exit
          -c CONFIG, --config CONFIG
                                specify an additional config.yaml
          -v, --verbose         The scripts will produce verbose output.
          --version             show program's version number and exit
          -d DIR, --output-directory DIR
                                Specify a directory where the output will be
                                generated. If the directory already exists, the script
                                will abort unless -f option (see below) is used, in
                                which case the contents of the directory will be
                                overwritten. If this option is not specified, then
                                wa_output will be used instead.
          -f, --force           Overwrite output directory if it exists. By default,
                                the script will abort in this situation to prevent
                                accidental data loss.
          -i ID, --id ID        Specify a workload spec ID from an agenda to run. If
                                this is specified, only that particular spec will be
                                run, and other workloads in the agenda will be
                                ignored. This option may be used to specify multiple
                                IDs.
          --disable INSTRUMENT  Specify an instrument or output processor to disable
                                from the command line. This equivalent to adding
                                "~{metavar}" to the instruments list in the
                                agenda. This can be used to temporarily disable a
                                troublesome instrument for a particular run without
                                introducing permanent change to the config (which one
                                might then forget to revert). This option may be
                                specified multiple times.

.. _list-command:

List
----

This lists all plugins of a particular type. For example ::

        wa list instruments

will list all instruments currently included in WA. The list will consist of
plugin names and short descriptions of the functionality they offer e.g.

.. code-block:: none

    #..
               cpufreq:    Collects dynamic frequency (DVFS) settings before and after
                           workload execution.
                 dmesg:    Collected dmesg output before and during the run.
    energy_measurement:    This instrument is designed to be used as an interface to
                           the various energy measurement instruments located
                           in devlib.
        execution_time:    Measure how long it took to execute the run() methods of
                           a Workload.
           file_poller:    Polls the given files at a set sample interval. The values
                           are output in CSV format.
                   fps:    Measures Frames Per Second (FPS) and associated metrics for
                           a workload.
    #..


You can use the same syntax to quickly display information about ``commands``,
``energy_instrument_backends``, ``instruments``, ``output_processors``, ``resource_getters``,
``targets`` and ``workloads``

.. _show-command:

Show
----

This will show detailed information about an plugin (workloads, targets,
instruments etc.), including a full description and any relevant
parameters/configuration that are available. For example executing ::

        wa show benchmarkpi

will produce something like: ::


        benchmarkpi
        -----------

        Measures the time the target device takes to run and complete the Pi
        calculation algorithm.

        http://androidbenchmark.com/howitworks.php

        from the website:

        The whole idea behind this application is to use the same Pi calculation
        algorithm on every Android Device and check how fast that process is.
        Better calculation times, conclude to faster Android devices. This way you
        can also check how lightweight your custom made Android build is. Or not.

        As Pi is an irrational number, Benchmark Pi does not calculate the actual Pi
        number, but an approximation near the first digits of Pi over the same
        calculation circles the algorithms needs.

        So, the number you are getting in milliseconds is the time your mobile device
        takes to run and complete the Pi calculation algorithm resulting in a
        approximation of the first Pi digits.

        parameters
        ~~~~~~~~~~

        cleanup_assets : boolean
            If ``True``, if assets are deployed as part of the workload they
            will be removed again from the device as part of finalize.

            default: ``True``

        package_name : str
            The package name that can be used to specify
            the workload apk to use.

        install_timeout : integer
            Timeout for the installation of the apk.

            constraint: ``value > 0``

            default: ``300``

        version : str
            The version of the package to be used.

        variant : str
            The variant of the package to be used.

        strict : boolean
            Whether to throw an error if the specified package cannot be found
            on host.

        force_install : boolean
            Always re-install the APK, even if matching version is found already installed
            on the device.

        uninstall : boolean
            If ``True``, will uninstall workload's APK as part of teardown.'

        exact_abi : boolean
            If ``True``, workload will check that the APK matches the target
            device ABI, otherwise any suitable APK found will be used.

        markers_enabled : boolean
            If set to ``True``, workloads will insert markers into logs
            at various points during execution. These markers may be used
            by other plugins or post-processing scripts to provide
            measurements or statistics for specific parts of the workload
            execution.

.. note:: You can also use this command to view global settings by using ``wa show settings``


.. _create-command:

Create
------

This aids in the creation of new WA-related objects for example agendas and workloads.
For more detailed information on creating workloads please see the
:ref:`adding a workload <adding-a-workload-example>` section for more details.

As an example to create an agenda that will run the dhrystone and memcpy workloads
that will use the status and hwmon augmentations, run each test 3 times and save
into the file ``my_agenda.yaml`` the following command can be used::

        wa create agenda dhrystone memcpy status hwmon -i 3 -o my_agenda.yaml

Which will produce something like::

        config:
            augmentations:
            - status
            - hwmon
            status: {}
            hwmon: {}
            iterations: 3
        workloads:
        -   name: dhrystone
            params:
                cleanup_assets: true
                delay: 0
                duration: 0
                mloops: 0
                taskset_mask: 0
                threads: 4
        -   name: memcpy
            params:
                buffer_size: 5242880
                cleanup_assets: true
                cpus: null
                iterations: 1000

This will be populated with default values which can then be customised for the
particular use case.

Additionally the create command can be used to initialize (and update) a
Postgres database which can be used by the ``postgres`` output processor.

The most of database connection parameters have a default value however they can
be overridden via command line arguments. When initializing the database WA will
also save the supplied parameters into the default user config file so that they
do not need to be specified time the output processor is used.

As an example if we had a database server running on at 10.0.0.2 using the
standard port we could use the following command to initialize a database for
use with WA::

        wa create database -a 10.0.0.2 -u my_username -p Pa55w0rd

This will log into the database server with the supplied credentials and create
a database (defaulting to 'wa') and will save the configuration to the
``~/.workload_automation/config.yaml`` file.

With updates to WA there may be changes to the database schema used. In this
case the create command can also be used with the ``-U`` flag to update the
database to use the new schema as follows::

        wa create database -a 10.0.0.2 -u my_username -p Pa55w0rd -U

This will upgrade the database sequentially until the database schema is using
the latest version.

.. _process-command:

Process
--------

This command allows for output processors to be ran on data that was produced by
a previous run.

There are 2 ways of specifying which processors you wish to use, either passing
them directly as arguments to the process command with the ``--processor``
argument or by providing an additional config file with the ``--config``
argument. Please note that by default the process command will not rerun
processors that have already been ran during the run, in order to force a rerun
of the processors you can specific the ``--force`` argument.

Additionally if you have a directory containing multiple run directories you can
specify the ``--recursive`` argument which will cause WA to walk the specified
directory processing all the WA output sub-directories individually.


As an example if we had performed multiple experiments and have the various WA
output directories in our ``my_experiments`` directory, and we now want to process
the outputs with a tool that only supports CSV files. We can easily generate CSV
files for all the runs contained in our directory using the CSV processor by
using the following command::

      wa process -r -p csv my_experiments


.. _record_command:

Record
------

This command simplifies the process of recording revent files. It will
automatically deploy revent and has options to automatically open apps and
record specified stages of a workload. Revent allows you to record raw inputs
such as screen swipes or button presses. This can be useful for recording inputs
for workloads such as games that don't have XML UI layouts that can be used with
UIAutomator. As a drawback from this, revent recordings are specific to the
device type they were recorded on. WA uses two parts to the names of revent
recordings in the format, ``{device_name}.{suffix}.revent``. - device_name can
either be specified manually with the ``-d`` argument or it can be automatically
determined. On Android device it will be obtained from ``build.prop``, on Linux
devices it is obtained from ``/proc/device-tree/model``. - suffix is used by WA
to determine which part of the app execution the recording is for, currently
these are either ``setup``, ``run``, ``extract_results`` or ``teardown``. All
stages except ``run`` are optional for playback and to specify which stages
should be recorded the ``-s``, ``-r``, ``-e`` or ``-t`` arguments respectively,
or optionally ``-a`` to indicate all stages should be recorded.


The full set of options for this command are::

        usage: wa record [-h] [-c CONFIG] [-v] [--version] [-d DEVICE] [-o FILE] [-s]
                         [-r] [-e] [-t] [-a] [-C] [-p PACKAGE | -w WORKLOAD]

        optional arguments:
          -h, --help            show this help message and exit
          -c CONFIG, --config CONFIG
                                specify an additional config.yaml
          -v, --verbose         The scripts will produce verbose output.
          --version             show program's version number and exit
          -d DEVICE, --device DEVICE
                                Specify the device on which to run. This will take
                                precedence over the device (if any) specified in
                                configuration.
          -o FILE, --output FILE
                                Specify the output file
          -s, --setup           Record a recording for setup stage
          -r, --run             Record a recording for run stage
          -e, --extract_results Record a recording for extract_results stage
          -t, --teardown        Record a recording for teardown stage
          -a, --all             Record recordings for available stages
          -C, --clear           Clear app cache before launching it
          -p PACKAGE, --package PACKAGE
                                Android package to launch before recording
          -w WORKLOAD, --workload WORKLOAD
                                Name of a revent workload (mostly games)

For more information please see :ref:`Revent Recording <revent-recording>`.

.. _replay-command:

Replay
------

Alongside ``record`` wa also has a command to playback a single recorded revent
file. It behaves similar to the ``record`` command taking a subset of the same
options allowing you to automatically launch a package on the device ::

    usage: wa replay [-h] [-c CONFIG] [-v] [--debug] [--version] [-p PACKAGE] [-C]
                 revent

    positional arguments:
      revent                The name of the file to replay

    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            specify an additional config.py
      -v, --verbose         The scripts will produce verbose output.
      --debug               Enable debug mode. Note: this implies --verbose.
      --version             show program's version number and exit
      -p PACKAGE, --package PACKAGE
                            Package to launch before recording
      -C, --clear           Clear app cache before launching it

For more information please see :ref:`Revent Replaying  <revent_replaying>`.
