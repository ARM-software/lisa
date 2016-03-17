.. _invocation:

========
Commands
========

Installing the wlauto package will add ``wa`` command to your system,
which you can run from anywhere. This has a number of sub-commands, which can 
be viewed by executing ::

        wa -h

Individual sub-commands are discussed in detail below.

run
---

The most common sub-command you will use is ``run``. This will run specfied
workload(s) and process resulting output. This takes a single mandatory
argument that specifies what you want WA to run. This could be either a
workload name, or a path  to an "agenda" file that allows to specify multiple
workloads as well as a lot additional configuration (see :ref:`agenda`
section for details). Executing ::

        wa run -h

Will display help for this subcommand that will look somehtign like this::

        usage: run [-d DIR] [-f] AGENDA

        Execute automated workloads on a remote device and process the resulting
        output.

        positional arguments:
          AGENDA                Agenda for this workload automation run. This defines
                                which workloads will be executed, how many times, with
                                which tunables, etc. See /usr/local/lib/python2.7
                                /dist-packages/wlauto/agenda-example.csv for an
                                example of how this file should be structured.

        optional arguments:
          -h, --help            show this help message and exit
          -c CONFIG, --config CONFIG
                                specify an additional config.py
          -v, --verbose         The scripts will produce verbose output.
          --version             Output the version of Workload Automation and exit.
          --debug               Enable debug mode. Note: this implies --verbose.
          -d DIR, --output-directory DIR
                                Specify a directory where the output will be
                                generated. If the directoryalready exists, the script
                                will abort unless -f option (see below) is used,in
                                which case the contents of the directory will be
                                overwritten. If this optionis not specified, then
                                wa_output will be used instead.
          -f, --force           Overwrite output directory if it exists. By default,
                                the script will abort in thissituation to prevent
                                accidental data loss.
          -i ID, --id ID        Specify a workload spec ID from an agenda to run. If
                                this is specified, only that particular spec will be
                                run, and other workloads in the agenda will be
                                ignored. This option may be used to specify multiple
                                IDs.


Output Directory
~~~~~~~~~~~~~~~~

The exact contents on the output directory will depend on configuration options
used, instrumentation and output processors enabled, etc. Typically, the output
directory will contain a results file at the top level that lists all
measurements that were collected (currently, csv and json formats are
supported), along with a subdirectory for each iteration executed with output
for that specific iteration.

At the top level, there will also be a run.log file containing the complete log
output for the execution. The contents of this file is equivalent to what you
would get in the console when using --verbose option.

Finally, there will be a __meta subdirectory. This will contain a copy of the
agenda file used to run the workloads along with any other device-specific
configuration files used during execution.


list
----

This lists all plugins of a particular type. For example ::

        wa list workloads

will list all workloads currently included in WA. The list will consist of
plugin names and short descriptions of the functionality they offer.


show
----

This will show detailed information about an plugin, including more in-depth
description and any parameters/configuration that are available.  For example
executing ::

        wa show andebench

will produce something like ::


        andebench

        AndEBench is an industry standard Android benchmark provided by The Embedded Microprocessor Benchmark Consortium
        (EEMBC).

        parameters:

        number_of_threads
        Number of threads that will be spawned by AndEBench.
                type: int

        single_threaded
        If ``true``, AndEBench will run with a single thread. Note: this must not be specified if ``number_of_threads``
        has been specified.
                type: bool

        http://www.eembc.org/andebench/about.php

        From the website:

        - Initial focus on CPU and Dalvik interpreter performance
        - Internal algorithms concentrate on integer operations
        - Compares the difference between native and Java performance
        - Implements flexible multicore performance analysis
        - Results displayed in Iterations per second
        - Detailed log file for comprehensive engineering analysis



