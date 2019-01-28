=================================
What's New in Workload Automation
=================================

*************
Version 3.1.1
*************

Fixes/Improvements
==================

Other
-----
    - Improve formatting when displaying metrics
    - Update revent binaries to include latest fixes
    - Update DockerImage to use new released version of WA and Devlib
    - Fix broken package on PyPi

*************
Version 3.1.0
*************

New Features:
==============

Commands
---------
    - ``create database``: Added :ref:`create subcommand <create-command>`
      command in order to initialize a PostgresSQL database to allow for storing
      WA output with the Postgres Output Processor.

Output Processors:
------------------
    - ``Postgres``: Added output processor which can be used to populate a
      Postgres database with the output generated from a WA run.
    - ``logcat-regex``: Add new output processor to extract arbitrary "key"
      "value" pairs from logcat.

Configuration:
--------------
    - :ref:`Configuration Includes <config-include>`: Add support for including
      other YAML files inside agendas and config files using ``"include#:"``
      entries.
    - :ref:`Section groups <section-groups>`: This allows for a ``group`` entry
      to be specified for each section and will automatically cross product the
      relevant sections with sections from other groups adding the relevant
      classifiers.

Framework:
----------
    - Added support for using the :ref:`OutputAPI <output_processing_api>` with a
      Postgres Database backend. Used to retrieve and
      :ref:`process <processing_output>` run data uploaded by the ``Postgres``
      output processor.

Workloads:
----------
    - ``gfxbench-corporate``: Execute a set of on and offscreen graphical benchmarks from
      GFXBench including Car Chase and Manhattan.
    - ``glbench``: Measures the graphics performance of Android devices by
      testing the underlying OpenGL (ES) implementation.


Fixes/Improvements
==================

Framework:
----------
  - Remove quotes from ``sudo_cmd`` parameter default value due to changes in
    devlib.
  - Various Python 3 related fixes.
  - Ensure plugin names are converted to identifiers internally to act more
    consistently when dealing with names containing ``-``'s etc.
  - Now correctly updates RunInfo with project and run name information.
  - Add versioning support for POD structures with the ability to
    automatically update data structures / formats to new versions.

Commands:
---------
  - Fix revent target initialization.
  - Fix revent argument validation.

Workloads:
----------
  - ``Speedometer``: Close open tabs upon workload completion.
  - ``jankbench``: Ensure that the logcat monitor thread is terminated
    correctly to prevent left over adb processes.
  - UiAutomator workloads are now able to dismiss android warning that a
    workload has not been designed for the latest version of android.

Other:
------
- Report additional metadata about target, including: system_id,
  page_size_kb.
- Uses cache directory to reduce target calls, e.g. will now use cached
  version of TargetInfo if local copy is found.
- Update recommended :ref:`installation <github>` commands when installing from
  github due to pip not following dependency links correctly.
- Fix incorrect parameter names in runtime parameter documentation.


--------------------------------------------------


*************
Version 3.0.0
*************

WA3 is a more or less from-scratch re-write of WA2. We have attempted to
maintain configuration-level compatibility wherever possible (so WA2 agendas
*should* mostly work with WA3), however some breaks are likely and minor tweaks
may be needed.

It terms of the API, WA3 is completely different, and WA2 extensions **will not
work** with WA3 -- they would need to be ported into WA3 plugins.

For more information on migrating from WA2 to WA3 please see the
:ref:`migration-guide`.

Not all of WA2 extensions have been ported for the initial 3.0.0 release. We
have ported the ones we believe to be most widely used and useful. The porting
work will continue, and more of WA2's extensions will be in the future releases.
However, we do not intend to port absolutely everything, as some things we
believe to be no longer useful.

.. note:: If there a particular WA2 extension you would like to see in WA3 that
          is not yet there, please let us know via the GitHub issues. (And, of
          course, we always welcome pull requests, if you have the time to
          do the port yourselves :-) ).

New Features
============

- Python 3 support. WA now runs on both Python 2 and Python 3.

  .. warning:: Python 2 support should now be considered deprecated. Python 2
               will still be fully supported up to the next major release
               (v3.1). After that, Python 2 will be supported for existing
               functionality, however there will be no guarantee that newly
               added functionality would be compatible with Python 2. Support
               for Python 2 will be dropped completely after release v3.2.

- There is a new Output API which can be used to aid in post processing a
  run's output. For more information please see :ref:`output_processing_api`.
- All "augmentations" can now be enabled on a per workload basis (in WA2 this
  was available for instruments, but not result processors).
- More portable runtime parameter specification. Runtime parameters now support
  generic aliases, so instead of specifying ``a73_frequency: 1805000`` in your
  agenda, and then having to modify this for another target, it is now possible
  to specify ``big_frequency: max``.
- ``-c`` option can now be used multiple times to specify several config files
  for a single run, allowing for a more fine-grained configuration management.
- It is now possible to disable all previously configured augmentations from an
  agenda using ``~~``.
- Offline output processing with ``wa process`` command. It is now possible to
  run processors on previously collected WA results, without the need for a
  target connection.
- A lot more metadata is collected as part of the run, including much more
  detailed information about the target, and MD5 hashes of all resources used
  during the run.
- Better ``show`` command. ``wa show`` command now utilizes ``pandoc`` and
  ``man`` to produce easier-to-browse documentation format, and has been
  enhanced to include documentation on general settings, runtime parameters, and
  plugin aliases.
- Better logging. The default ``stdout`` output is now more informative.
  The verbose output is much more detailed. Nested indentation is used for
  different phases of execution to make log output easier to parse visually.
- Full ``ChromeOS`` target support. Including support for the Android container
  apps.
- Implemented on top of devlib_. WA3 plugins can make use of devlib's enhanced
  target API (much richer and more robust than WA2's Device API).
- All-new documentation. The docs have been revamped to be more useful and
  complete.

.. _devlib: https://github.com/ARM-software/devlib

Changes
=======

- Configuration files ``config.py`` are now specified in YAML format in
  ``config.yaml``. WA3 has support for automatic conversion of the default
  config file and will be performed upon first invocation of WA3.
- The "config" and "global" sections in an agenda are now interchangeable so can
  all be specified in a "config" section.
- "Results Processors" are now known as "Output Processors" and can now be ran
  offline.
- "Instrumentation" is now known as "Instruments" for more consistent naming.
- Both "Output Processor" and "Instrument" configuration have been merged into
  "Augmentations" (support for the old naming schemes have been retained for
  backwards compatibility)


