=================================
What's New in Workload Automation
=================================

-------------
Version 3.0
-------------

WA3 is a re-write of WA2 therefore please note that while backwards compatibility
has attempted to be maintained where possible, there maybe breaking
changes moving from WA2 to WA3.

- Changes:
    - Configuration files ``config.py`` are now specified in YAML format in
      ``config.yaml``. WA3 has support for automatic conversion of the default
      config file and will be performed upon first invocation of WA3.
    - The "config" and "global" sections in an agenda are not interchangeable so can all be specified in a "config" section.
    - "Results Processors" are now known as "Output Processors" and can now be ran offline.
    - "Instrumentation" is now known as "Instruments" for more consistent naming.
    - "Both "Output Processor" and "Instrument" configuration have been merged
      into "Augmentations" (support for the old naming schemes have been
      retained for backwards compatibility)


- New features:
    - There is a new Output API which can be used to aid in post processing a run's output. For more information please see :ref:`output-api`.
    - All "augmentations" can now be enabled on a per workload basis.

For more information on migrating from WA2 to WA3 please see the :ref:`migration-guide`.
