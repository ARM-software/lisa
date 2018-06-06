APIs
====

---------------------

.. _target-info-api:

TargetInfo API
---------------

The :class:`TargetInfo` class is designed to provide an easy way of presenting
various pieces of information about the target device. An instance of this class
will be instantiated and populated automatically from the devlib
`target <http://devlib.readthedocs.io/en/latest/target.html>`_ created
during a WA run and serialized to a json file as part of the metadata exported
by WA at the end of a run.

The available attributes of the class are as follows:

- :confval:`target` - The name of the target that was created. E.g.
  AndroidTarget, LinuxTarget etc.
- :confval:`cpus` - A list of the unique cpu types that have been detected on
  the target. E.g. 'A53', 'A72' etc.
- :confval:`os` - The OS that the target was running.
- :confval:`os_version` - A dict that contains a mapping of OS version elements
  to their values. This mapping is OS-specific.
- :confval:`abi` - The ABI of the target device.
- :confval:`hostname` - The hostname of the the device the run was executed on.
- :confval:`is_rooted` - A boolean value specifying whether root was detected on
  the device.
- :confval:`kernel_version` - The version of the kernel on the target device.
  This returns a :class:`KernelVersion` instance that has separate version and
  release fields.
- :confval:`kernel_config` - A :class:`KernelConfig` instance that contains
  parsed kernel config from the target device. This may be ``None`` if the
  kernel config could not be extracted.
- :confval:`sched_features` - A list of the available tweaks to the scheduler,
  if available from the device.
- :confval:`hostid` - The unique identifier of the particular device the WA run
  was executed on.

---------------------

.. _run-info-api:

Run Info API
------------

The :class:`RunInfo` can be used to provide information about the run in
general. The available attributes of the class are as follows:


- :confval:`uuid` - A unique identifier for that particular run.
- :confval:`run_name` - The name of the run (if provided)
- :confval:`project` - The name of the project the run belongs to (if provided)
- :confval:`project_stage` - The project stage the run is associated with (if provided)
- :confval:`duration` - The length of time the run took to complete.
- :confval:`start_time` - The time the run was stared.
- :confval:`end_time` - The time at which the run finished.
