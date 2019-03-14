*********
Workloads
*********

Introduction
============

LISA is not intended to automate the execution of complex workloads such as
Android benchmarks (Jankbench, PCMark...). For this kind of application, you
should turn to `Workload Automation <https://github.com/ARM-software/workload-automation>`_
instead.

However, a minimal amount of support for running experiments is still useful.
That is why LISA provides the wlgen API, which lets you execute commands and
binaries while providing some help for managing tasksets, cgroups, and more.

Available workloads
===================

Most of these workloads are based on the :class:`~lisa.wlgen.workload.Workload`
class, see the documentation for common functionalities.

rt-app
++++++

`rt-app <https://github.com/scheduler-tools/rt-app>`_ lets you run configurable
workloads described in JSON, and is the backbone of our scheduler tests. See
the github page for more information.

Our :class:`~lisa.wlgen.rta.RTA` class is a python wrapper around rt-app, which
lets us abstract away the generation of the JSON files. On top of that, the
:class:`~lisa.wlgen.rta.RTATask` class (and its children) facilitate the workload
description. For instance, creating a workload with 4 tasks with a 50% duty
cycle is as simple as this::

   profile = {}

   for i in range(4):
       profile["task_{}".format(i)] = Periodic(duty_cycle_pct=50)

   wload = RTA.by_profile(target, "4_50_tasks", profile)
   wload.run()

See the :class:`~lisa.wlgen.rta.RTA` & :class:`~lisa.wlgen.rta.RTATask`
documentations for more details.

Sysbench
++++++++

Sysbench is a useful workload to get some performance numbers, e.g. to
assert that higher frequencies lead to more work done
(as done in :class:`~lisa.tests.cpufreq.sanity.UserspaceSanity`).

See :class:`~lisa.wlgen.sysbench.Sysbench` for more details.

Target script
+++++++++++++

The :class:`~lisa.target_script.TargetScript` class lets you define a Bash
script to be later executed on the target.

It is useful when you want generate some activity on the target without the
overhead of an ssh/adb connection.

API
===

Base class
++++++++++

.. automodule:: lisa.wlgen.workload
   :members:

rt-app
++++++

.. autoclass:: lisa.wlgen.rta.RTA
   :members:

.. autoclass:: lisa.wlgen.rta.Phase
   :members:

.. autoclass:: lisa.wlgen.rta.RTATask
   :members:

.. automodule:: lisa.wlgen.rta
   :exclude-members: RTA, RTATask, Phase
   :members:

.. automodule:: lisa.analysis.rta
   :members:

Sysbench
++++++++

.. automodule:: lisa.wlgen.sysbench
   :members:

Target script
+++++++++++++

.. automodule:: lisa.target_script
   :members:
